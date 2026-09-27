"""在未標註的測試影像上跑三個模型，輸出四格對照圖。

每張輸入產生一張輸出，由左至右：

    Original │ Mask R-CNN │ SAM 2 │ MedSAM

每格上的數字意義不同，標題已註明：Mask R-CNN 印的是**偵測信心**（這裡有沒有一顆
牙），兩個 SAM 印的是**預測 IoU**（模型自評遮罩品質）。SAM 不做偵測——給一個框
就一定回一個遮罩，從不拒絕——所以它們沒有偵測信心可言。

ROI 的來源
----------
測試影像沒有標註，所以 SAM 2 與 MedSAM 沒有 GT box 可以當 prompt。這裡用
**Mask R-CNN 預測的 box** 餵給兩個 SAM，於是三格是「同一個 ROI、三種分割器」
的對照——正好就是第 2、3 條 pipeline 的結構：前面一個偵測器決定框，後面一個
模型負責把邊界切細。

這也代表：如果 Mask R-CNN 漏了一顆牙，另外兩個模型也不會有那顆牙。三格的
偵測結果一定相同，差別只在遮罩的形狀。

沒有標註就沒有指標
------------------
這批圖只能目視比較，不會輸出 Dice 或 HD95——沒有 ground truth 就沒有分母。
要量化必須回到 koi/annotations 那 25 張的 5-fold 評估。

用法：
    py koi/scripts/predict_testset.py
    py koi/scripts/predict_testset.py --fold 2 --conf 0.5
    py koi/scripts/predict_testset.py --only 101,113
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weights import weight  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import CKPT, build_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "testset"
OUT = ROOT / "testset_vis"

PANEL_W = 620          # 每格縮放後的寬度
COLORS = [(0, 255, 80), (80, 160, 255), (255, 90, 255), (60, 255, 255),
          (255, 200, 60), (140, 255, 180)]


def draw(gray: np.ndarray, masks: np.ndarray, scores, title: str) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()
    for i, m in enumerate(masks):
        c = COLORS[i % len(COLORS)]
        overlay[m] = c
        cv2.drawContours(vis, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)[0], -1, c, 4)
    vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
    for i, m in enumerate(masks):
        if scores is not None and len(scores) > i:
            ys, xs = np.nonzero(m)
            if len(xs):
                cv2.putText(vis, f"{scores[i]:.2f}", (int(xs.min()), max(34, int(ys.min()) - 10)),
                            0, 1.1, COLORS[i % len(COLORS)], 3)
    banner = np.zeros((58, vis.shape[1], 3), np.uint8)
    cv2.putText(banner, f"{title}  ({len(masks)})", (14, 42), 0, 1.0, (255, 255, 255), 2)
    return np.vstack([banner, vis])


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0,
                    help="用哪個 fold 的 Mask R-CNN 權重，預設 0（val Dice 最高）")
    ap.add_argument("--tag", default="original", help="checkpoints/ 下的子資料夾")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="由 sweep_conf.py 掃出的 Mask R-CNN 門檻")
    ap.add_argument("--only", default="", help="只跑指定影像，逗號分隔，例如 101,113")
    args = ap.parse_args()

    from transformers import SamModel, SamProcessor
    from ultralytics import SAM

    print("載入模型…")
    rcnn = build_model(False)
    rcnn.load_state_dict(torch.load(CKPT / args.tag / f"maskrcnn_fold{args.fold}.pt",
                                    map_location="cpu", weights_only=False)["model"])
    rcnn.eval()
    sam2 = SAM(weight("sam2.1_b.pt"))
    med = SamModel.from_pretrained("wanglab/medsam-vit-base").eval()
    med_proc = SamProcessor.from_pretrained("wanglab/medsam-vit-base")

    OUT.mkdir(parents=True, exist_ok=True)
    wanted = {f"{s.strip()}.jpg" for s in args.only.split(",") if s.strip()}
    files = sorted(TESTSET.glob("*.jpg"), key=lambda p: int(p.stem))
    files = [f for f in files if not wanted or f.name in wanted]

    for f in files:
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape

        # --- Mask R-CNN：偵測 + 分割，它的 box 同時是另外兩個模型的 prompt ---
        img = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1)
        out = rcnn([img])[0]
        keep = out["scores"].numpy() >= args.conf
        rcnn_m = np.array([clean_mask(m) for m in out["masks"].numpy()[keep, 0] > 0.5],
                          bool).reshape(-1, h, w)
        boxes = out["boxes"].numpy()[keep]
        scores = out["scores"].numpy()[keep]

        panels = [draw(gray, np.zeros((0, h, w), bool), None, "Original"),
                  draw(gray, rcnn_m, scores, "Mask R-CNN  det-conf")]

        if len(boxes) == 0:
            empty = np.zeros((0, h, w), bool)
            panels += [draw(gray, empty, None, "SAM 2  [box from Mask R-CNN]"),
                       draw(gray, empty, None, "MedSAM  [box from Mask R-CNN]")]
        else:
            r = sam2.predict(str(f), bboxes=boxes.tolist(), verbose=False)[0]
            m = r.masks.data.cpu().numpy() > 0.5
            sam2_m = (np.stack([cv2.resize(x.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                                for x in m]) if m.shape[1:] != (h, w) else m)
            sam_iou = r.boxes.conf.cpu().numpy() if r.boxes is not None else None
            sam2_m = np.array([clean_mask(m) for m in sam2_m], bool).reshape(-1, h, w)
            panels.append(draw(gray, sam2_m, sam_iou, "SAM 2  pred-IoU  [box:R-CNN]"))

            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            inp = med_proc(rgb, input_boxes=[boxes.tolist()], return_tensors="pt")
            mo = med(**inp, multimask_output=False)
            mm = med_proc.image_processor.post_process_masks(
                mo.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
            med_m = np.array([clean_mask(m) for m in mm.squeeze(1).numpy().astype(bool)],
                             bool).reshape(-1, h, w)
            panels.append(draw(gray, med_m,
                               mo.iou_scores.squeeze(0).squeeze(-1).numpy(),
                               "MedSAM  pred-IoU  [box:R-CNN]"))

        panels = [cv2.resize(p, (PANEL_W, int(PANEL_W * p.shape[0] / p.shape[1]))) for p in panels]
        H = max(p.shape[0] for p in panels)
        panels = [np.vstack([p, np.zeros((H - p.shape[0], p.shape[1], 3), np.uint8)]) for p in panels]
        sep = np.full((H, 4, 3), 90, np.uint8)
        grid = panels[0]
        for p in panels[1:]:
            grid = np.hstack([grid, sep, p])
        cv2.imwrite(str(OUT / f"{f.stem}.png"), grid)
        print(f"  {f.name}  偵測到 {len(boxes)} 顆 → {OUT.name}/{f.stem}.png")

    print(f"\n完成 {len(files)} 張 → {OUT}")


if __name__ == "__main__":
    main()
