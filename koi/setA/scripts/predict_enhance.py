"""每張測試圖產生一張「所有影像增強方法」的對照圖。

每一格是：該方法增強後的影像 + **用該方法自己訓練出來的權重**跑出的預測。
所以格與格之間的差異，就是影像處理造成的差異——訓練資料、fold 切分、超參數
全部相同，只有輸入影像的處理方式不同。

    --model maskrcnn   每格用 koi/checkpoints/<方法>/maskrcnn_fold<k>.pt
    --model sam2       每格用同一份 SAM 2 權重（它零訓練），輸入是該方法增強過的
                       影像。box 預設**全部用 original 的 Mask R-CNN 提供**：
                       SAM 2 沒有 per-method 權重，若讓每格各自用該方法的偵測器，
                       格與格的差異就混進了偵測器的變異，看不出增強對 SAM 2 的
                       單獨影響。box 固定、只變輸入，才是乾淨的對照。
                       要改用各方法自己的偵測器，加 --boxes matched。

尚未訓練完成的方法會顯示 "no weights" 並留白，不會中斷——這樣訓練途中就能先看
已完成的部分。

用法：
    py koi/scripts/predict_enhance.py
    py koi/scripts/predict_enhance.py --model sam2
    py koi/scripts/predict_enhance.py --only 121,142
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
from enhance import METHODS  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import CKPT, build_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "testset"
COLS = 4
TILE_W = 460
COLORS = [(0, 255, 80), (80, 160, 255), (255, 90, 255), (60, 255, 255),
          (255, 200, 60), (140, 255, 180)]


def panel(img: np.ndarray, masks, title: str) -> np.ndarray:
    v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if masks is not None and len(masks):
        ov = v.copy()
        for i, m in enumerate(masks):
            c = COLORS[i % len(COLORS)]
            ov[m] = c
            cv2.drawContours(v, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)[0], -1, c, 4)
        v = cv2.addWeighted(ov, 0.25, v, 0.75, 0)
    v = cv2.resize(v, (TILE_W, int(TILE_W * v.shape[0] / v.shape[1])))
    n = "-" if masks is None else str(len(masks))
    b = np.zeros((46, v.shape[1], 3), np.uint8)
    cv2.putText(b, f"{title}  ({n})", (10, 32), 0, 0.72, (255, 255, 255), 2)
    return np.vstack([b, v])


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["maskrcnn", "sam2"], default="maskrcnn")
    ap.add_argument("--fold", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--only", default="")
    ap.add_argument("--boxes", choices=["original", "matched"], default="original",
                    help="sam2 模式的 box 來源；maskrcnn 模式此參數無效")
    args = ap.parse_args()

    out_dir = ROOT / f"enhance_pred_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    for name in METHODS:
        f = CKPT / name / f"maskrcnn_fold{args.fold}.pt"
        if not f.exists():
            continue
        m = build_model(False)
        m.load_state_dict(torch.load(f, map_location="cpu", weights_only=False)["model"])
        m.eval()
        models[name] = m
    print(f"已載入 {len(models)}/{len(METHODS)} 種方法的權重: {', '.join(models) or '無'}")

    sam = None
    if args.model == "sam2":
        from ultralytics import SAM
        sam = SAM(weight("sam2.1_b.pt"))

    wanted = {f"{s.strip()}.jpg" for s in args.only.split(",") if s.strip()}
    files = [f for f in sorted(TESTSET.glob("*.jpg"), key=lambda p: int(p.stem))
             if not wanted or f.name in wanted]
    tmp = ROOT / "_pred_tmp.png"

    for f in files:
        gray0 = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        h, w = gray0.shape
        tiles = []
        for name, fn in METHODS.items():
            g = fn(gray0)
            # box 來源：maskrcnn 模式必為該方法自己；sam2 模式預設固定用 original
            det = name if (args.model == "maskrcnn" or args.boxes == "matched") else "original"
            if det not in models:
                tiles.append(panel(g, None, f"{name}  no weights"))
                continue
            src = g if det == name else gray0
            t = torch.from_numpy(src).float().div(255).unsqueeze(0).repeat(3, 1, 1)
            o = models[det]([t])[0]
            k = o["scores"].numpy() >= args.conf
            boxes = o["boxes"].numpy()[k]
            if args.model == "maskrcnn":
                ms = [clean_mask(x) for x in o["masks"].numpy()[k, 0] > 0.5]
            elif len(boxes) == 0:
                ms = []
            else:
                cv2.imwrite(str(tmp), g)
                r = sam.predict(str(tmp), bboxes=boxes.tolist(), verbose=False)[0]
                d = r.masks.data.cpu().numpy() > 0.5
                d = (np.stack([cv2.resize(x.astype(np.uint8), (w, h),
                                          interpolation=cv2.INTER_NEAREST).astype(bool) for x in d])
                     if d.shape[1:] != (h, w) else d)
                ms = [clean_mask(x) for x in d]
            tiles.append(panel(g, ms, name))

        H = max(t.shape[0] for t in tiles)
        tiles = [np.vstack([t, np.zeros((H - t.shape[0], t.shape[1], 3), np.uint8)]) for t in tiles]
        rows = []
        for i in range(0, len(tiles), COLS):
            row = tiles[i:i + COLS]
            while len(row) < COLS:
                row.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(row))
        cv2.imwrite(str(out_dir / f"{f.stem}.png"), np.vstack(rows))
        print(f"  {f.name} → {out_dir.name}/{f.stem}.png")

    tmp.unlink(missing_ok=True)
    print(f"\n完成 {len(files)} 張 → {out_dir}")


if __name__ == "__main__":
    main()
