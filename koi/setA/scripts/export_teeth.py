"""把每張影像的每顆牙輸出成四階段圖，一張影像一個資料夾。

這是整條流程的最後一步：從「模型知道牙齒在哪」到「拿到一張只有那顆牙的影像」，
供下游的 CRR 量測使用。四個階段對應四張圖：

    1_original   整張原圖
    2_detected   整張原圖疊上所有偵測到的牙齒，每顆一色並標註信心分數
每顆牙再各自三張，前綴 tooth1、tooth2……由左至右編號：

    tooth{n}_1_roi      該顆牙的 ROI（bbox 外擴 20%），仍是原始影像內容
    tooth{n}_2_outline  同一個 ROI，畫上模型預測的牙齒輪廓（尚未切出）
    tooth{n}_3_cutout   只保留輪廓內的像素，背景透明，裁到遮罩的最小外接矩形

同時輸出 tooth{n}.json —— 給下游 CRR 用
----------------------------------------
CRR_PA/scripts/ 的 find_cej.py、find_ridge.py、find_axis_paper.py 吃的是
**labelme 格式的 .json 加上整張原圖**，不是切好的 PNG：

    data = load_annotation(jf); img = load_image(data)
    mask = label_mask(data, "1"); points = mask_points(mask)

find_ridge 還需要原始灰階影像來算殘差，光有遮罩不夠。而且 label_mask 會把同一個
json 裡所有 label "1" 的多邊形**合併**成一張遮罩，所以一顆牙必須一個 json。

因此每顆牙額外輸出一個 tooth{n}.json：單一多邊形、label "1"、座標在**原圖座標系**、
imagePath 指向同資料夾的 1_original.png。下游可直接執行：

    py CRR_PA/scripts/find_cej.py koi/test_output/101

一張影像多顆牙
--------------
根尖片每張有 2–4 顆牙，1_original 與 2_detected 是整張影像的，每個資料夾各一張；tooth{n}_* 則是每顆牙
各一組。編號各自從 1 起算——資料夾層級的 1、2 指流程階段，牙齒層級的 1、2、3 指
該顆牙的三個步驟。

為什麼 4_cutout 用透明背景而不是黑底
------------------------------------
黑色在根尖片裡是有意義的（口腔空隙、影像邊界），用黑色填背景會讓下游無法分辨
「這裡是背景」與「這裡是暗的組織」。RGBA 的 alpha 通道沒有這個歧義。

用法：
    py koi/scripts/export_teeth.py
    py koi/scripts/export_teeth.py --only 101,142
    py koi/scripts/export_teeth.py --src images     # 改用有標註的 63 張
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import CKPT, ROOT, build_model  # noqa: E402

# 輸出放在資料根目錄的**外面**（koi/test_output），與 setA、baselines 平行——
# 它是要交給下游的產出，不屬於任何一組實驗。
OUT = ROOT.parent / "test_output"
PAD = 0.20
OUTLINE = (0, 255, 80)
# 每顆牙一色，與 testset 對照圖一致，方便交叉比對
COLORS = [(255, 90, 255), (60, 255, 255), (80, 160, 255), (0, 255, 80),
          (255, 200, 60), (140, 255, 180)]


def roi_box(mask: np.ndarray, w: int, h: int, pad: float) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return (max(0, int(x0 - px)), max(0, int(y0 - py)),
            min(w, int(x1 + px)), min(h, int(y1 + py)))


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="testset", help="來源資料夾，預設 testset")
    ap.add_argument("--tag", default="original", help="checkpoints/ 下的子資料夾")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--pad", type=float, default=PAD)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    ck = torch.load(CKPT / args.tag / f"maskrcnn_fold{args.fold}.pt",
                    map_location="cpu", weights_only=False)
    model = build_model(False, ck.get("mask_res", 28))
    model.load_state_dict(ck["model"])
    model.eval()

    src = ROOT / args.src
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    files = [f for f in sorted(src.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
             if not wanted or f.stem in wanted]

    total = 0
    for f in files:
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        t = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1)
        out = model([t])[0]
        keep = out["scores"].numpy() >= args.conf
        masks = [clean_mask(m) for m in out["masks"].numpy()[keep, 0] > 0.5]
        scores = out["scores"].numpy()[keep]

        # 由左至右編號，讓 tooth1/tooth2 對應到看得懂的位置
        order = sorted(range(len(masks)), key=lambda i: np.nonzero(masks[i])[1].mean() if masks[i].any() else 0)

        d = OUT / f.stem
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.png"):
            old.unlink()
        cv2.imwrite(str(d / "1_original.png"), gray)

        # 全圖偵測結果：每顆牙一色，半透明填色 + 輪廓 + 信心分數
        vis_all = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        fill = vis_all.copy()
        for n, i in enumerate(order, 1):
            c = COLORS[(n - 1) % len(COLORS)]
            fill[masks[i]] = c
            cv2.drawContours(vis_all, cv2.findContours(masks[i].astype(np.uint8),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], -1, c, 3)
        vis_all = cv2.addWeighted(fill, 0.28, vis_all, 0.72, 0)
        for n, i in enumerate(order, 1):
            ys, xs = np.nonzero(masks[i])
            if len(xs):
                cv2.putText(vis_all, f"{n}: {scores[i]:.2f}", (int(xs.min()), max(30, int(ys.min()) - 10)),
                            0, 0.9, COLORS[(n - 1) % len(COLORS)], 2)
        cv2.imwrite(str(d / "2_detected.png"), vis_all)

        for n, i in enumerate(order, 1):
            m = masks[i]
            if not m.any():
                continue
            x0, y0, x1, y1 = roi_box(m, w, h, args.pad)

            roi = gray[y0:y1, x0:x1]
            cv2.imwrite(str(d / f"tooth{n}_1_roi.png"), roi)

            vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            cnt, _ = cv2.findContours(m[y0:y1, x0:x1].astype(np.uint8),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnt, -1, OUTLINE, 2)
            cv2.imwrite(str(d / f"tooth{n}_2_outline.png"), vis)

            # 切出：裁到遮罩本身的外接矩形（不含 padding），背景 alpha = 0
            ys, xs = np.nonzero(m)
            cy0, cy1, cx0, cx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            sub = gray[cy0:cy1, cx0:cx1]
            alpha = (m[cy0:cy1, cx0:cx1] * 255).astype(np.uint8)
            cv2.imwrite(str(d / f"tooth{n}_3_cutout.png"),
                        cv2.merge([sub, sub, sub, alpha]))

            # 給下游的 labelme json：一顆牙一個檔，座標維持在原圖座標系
            cnt_full, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            poly = max(cnt_full, key=cv2.contourArea).reshape(-1, 2)
            (d / f"tooth{n}.json").write_text(json.dumps({
                "version": "5.2.1", "flags": {},
                "shapes": [{"label": "1", "points": poly.astype(float).tolist(),
                            "group_id": None, "shape_type": "polygon", "flags": {}}],
                "imagePath": "1_original.png", "imageData": None,
                "imageHeight": h, "imageWidth": w,
            }, ensure_ascii=False), "utf-8")
            total += 1

        print(f"  {f.name}  {len(order)} 顆 → test_output/{f.stem}/  "
              f"(信心 {', '.join(f'{scores[i]:.2f}' for i in order)})")

    print(f"\n完成 {len(files)} 張影像、{total} 顆牙 → {OUT}")


if __name__ == "__main__":
    main()
