"""五條 pipeline 在無標註測試集上的並排目視比較。

testset/ 沒有標註，算不出任何指標；此處純為目視。每張輸出一列：
原圖 │ ①Mask R-CNN │ ②HBB→HRNet │ ③YOLO-seg │ ④OBB→HRNet │ ⑤YOLO-OBB→HRNet

各條 pipeline 的推論設定與量化評估完全一致：門檻 0.35、padding 0.2、
clean_mask、單模型、無 TTA。此處用 fold0 的權重（測試集未參與任何訓練，
五折權重皆有效；固定用同一折是為了讓五條的比較條件一致）。

用法：
    py scripts/vis_testset_all.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_holdout_all import NAMES, predict_one  # noqa: E402
from train_maskrcnn import ROOT  # noqa: E402

TEST, OUT = ROOT / "testset", ROOT / "testset_vis_all"
TILE_W = 460
TITLES = ["Original", "(1) Mask R-CNN", "(2) MaskRCNN>HBB>HRNet",
          "(3) YOLO11-seg", "(4) MaskRCNN>OBB>HRNet", "(5) YOLO-OBB>OBB>HRNet"]
COLORS = [(90, 220, 120), (90, 200, 250), (250, 190, 90), (200, 140, 250), (120, 120, 255)]


def tile(gray, masks, title, color):
    v = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for m in masks:
        cnts = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)[0]
        cv2.drawContours(v, cnts, -1, color, 5)
    v = cv2.resize(v, (TILE_W, int(TILE_W * v.shape[0] / v.shape[1])))
    bar = np.full((44, TILE_W, 3), 26, np.uint8)
    cv2.putText(bar, f"{title}  n={len(masks)}", (9, 30), 0, 0.52, (245, 245, 245), 1,
                cv2.LINE_AA)
    return np.vstack([bar, v])


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--only", default="", help="逗號分隔的檔名主幹，例如 121,142")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    keep = set(args.only.split(",")) if args.only else None
    files = sorted(f for f in TEST.glob("*.jpg") if keep is None or f.stem in keep)

    for f in files:
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        hw = gray.shape
        tiles = [tile(gray, [], TITLES[0], (0, 0, 0))]
        counts = []
        for v in (1, 2, 3, 4, 5):
            pred, _ = predict_one(v, args.fold, gray, f, args.thr, False, hw)
            counts.append(len(pred))
            tiles.append(tile(gray, list(pred), TITLES[v], COLORS[v - 1]))
        h = max(t.shape[0] for t in tiles)
        tiles = [np.vstack([t, np.full((h - t.shape[0], t.shape[1], 3), 26, np.uint8)])
                 if t.shape[0] < h else t for t in tiles]
        sep = np.full((h, 5, 3), 26, np.uint8)
        grid = tiles[0]
        for t in tiles[1:]:
            grid = np.hstack([grid, sep, t])
        cv2.imwrite(str(OUT / f"{f.stem}.png"), grid)
        print(f"{f.stem}: 偵測數 ①{counts[0]} ②{counts[1]} ③{counts[2]} "
              f"④{counts[3]} ⑤{counts[4]}", flush=True)
    print(f"\n→ {OUT}/")


if __name__ == "__main__":
    main()
