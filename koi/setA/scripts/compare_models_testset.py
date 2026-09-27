"""在測試集上並排比較兩個 Mask R-CNN 模型，看多標的資料改變了什麼。

輸出每張測試圖一張：Original │ 模型A │ 模型B

這是唯一乾淨的比較方式。兩個模型的 5-fold 切分不同、驗證集是不同的牙，所以
val Dice 不能直接對比；只有讓它們跑**同一批從未見過的影像**才是公平的。
測試集沒有標註，所以算不出 Dice——這裡是目視比較，量化仍需標註。

用法：
    py koi/scripts/compare_models_testset.py
    py koi/scripts/compare_models_testset.py --only 121,142
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import build_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTSET, OUT = ROOT / "testset", ROOT / "model_compare"
TILE_W = 620
COLORS = [(0, 255, 80), (80, 160, 255), (255, 90, 255), (60, 255, 255),
          (255, 200, 60), (140, 255, 180)]

MODELS = [
    ("Mask R-CNN  25 imgs / 57 teeth", ROOT / "archive_25img/checkpoints/original/maskrcnn_fold2.pt"),
    ("Mask R-CNN  63 imgs / 155 teeth", ROOT / "checkpoints/original/maskrcnn_fold0.pt"),
]


def panel(gray: np.ndarray, masks, scores, title: str) -> np.ndarray:
    v = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if masks is not None and len(masks):
        ov = v.copy()
        for i, m in enumerate(masks):
            c = COLORS[i % len(COLORS)]
            ov[m] = c
            cv2.drawContours(v, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)[0], -1, c, 4)
        v = cv2.addWeighted(ov, 0.25, v, 0.75, 0)
        for i, m in enumerate(masks):
            ys, xs = np.nonzero(m)
            if len(xs) and scores is not None:
                cv2.putText(v, f"{scores[i]:.2f}", (int(xs.min()), max(34, int(ys.min()) - 10)),
                            0, 1.1, COLORS[i % len(COLORS)], 3)
    v = cv2.resize(v, (TILE_W, int(TILE_W * v.shape[0] / v.shape[1])))
    n = "-" if masks is None else str(len(masks))
    b = np.zeros((52, v.shape[1], 3), np.uint8)
    cv2.putText(b, f"{title}  ({n})", (10, 36), 0, 0.78, (255, 255, 255), 2)
    return np.vstack([b, v])


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    nets = []
    for label, path in MODELS:
        if not path.exists():
            raise FileNotFoundError(path)
        m = build_model(False)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
        m.eval()
        nets.append((label, m))
        print(f"已載入 {label}")

    OUT.mkdir(parents=True, exist_ok=True)
    wanted = {f"{s.strip()}.jpg" for s in args.only.split(",") if s.strip()}
    files = [f for f in sorted(TESTSET.glob("*.jpg"), key=lambda p: int(p.stem))
             if not wanted or f.name in wanted]

    for f in files:
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        t = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1)
        tiles = [panel(gray, None, None, "Original")]
        counts = []
        for label, m in nets:
            o = m([t])[0]
            k = o["scores"].numpy() >= args.conf
            ms = [clean_mask(x) for x in o["masks"].numpy()[k, 0] > 0.5]
            tiles.append(panel(gray, ms, o["scores"].numpy()[k], label))
            counts.append(len(ms))

        H = max(x.shape[0] for x in tiles)
        tiles = [np.vstack([x, np.zeros((H - x.shape[0], x.shape[1], 3), np.uint8)]) for x in tiles]
        sep = np.full((H, 5, 3), 90, np.uint8)
        grid = tiles[0]
        for x in tiles[1:]:
            grid = np.hstack([grid, sep, x])
        cv2.imwrite(str(OUT / f"{f.stem}.png"), grid)
        print(f"  {f.name}  25張偵測 {counts[0]} 顆 / 63張偵測 {counts[1]} 顆")

    print(f"\n完成 {len(files)} 張 → {OUT}")


if __name__ == "__main__":
    main()
