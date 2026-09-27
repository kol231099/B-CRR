"""在保留測試集上評估最終模型。

與 5-fold 的差別
----------------
5-fold 回答「這個方法有多好」——它用同一批資料反覆切分，樣本數大（155 顆牙）但
所有影像都參與過訓練。保留測試集回答「這個模型在全新資料上有多好」——樣本小
（26 顆牙）但完全獨立，模型從未以任何形式接觸過。

兩者互補，都該報告。若兩個數字接近，代表泛化良好。

樣本數只有 26 顆，點估計的不確定性很大，因此一併輸出 bootstrap 信賴區間——
只報一個數字會讓人高估它的精確度。

用法：
    py scripts/eval_holdout.py --tta
    py scripts/eval_holdout.py --ckpt maskrcnn_final.pt --conf 0.35
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import FIELDS, match, summarize  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, CKPT, ROOT, build_model  # noqa: E402
from tta import predict_tta  # noqa: E402

EVAL = ROOT / "eval"


def boot_ci(x: np.ndarray, stat=np.median, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """bootstrap 95% 信賴區間。26 顆牙的點估計不確定性大，必須一併報告。"""
    rng = np.random.default_rng(seed)
    vals = [stat(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="maskrcnn_final.pt")
    ap.add_argument("--tag", default="original")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    ck = torch.load(CKPT / args.tag / args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(False, ck.get("mask_res", 28))
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"權重 {args.ckpt}　訓練影像 {ck.get('n_train', '?')} 張　TTA {'開' if args.tta else '關'}")

    coco = json.loads((ANN / "holdout.json").read_text(encoding="utf-8"))
    by: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by.setdefault(a["image_id"], []).append(a)

    rows = []
    for im in coco["images"]:
        h, w = im["height"], im["width"]
        gray = cv2.imread(str(ROOT / "holdout" / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        gt = np.stack([cv2.fillPoly(np.zeros((h, w), np.uint8),
                                    [np.array(a["segmentation"][0], np.int32).reshape(-1, 2)], 1).astype(bool)
                       for a in by.get(im["id"], [])])
        if args.tta:
            prob, _, scores = predict_tta(model, gray, args.conf)
            pred = np.array([clean_mask(m) for m in prob > 0.5], bool).reshape(-1, h, w)
        else:
            t = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1)
            o = model([t])[0]
            k = o["scores"].numpy() >= args.conf
            scores = o["scores"].numpy()[k]
            pred = np.array([clean_mask(m) for m in o["masks"].numpy()[k, 0] > 0.5],
                            bool).reshape(-1, h, w)
        rows += match(pred, scores, gt, im["file_name"])[0]

    suffix = "_tta" if args.tta else ""
    EVAL.mkdir(parents=True, exist_ok=True)
    with (EVAL / f"holdout{suffix}.csv").open("w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=FIELDS)
        w_.writeheader()
        w_.writerows(rows)

    summarize(rows, f"保留測試集{'（+TTA）' if args.tta else ''}")
    tp = [r for r in rows if r["kind"] == "TP"]
    d = np.array([float(r["dice"]) for r in tp])
    hh = np.array([float(r["hd95"]) for r in tp])
    dl, dh = boot_ci(d)
    hl, hhi = boot_ci(hh)
    print(f"  Dice 中位 95% CI  [{dl:.4f}, {dh:.4f}]")
    print(f"  HD95 中位 95% CI  [{hl:.1f}, {hhi:.1f}] px")
    print(f"  → {EVAL}/holdout{suffix}.csv")


if __name__ == "__main__":
    main()
