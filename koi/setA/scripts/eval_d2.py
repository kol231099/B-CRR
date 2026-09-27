#!/usr/bin/env python3
"""評估 detectron2 產出的遮罩，用的是專案既有的 metrics.py。

detectron2 跑在獨立 venv（torch 2.1.2 / Python 3.11），只負責產出遮罩 .npz；
指標一律在主環境用同一份 match() 計算，兩邊的數字才可比。

    py scripts/eval_d2.py --arch pointrend --split holdout
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import FIELDS, match  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, ROOT  # noqa: E402

D2OUT = ROOT / "d2out"
EVAL = ROOT / "eval"


def gt_masks(anns: list, h: int, w: int) -> np.ndarray:
    ms = []
    for a in anns:
        m = np.zeros((h, w), np.uint8)
        for poly in a["segmentation"]:
            cv2.fillPoly(m, [np.array(poly, np.int32).reshape(-1, 2)], 1)
        ms.append(m.astype(bool))
    return np.stack(ms) if ms else np.zeros((0, h, w), bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="pointrend")
    ap.add_argument("--split", default="holdout")
    args = ap.parse_args()

    src = D2OUT / f"pred_{args.arch}_{args.split}"
    if not src.exists():
        sys.exit(f"找不到 {src}，請先在 d2env 裡跑 predict_d2.py")

    coco = json.loads((ANN / f"{args.split}.json").read_text(encoding="utf-8"))
    imgs = {i["file_name"]: i for i in coco["images"]}
    per: dict[str, list] = {}
    id2name = {i["id"]: i["file_name"] for i in coco["images"]}
    for a in coco["annotations"]:
        if not a.get("iscrowd", 0):
            per.setdefault(id2name[a["image_id"]], []).append(a)

    rows = []
    for p in sorted(src.glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        name = str(d["image"])
        if name not in per:
            continue
        im = imgs[name]
        h, w = im["height"], im["width"]
        gt = gt_masks(per[name], h, w)
        pred = d["masks"].astype(bool)
        pred = np.array([clean_mask(m) for m in pred], bool) if len(pred) \
            else np.zeros((0, h, w), bool)
        rows += match(pred, d["scores"], gt, name)[0]

    EVAL.mkdir(exist_ok=True)
    out = EVAL / f"d2_{args.arch}_{args.split}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)

    tp = [r for r in rows if r["kind"] == "TP"]
    fp = sum(1 for r in rows if r["kind"] == "FP")
    fn = sum(1 for r in rows if r["kind"] == "FN")
    g = lambda k: np.median([float(r[k]) for r in tp])
    print(f"  {args.arch} [{args.split}]　TP {len(tp)}　FP {fp}　FN {fn}")
    print(f"    Dice {g('dice'):.4f}　B-IoU {g('biou'):.4f}　HD95 {g('hd95'):.1f}　"
          f"ASSD {g('assd'):.2f}　NSD@3 {g('nsd3'):.4f}")
    print(f"    → {out.name}")


if __name__ == "__main__":
    main()
