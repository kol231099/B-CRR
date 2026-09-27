#!/usr/bin/env python3
"""把 BPR 套在某個模型的驗證集預測上，重算指標。

粗糙遮罩來自 bpr_dump.py --split val（out-of-fold），精修網路來自同一個 fold，
兩者都沒看過該 fold 的驗證影像。

用法
    py scripts/eval_bpr.py --model unetpp_resnet34_tta
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpr import build_refiner, refine  # noqa: E402
from metrics import FIELDS, match  # noqa: E402
from train_maskrcnn import ANN, CKPT, ROOT  # noqa: E402
from train_seg2 import CROPS  # noqa: E402

BPR = ROOT / "bpr"
EVAL = ROOT / "eval"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unetpp_resnet34_tta")
    args = ap.parse_args()

    manifest = {r["crop_id"]: r for r in csv.DictReader((CROPS / "manifest.csv").open(encoding="utf-8"))}
    coco = json.loads((ANN / "instances_all.json").read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    id_of_name = {v["file_name"]: k for k, v in imgs.items()}
    per: dict[int, list] = {}
    for a in coco["annotations"]:
        per.setdefault(a["image_id"], []).append(a)
    order = {}
    for iid, lst in per.items():
        for i, a in enumerate([x for x in lst if not x["iscrowd"]]):
            order[(iid, a["id"])] = i

    nets = {}
    for f in range(5):
        ck = CKPT / "bpr" / f"fold{f}.pt"
        if ck.exists():
            n = build_refiner()
            n.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
            n.eval()
            nets[f] = n
    if not nets:
        sys.exit("找不到 BPR 檢查點，請先跑 train_bpr.py")

    rows_before, rows_after = [], []
    src = BPR / args.model / "val"
    for p in sorted(src.glob("*.npz")):
        d = np.load(p)
        fold = int(d["fold"])
        if fold not in nets:
            continue
        img, coarse, gt = d["img"], d["coarse"].astype(bool), d["gt"].astype(bool)
        if img.shape != coarse.shape:
            import cv2
            img = cv2.resize(img, coarse.shape[::-1], interpolation=cv2.INTER_AREA)
        fine = refine(nets[fold], img, coarse)

        r = manifest[p.stem]
        gi = order[(id_of_name[r["image"]], int(r["ann_id"]))]
        for tgt, m in ((rows_before, coarse), (rows_after, fine)):
            rr = match(m[None], np.array([1.0]), gt[None], r["image"])[0]
            for x in rr:
                x["gt_idx"] = gi
            tgt += rr

    EVAL.mkdir(exist_ok=True)
    # 前後都要存：配對檢定必須拿同一顆牙的兩個數字比，只存精修後就只能比分布。
    out = EVAL / f"bpr_{args.model}.csv"
    for path, rs in ((EVAL / f"bpr_{args.model}_before.csv", rows_before), (out, rows_after)):
        with path.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(rs)

    for lab, rs in (("精修前", rows_before), ("精修後", rows_after)):
        tp = [r for r in rs if r["kind"] == "TP"]
        b = np.array([r["biou"] for r in tp])
        h = np.array([r["hd95"] for r in tp])
        dc = np.array([r["dice"] for r in tp])
        print(f"  {args.model} {lab}　n={len(tp)}　Dice {np.median(dc):.4f}　"
              f"B-IoU {np.median(b):.4f}　HD95 {np.median(h):5.1f}", flush=True)
    print(f"    → {out.name}", flush=True)


if __name__ == "__main__":
    main()
