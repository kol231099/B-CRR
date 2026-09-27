#!/usr/bin/env python3
"""用 fold0 的精修網路在 18 張保留測試集上做 BPR 前後對照，並輸出比對圖。

保留測試集不屬於任何 fold，精修網路（用五折的訓練影像訓練）沒看過它們，
因此拿 fold0 一個網路來測全部模型沒有洩漏。

基礎模型的用法與主表一致：第二階段取五折機率平均，Mask R-CNN 用
maskrcnn_final.pt（63 張全部訓練）。

用法
    py scripts/bpr_holdout.py --all --figures
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
from bpr import build_refiner, refine  # noqa: E402
from eval_seg2_holdout import HOLD, gt_mask, predict  # noqa: E402
from make_crops import crop_box  # noqa: E402
from metrics import FIELDS, match  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, CKPT, ROOT, build_model  # noqa: E402
from train_seg2 import ARCHS, build_seg2, split_tag  # noqa: E402
from tta import predict_tta  # noqa: E402

EVAL = ROOT / "eval"
FIG = ROOT / "eval" / "bpr_figures"


def load_seg2(tag: str):
    arch, enc = split_tag(tag)
    ms = []
    for f in range(5):
        ck = CKPT / "seg2" / tag / f"fold{f}.pt"
        if ck.exists():
            m = build_seg2(arch, enc, pretrained=False)
            m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
            m.eval()
            ms.append(m)
    return ms


def coarse_masks(tag: str, use_tta: bool = True) -> list[dict]:
    """回傳每顆牙的 {img, coarse, gt, image, gt_idx}，座標都在該牙的 crop 範圍內。"""
    coco = json.loads((ANN / "holdout.json").read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    per: dict[int, list] = {}
    for a in coco["annotations"]:
        per.setdefault(a["image_id"], []).append(a)

    out = []
    if tag == "maskrcnn":
        ck = torch.load(CKPT / "original" / "maskrcnn_final.pt",
                        map_location="cpu", weights_only=False)
        model = build_model(False, ck.get("mask_res", 28))
        model.load_state_dict(ck["model"])
        model.eval()
        for iid, anns in sorted(per.items()):
            im = imgs[iid]
            gray = cv2.imread(str(HOLD / im["file_name"]), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            h, w = gray.shape
            keep = [a for a in anns if not a.get("iscrowd", 0)]
            gts = np.stack([gt_mask(a, h, w) for a in keep])
            prob, _, scores = predict_tta(model, gray, 0.35) if use_tta else (None, None, None)
            pred = np.array([clean_mask(m) for m in prob > 0.5], bool)
            if not len(pred):
                continue
            _, matched = match(pred, scores, gts, im["file_name"])
            for pi, gi in matched.items():
                x0, y0, x1, y1 = crop_box(keep[gi]["bbox"], 0.2, w, h)
                ys, xs = np.nonzero(pred[pi])
                if len(xs):
                    x0, y0 = min(x0, int(xs.min())), min(y0, int(ys.min()))
                    x1, y1 = max(x1, int(xs.max()) + 1), max(y1, int(ys.max()) + 1)
                out.append({"img": gray[y0:y1, x0:x1], "coarse": pred[pi][y0:y1, x0:x1],
                            "gt": gts[gi][y0:y1, x0:x1], "image": im["file_name"], "gt_idx": gi})
        return out

    ms = load_seg2(tag)
    for iid, anns in sorted(per.items()):
        im = imgs[iid]
        gray = cv2.imread(str(HOLD / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h, w = gray.shape
        for i, a in enumerate([x for x in anns if not x.get("iscrowd", 0)]):
            x0, y0, x1, y1 = crop_box(a["bbox"], 0.2, w, h)
            prob = predict(ms, gray[y0:y1, x0:x1], use_tta)
            small = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0.5
            out.append({"img": gray[y0:y1, x0:x1], "coarse": clean_mask(small),
                        "gt": gt_mask(a, h, w)[y0:y1, x0:x1], "image": im["file_name"], "gt_idx": i})
    return out


def contour(ax_img, m, color, thick=2):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ax_img, cs, -1, color, thick)


def figure(tag: str, items: list[dict], befores, afters, n: int = 6) -> None:
    """挑改動最大的幾顆牙，把 GT／精修前／精修後三條輪廓畫在同一張圖上。"""
    FIG.mkdir(parents=True, exist_ok=True)
    ch = [np.abs(a["_fine"].astype(int) - it["coarse"].astype(int)).sum() for it, a in
          zip(items, [{"_fine": f} for f in afters])]
    idx = np.argsort(ch)[::-1][:n]
    panels = []
    for i in idx:
        it = items[i]
        rgb = cv2.cvtColor(it["img"], cv2.COLOR_GRAY2BGR)
        contour(rgb, it["gt"], (0, 255, 0), 2)          # 綠 = 標註
        contour(rgb, it["coarse"], (0, 0, 255), 1)      # 紅 = 精修前
        contour(rgb, afters[i], (255, 128, 0), 1)       # 藍 = 精修後
        hh = 420
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * hh / rgb.shape[0]), hh))
        cv2.putText(rgb, f"{it['image']} #{it['gt_idx']}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        panels.append(rgb)
    wmax = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 0, 0, wmax - p.shape[1], cv2.BORDER_CONSTANT, value=0)
              for p in panels]
    out = np.hstack(panels)
    cv2.putText(out, "green=GT   red=before BPR   blue=after BPR", (6, out.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(FIG / f"{tag}.png"), out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unetpp_resnet34")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    net = build_refiner()
    net.load_state_dict(torch.load(CKPT / "bpr" / "fold0.pt", map_location="cpu",
                                   weights_only=False)["model"])
    net.eval()

    tags = ["maskrcnn"] + sorted(d.name for d in (CKPT / "seg2").iterdir()
                                 if d.is_dir() and list(d.glob("fold*.pt"))) \
        if args.all else [args.model]

    EVAL.mkdir(exist_ok=True)
    print(f"{'模型':<26}{'n':>4}{'Dice 前→後':>22}{'B-IoU 前→後':>24}{'HD95 前→後':>20}")
    print("-" * 96)
    for tag in tags:
        items = coarse_masks(tag)
        if not items:
            continue
        rb, ra, fines = [], [], []
        for it in items:
            fine = refine(net, it["img"], it["coarse"])
            fines.append(fine)
            for dst, m in ((rb, it["coarse"]), (ra, fine)):
                r = match(m[None], np.array([1.0]), it["gt"][None], it["image"])[0]
                for x in r:
                    x["gt_idx"] = it["gt_idx"]
                dst += r
        for name, rs in ((f"bpr_hold_{tag}_before", rb), (f"bpr_hold_{tag}_after", ra)):
            with (EVAL / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rs)
        g = lambda rs, k: np.median([float(r[k]) for r in rs if r["kind"] == "TP"])
        tp = len([r for r in ra if r["kind"] == "TP"])
        print(f"{tag:<26}{tp:>4}"
              f"{g(rb,'dice'):>11.4f}→{g(ra,'dice'):<10.4f}"
              f"{g(rb,'biou'):>12.4f}→{g(ra,'biou'):<11.4f}"
              f"{g(rb,'hd95'):>9.1f}→{g(ra,'hd95'):<9.1f}", flush=True)
        if args.figures:
            figure(tag, items, rb, fines)
    if args.figures:
        print(f"\n比對圖 → {FIG}")


if __name__ == "__main__":
    main()
