#!/usr/bin/env python3
"""把第二階段的七個模型跑在 18 張保留測試集上。

與 eval_seg2.py 的差別只有資料來源與模型選法：

  資料  crop 直接從 holdout.json 的 GT bbox 現切（pad 0.2、原尺寸），
        與 make_crops.py 的規則相同。因此和五折比較表一樣是 oracle bbox
        的設定——偵測誤差不計入，這點在論文裡要寫清楚。

  模型  第二階段沒有「用全部 63 張重訓」的最終模型（那是 maskrcnn_final.pt
        才有的）。這裡把五個 fold 的機率圖平均當作整體模型，這是實務上會
        部署的形式；--per-fold 可另外看單折的離散程度。

用法
    py scripts/eval_seg2_holdout.py --all --tta
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
from make_crops import crop_box  # noqa: E402
from metrics import FIELDS, match  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, CKPT, ROOT  # noqa: E402
from train_seg2 import ARCHS, SIZE, build_seg2  # noqa: E402

EVAL = ROOT / "eval"
HOLD = ROOT / "holdout"


def gt_mask(a: dict, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for poly in a["segmentation"]:
        cv2.fillPoly(m, [np.array(poly, np.int32).reshape(-1, 2)], 1)
    return m.astype(bool)


@torch.no_grad()
def predict(models: list, crop: np.ndarray, use_tta: bool) -> np.ndarray:
    """對一個 crop 做（模型集成 × TTA）的機率平均。"""
    base = cv2.resize(crop, SIZE[::-1], interpolation=cv2.INTER_AREA)
    views = [(False, False), (True, False), (False, True), (True, True)] if use_tta \
        else [(False, False)]
    acc = []
    for m in models:
        for fh, fv in views:
            v = base[:, ::-1] if fh else base
            v = v[::-1] if fv else v
            t = torch.from_numpy(np.ascontiguousarray(v))
            t = t.float().div(255).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
            o = torch.sigmoid(m(t))[0, 0].numpy()
            if fv:
                o = o[::-1]
            if fh:
                o = o[:, ::-1]
            acc.append(np.ascontiguousarray(o))
    return np.mean(acc, axis=0)


def run_one(arch: str, encoder: str, use_tta: bool, per_fold: bool) -> list[dict]:
    import segmentation_models_pytorch as smp

    tag = f"{arch}_{encoder}"
    coco = json.loads((ANN / "holdout.json").read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    per: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        per.setdefault(a["image_id"], []).append(a)

    models = []
    for fold in range(5):
        ck = CKPT / "seg2" / tag / f"fold{fold}.pt"
        if not ck.exists():
            continue
        m = build_seg2(arch, encoder, pretrained=False)
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
        m.eval()
        models.append(m)
    if not models:
        return []

    groups = [("集成", models)] + ([(f"fold{i}", [m]) for i, m in enumerate(models)] if per_fold else [])
    out = []
    for label, sel in groups:
        rows = []
        for iid, anns in sorted(per.items()):
            im = imgs[iid]
            gray = cv2.imread(str(HOLD / im["file_name"]), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            h, w = gray.shape
            for i, a in enumerate([x for x in anns if not x.get("iscrowd", 0)]):
                x0, y0, x1, y1 = crop_box(a["bbox"], 0.2, w, h)
                prob = predict(sel, gray[y0:y1, x0:x1], use_tta)
                small = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0.5
                pred = np.zeros((h, w), bool)
                pred[y0:y1, x0:x1] = small
                pred = clean_mask(pred)
                rr = match(pred[None], np.array([1.0]), gt_mask(a, h, w)[None], im["file_name"])[0]
                for x in rr:
                    x["gt_idx"] = i
                rows += rr
        d = np.array([r["dice"] for r in rows if r["kind"] == "TP"])
        b = np.array([r["biou"] for r in rows if r["kind"] == "TP"])
        hd = np.array([r["hd95"] for r in rows if r["kind"] == "TP"])
        print(f"  {tag:<24} {label:<7} n={len(d):3d}  Dice {np.median(d):.4f}  "
              f"B-IoU {np.median(b):.4f}  HD95 {np.median(hd):5.1f}", flush=True)
        if label == "集成":
            out = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--per-fold", action="store_true", help="另外列出每個 fold 單獨的成績")
    args = ap.parse_args()

    todo = []
    if args.all:
        for d in sorted((CKPT / "seg2").iterdir()):
            if not d.is_dir() or not list(d.glob("fold*.pt")):
                continue
            for a in ARCHS:
                if d.name.startswith(a + "_"):
                    todo.append((a, d.name[len(a) + 1:]))
                    break
    else:
        todo = [(args.arch, args.encoder)]

    EVAL.mkdir(exist_ok=True)
    sfx = "_tta" if args.tta else ""
    for arch, enc in todo:
        rows = run_one(arch, enc, args.tta, args.per_fold)
        if not rows:
            continue
        p = EVAL / f"holdout_seg2_{arch}_{enc}{sfx}.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(rows)
        print(f"    → {p.name}", flush=True)


if __name__ == "__main__":
    main()
