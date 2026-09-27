#!/usr/bin/env python3
"""為 BPR 產生「粗糙遮罩 + GT + 影像」三元組。

BPR（Tang et al., CVPR 2021）的精修網路吃的是粗糙遮罩，因此要先把基礎模型的
預測存下來。存的單位是「一顆牙」，範圍是該牙的 crop box（與 make_crops 相同的
pad 0.2），因為邊界 patch 只會在這個範圍內取。

split 的處理是關鍵：
  --split val    每個 fold 的驗證集，用該 fold 的模型預測（out-of-fold）。
                 這是最後要被精修、被評分的那批。
  --split train  每個 fold 的訓練集，用該 fold 的模型預測。
                 這是拿來訓練精修網路的。基礎模型看過這些影像，遮罩會比
                 測試時乾淨一些——原論文即是如此，屬已知的輕微落差。

如此 fold k 的精修網路只看過 fold k 的訓練影像，評估時套在 fold k 的驗證集上，
沒有洩漏。

用法
    py scripts/bpr_dump.py --model unetpp_resnet34 --split train --tta
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
from metrics import match  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, CKPT, ROOT, ToothDataset, build_model  # noqa: E402
from tta import predict_tta  # noqa: E402
from train_seg2 import ARCHS, CROPS, SIZE, fold_ids  # noqa: E402

BPR = ROOT / "bpr"


@torch.no_grad()
def infer(model, crop: np.ndarray, use_tta: bool) -> np.ndarray:
    base = cv2.resize(crop, SIZE[::-1], interpolation=cv2.INTER_AREA)
    views = [(False, False), (True, False), (False, True), (True, True)] if use_tta \
        else [(False, False)]
    acc = []
    for fh, fv in views:
        v = base[:, ::-1] if fh else base
        v = v[::-1] if fv else v
        t = torch.from_numpy(np.ascontiguousarray(v))
        t = t.float().div(255).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        o = torch.sigmoid(model(t))[0, 0].numpy()
        if fv:
            o = o[::-1]
        if fh:
            o = o[:, ::-1]
        acc.append(np.ascontiguousarray(o))
    return np.mean(acc, axis=0)


def dump_maskrcnn(split: str, use_tta: bool, score_thr: float = 0.35) -> int:
    """Mask R-CNN 的分支。

    它是所有統計檢定的比較基準，若只精修第二階段模型而基準不動，「X 顯著勝過
    Mask R-CNN」會被 BPR 放大成假象——不是 X 變好，是基準沒享受到同樣處理。

    與第二階段不同之處：遮罩來自偵測結果，要先與 GT 貪婪配對才知道哪顆對哪顆；
    只有配對成功（TP）的才有 GT 可精修。裁切範圍取 GT 的 crop box 與預測 bbox
    的聯集，確保預測若溢出 GT 範圍也不會被切掉。
    """
    manifest = list(csv.DictReader((CROPS / "manifest.csv").open(encoding="utf-8")))
    coco = json.loads((ANN / "instances_all.json").read_text(encoding="utf-8"))
    id_of_name = {i["file_name"]: i["id"] for i in coco["images"]}
    per: dict[int, list] = {}
    for a in coco["annotations"]:
        per.setdefault(a["image_id"], []).append(a)
    order = {}
    for iid, lst in per.items():
        for i, a in enumerate([x for x in lst if not x["iscrowd"]]):
            order[(iid, a["id"])] = i
    box_of = {}
    for r in manifest:
        iid = id_of_name[r["image"]]
        box_of[(r["image"], order[(iid, int(r["ann_id"]))])] = r

    out = BPR / f"maskrcnn{'_tta' if use_tta else ''}" / split
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for fold in range(5):
        ck = CKPT / "original" / f"maskrcnn_fold{fold}.pt"
        if not ck.exists():
            continue
        model = build_model()
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
        model.eval()
        ds = ToothDataset(ANN / f"fold{fold}_{split}.json", train=False)
        for i in range(len(ds)):
            img, tgt = ds[i]
            name = ds.items[i][0]["file_name"]
            gray = (img.numpy()[0] * 255).astype(np.uint8)
            gt = tgt["masks"].numpy().astype(bool)
            if use_tta:
                prob, _, scores = predict_tta(model, gray, score_thr)
                pred = np.array([clean_mask(m) for m in prob > 0.5], bool)
            else:
                with torch.no_grad():
                    o = model([img])[0]
                keep = o["scores"].numpy() >= score_thr
                scores = o["scores"].numpy()[keep]
                pred = np.array([clean_mask(m) for m in o["masks"].numpy()[keep, 0] > 0.5], bool)
            if not len(pred):
                continue
            _, matched = match(pred, scores, gt, name)
            for pi, gi in matched.items():
                r = box_of.get((name, gi))
                if r is None:
                    continue
                x0, y0, x1, y1 = (int(r[k]) for k in ("x0", "y0", "x1", "y1"))
                ys, xs = np.nonzero(pred[pi])          # 預測可能溢出 GT 的 crop box
                if len(xs):
                    x0, y0 = min(x0, int(xs.min())), min(y0, int(ys.min()))
                    x1, y1 = max(x1, int(xs.max()) + 1), max(y1, int(ys.max()) + 1)
                np.savez_compressed(out / f"{r['crop_id']}.npz",
                                    img=gray[y0:y1, x0:x1],
                                    coarse=pred[pi][y0:y1, x0:x1].astype(np.uint8),
                                    gt=gt[gi][y0:y1, x0:x1].astype(np.uint8), fold=fold)
                n += 1
        print(f"  fold{fold} {split} 完成", flush=True)
    print(f"→ {out}　共 {n} 顆牙", flush=True)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unetpp_resnet34")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    if args.model == "maskrcnn":
        dump_maskrcnn(args.split, args.tta)
        return

    import segmentation_models_pytorch as smp

    tag = args.model
    arch = next(a for a in ARCHS if tag.startswith(a + "_"))
    enc = tag[len(arch) + 1:]

    manifest = {r["crop_id"]: r for r in csv.DictReader((CROPS / "manifest.csv").open(encoding="utf-8"))}
    coco = json.loads((ANN / "instances_all.json").read_text(encoding="utf-8"))
    gt_of = {(a["image_id"], a["id"]): a for a in coco["annotations"]}
    id_of_name = {i["file_name"]: i["id"] for i in coco["images"]}

    out = BPR / f"{tag}{'_tta' if args.tta else ''}" / args.split
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for fold in range(5):
        ck = CKPT / "seg2" / tag / f"fold{fold}.pt"
        if not ck.exists():
            continue
        model = getattr(smp, ARCHS[arch])(encoder_name=enc, encoder_weights=None,
                                          in_channels=3, classes=1)
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
        model.eval()

        tr_ids, va_ids = fold_ids(fold)
        for cid in (tr_ids if args.split == "train" else va_ids):
            r = manifest[cid]
            x0, y0, x1, y1 = (int(r[k]) for k in ("x0", "y0", "x1", "y1"))
            img = cv2.imread(str(CROPS / "images" / f"{cid}.png"), cv2.IMREAD_GRAYSCALE)
            prob = infer(model, img, args.tta)
            coarse = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0.5
            coarse = clean_mask(coarse)

            a = gt_of[(id_of_name[r["image"]], int(r["ann_id"]))]
            gt = np.zeros((y1 - y0, x1 - x0), np.uint8)
            for poly in a["segmentation"]:
                p = np.array(poly, np.float32).reshape(-1, 2) - [x0, y0]
                cv2.fillPoly(gt, [p.astype(np.int32)], 1)

            # 檔名必須帶 fold：同一顆牙會出現在 5 個 fold 中的 4 個訓練集裡，
            # 且每個 fold 的模型對它產生的粗糙遮罩都不同。若只用 crop_id 當檔名，
            # 後面的 fold 會蓋掉前面的，最後只剩最高 fold 的版本。
            stem = f"f{fold}_{cid}" if args.split == "train" else cid
            np.savez_compressed(out / f"{stem}.npz", img=img, coarse=coarse.astype(np.uint8),
                                gt=gt, fold=fold)
            n += 1
        print(f"  fold{fold} {args.split} 完成", flush=True)
    print(f"→ {out}　共 {n} 顆牙", flush=True)


if __name__ == "__main__":
    main()
