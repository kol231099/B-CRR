"""五條 pipeline 在 holdout 上的統一評估。

holdout 的 18 張影像與五折的訓練集、驗證集完全零重疊（已驗證），因此**每一折的
權重都是對 holdout 的有效獨立評估**，不需要另外訓練全量模型。此處讓五折權重各跑
一次，報平均 ± 標準差：前者是方法的表現，後者是「換一批訓練資料會抖多少」——
這個資訊單一的全量模型給不出來。

五條變體逐項對齊：偵測門檻 0.35、padding 0.2、clean_mask、原圖座標、metrics.py
十項指標、單模型不集成。--tta 同時開偵測器與第二階段的 TTA。

**holdout 的 FP 不可解讀為偵測誤判。** 實測 holdout 標註密度僅 1.44 顆/張，而
五折是 2.46 顆/張；模型在 holdout 上偵測到 2.67 顆/張，與五折的真實密度相符，
且這些「FP」的分數中位數 0.990（22 個裡 17 個 >0.9）。也就是說 holdout 只標了
部分牙齒，模型找到的其餘牙齒被記成 FP。TP 上的分割指標與 recall 仍然有效，
precision 與 FP 不得報告。

--only：只評估指定的影像（主檔名，例如 114 122 …）。輸出檔名不變，會覆蓋同名 CSV。
--ckpt-dir：OBB 第二階段的權重目錄；框擾動版是 checkpoints_obb_jit，輸出檔名加 _jit。

用法：
    py scripts/eval_holdout_all.py --variant 1 2 4
    py scripts/eval_holdout_all.py --variant 1 2 3 4 5 --tta
    py scripts/eval_holdout_all.py --variant 1 4 --ckpt-dir checkpoints_obb_jit --only 114 122
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
from eval_seg2_holdout import HOLD, gt_mask, predict  # noqa: E402
from make_crops import crop_box  # noqa: E402
from make_crops_obb import obb_of, warp_of  # noqa: E402
from metrics import FIELDS, match  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, CKPT, ROOT, build_model  # noqa: E402
from train_seg2 import build_seg2, split_tag  # noqa: E402
from tta import predict_tta  # noqa: E402

EVAL, PAD, TAG = ROOT / "eval", 0.2, "unet_tu-hrnet_w32"
NAMES = {1: "MaskRCNN", 2: "MaskRCNN_HBB_HRNet", 3: "YOLOseg",
         4: "MaskRCNN_OBB_HRNet", 5: "YOLOOBB_OBB_HRNet"}
OBB_ARCHS = {5: "yolo11s", 6: "yolov8s", 7: "yolo12s", 8: "yolo26s"}
_cache: dict = {}
_OBB_ARCH: list = ["yolo11s"]   # 變體 5~8 共用同一段程式，只換這個
_OBB_CKPT: list = ["checkpoints_obb"]   # 由 --ckpt-dir 設定


def maskrcnn(fold: int):
    if ("mr", fold) not in _cache:
        ck = torch.load(CKPT / "original" / f"maskrcnn_fold{fold}.pt", map_location="cpu",
                        weights_only=False)
        m = build_model(False, ck.get("mask_res", 28))
        m.load_state_dict(ck["model"])
        m.eval()
        _cache[("mr", fold)] = m
    return _cache[("mr", fold)]


def seg2(fold: int, obb: bool):
    key = ("sg", fold, obb)
    if key not in _cache:
        arch, enc = split_tag(TAG)
        m = build_seg2(arch, enc, pretrained=False)
        root = ROOT / (_OBB_CKPT[0] if obb else "checkpoints")
        m.load_state_dict(torch.load(root / "seg2" / TAG / f"fold{fold}.pt",
                                     map_location="cpu", weights_only=False)["model"])
        m.eval()
        _cache[key] = m
    return _cache[key]


def yolo(fold: int, obb: bool):
    key = ("yl", fold, obb)
    if key not in _cache:
        from ultralytics import YOLO
        run = (ROOT / "yolo_obb_runs" / _OBB_ARCH[0]) if obb else (ROOT / "yolo_runs")
        _cache[key] = YOLO(run / f"fold{fold}" / "weights" / "best.pt")
    return _cache[key]


def mr_masks(fold, gray, thr, use_tta, hw):
    model = maskrcnn(fold)
    if use_tta:
        prob, _, sc = predict_tta(model, gray, thr)
        m = np.array([clean_mask(x) for x in prob > 0.5], bool).reshape(-1, *hw)
    else:
        t = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1)
        out = model([t])[0]
        keep = out["scores"].numpy() >= thr
        m = np.array([clean_mask(x) for x in out["masks"].numpy()[keep, 0] > 0.5],
                     bool).reshape(-1, *hw)
        sc = out["scores"].numpy()[keep]
    return m, np.asarray(sc)


def refine_hbb(seg, gray, coarse, hw, use_tta):
    h, w = hw
    out = []
    for cm in coarse:
        ys, xs = np.nonzero(cm)
        if len(ys) == 0:
            out.append(cm)
            continue
        box = [float(xs.min()), float(ys.min()),
               float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
        x0, y0, x1, y1 = crop_box(box, PAD, w, h)
        prob = predict([seg], gray[y0:y1, x0:x1], use_tta)
        m = np.zeros((h, w), bool)
        m[y0:y1, x0:x1] = cv2.resize(prob, (x1 - x0, y1 - y0),
                                     interpolation=cv2.INTER_LINEAR) > 0.5
        out.append(clean_mask(m))
    return np.array(out, bool).reshape(-1, h, w)


def refine_obb(seg, gray, obbs, hw, use_tta):
    h, w = hw
    out = []
    for (cx, cy, rw, rh, ang) in obbs:
        M, cw, ch = warp_of(cx, cy, rw, rh, ang, PAD)
        prob = predict([seg], cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR), use_tta)
        small = cv2.resize(prob, (cw, ch), interpolation=cv2.INTER_LINEAR)
        back = cv2.warpAffine(small, cv2.invertAffineTransform(M), (w, h),
                              flags=cv2.INTER_LINEAR) > 0.5
        out.append(clean_mask(back))
    return np.array(out, bool).reshape(-1, h, w)


@torch.no_grad()
def predict_one(v, fold, gray, path, thr, use_tta, hw):
    h, w = hw
    if v in (1, 2, 4):
        coarse, sc = mr_masks(fold, gray, thr, use_tta, hw)
        if v == 1:
            return coarse, sc
        if v == 2:
            return refine_hbb(seg2(fold, False), gray, coarse, hw, use_tta), sc
        obbs, keep = [], []
        for i, cm in enumerate(coarse):
            if cm.any():
                obbs.append(obb_of(cm.astype(np.uint8)))
                keep.append(i)
        return refine_obb(seg2(fold, True), gray, obbs, hw, use_tta), sc[keep]

    res = yolo(fold, v >= 5).predict(str(path), conf=thr, imgsz=1024,
                                     augment=use_tta, verbose=False)[0]
    if v == 3:
        if res.masks is None or len(res.masks.xy) == 0:
            return np.zeros((0, h, w), bool), np.zeros(0)
        m = np.array([clean_mask(cv2.fillPoly(np.zeros((h, w), np.uint8),
                                              [p.astype(np.int32)], 1).astype(bool))
                      for p in res.masks.xy], bool).reshape(-1, h, w)
        return m, res.boxes.conf.cpu().numpy()
    if res.obb is None or len(res.obb) == 0:
        return np.zeros((0, h, w), bool), np.zeros(0)
    obbs = []
    for quad in res.obb.xyxyxyxy.cpu().numpy():
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(quad.reshape(-1, 2).astype(np.float32))
        if rw > rh:
            rw, rh, ang = rh, rw, ang + 90
        obbs.append((cx, cy, rw, rh, ang))
    return refine_obb(seg2(fold, True), gray, obbs, hw, use_tta), res.obb.conf.cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", type=int, nargs="+", required=True,
                    choices=[1, 2, 3, 4, 5, 6, 7, 8],
                    help="6/7/8 與 5 同樣是 YOLO-OBB→HRNet，只換偵測器架構")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--ckpt-dir", default="checkpoints_obb",
                    help="OBB 第二階段權重目錄；框擾動版是 checkpoints_obb_jit")
    ap.add_argument("--only", nargs="+", default=[], metavar="IMG",
                    help="只評估這些影像（主檔名，逗號或空白分隔）")
    args = ap.parse_args()
    _OBB_CKPT[0] = args.ckpt_dir
    only = {Path(x.strip()).stem for a in args.only for x in a.split(",") if x.strip()}

    coco = json.loads((ANN / "holdout.json").read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    per: dict[int, list] = {}
    for a in coco["annotations"]:
        if not a.get("iscrowd"):
            per.setdefault(a["image_id"], []).append(a)
    if only:
        per = {i: v for i, v in per.items() if Path(imgs[i]["file_name"]).stem in only}
        miss = only - {Path(imgs[i]["file_name"]).stem for i in per}
        if miss:
            print(f"⚠ holdout.json 裡找不到：{', '.join(sorted(miss))}")
    n_teeth = sum(len(v) for v in per.values())
    print(f"holdout：{len(per)} 張、{n_teeth} 顆標註牙")

    jit = "_jit" if args.ckpt_dir.endswith("_jit") else ""
    tta = "_tta" if args.tta else ""
    for v in args.variant:
        if v in OBB_ARCHS:
            _OBB_ARCH[0] = OBB_ARCHS[v]
            NAMES[v] = f"{OBB_ARCHS[v]}OBB_OBB_HRNet"
            _cache.clear()
        per_fold = []
        allrows = []
        for fold in range(5):
            rows = []
            for iid, anns in sorted(per.items()):
                im = imgs[iid]
                gray = cv2.imread(str(HOLD / im["file_name"]), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                h, w = gray.shape
                gt = np.stack([gt_mask(a, h, w) for a in anns])
                pred, sc = predict_one(v, fold, gray, HOLD / im["file_name"],
                                       args.thr, args.tta, (h, w))
                r = match(pred, sc, gt, im["file_name"])[0]
                for x in r:
                    x["gt_idx"] = f"{x['gt_idx']}"
                rows += r
            for x in rows:
                x["image"] = x["image"]
            allrows += [dict(x, kind=x["kind"]) | {"fold": fold} for x in rows]
            tp = [x for x in rows if x["kind"] == "TP" and x.get("dice")]
            per_fold.append({k: np.median([float(x[k]) for x in tp])
                             for k in ("dice", "iou", "biou", "hd95", "assd")} |
                            {"tp": len(tp), "fp": sum(x["kind"] == "FP" for x in rows),
                             "fn": sum(x["kind"] == "FN" for x in rows)})
            sfx = (jit if v >= 4 else "") + tta   # 只有 OBB 第二階段吃 --ckpt-dir
            with (EVAL / f"hold5_{NAMES[v]}{sfx}_fold{fold}.csv").open(
                    "w", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(fh, fieldnames=FIELDS)
                wr.writeheader()
                wr.writerows(rows)

        print(f"\n變體 {v}　{NAMES[v]}　holdout {len(per)} 張 n={n_teeth}　{'含' if args.tta else '無'} TTA")
        print(f"  {'fold':<6}{'TP':>4}{'FP':>4}{'FN':>4}"
              f"{'Dice':>9}{'IoU':>9}{'B-IoU':>9}{'HD95':>8}{'ASSD':>8}")
        for f, m in enumerate(per_fold):
            print(f"  {f:<6}{m['tp']:>4}{m['fp']:>4}{m['fn']:>4}{m['dice']:>9.4f}"
                  f"{m['iou']:>9.4f}{m['biou']:>9.4f}{m['hd95']:>8.2f}{m['assd']:>8.2f}")
        line = f"  {'平均':<4}"
        for k in ("tp", "fp", "fn"):
            line += f"{np.mean([m[k] for m in per_fold]):>4.0f}"
        for k, w_ in (("dice", 9), ("iou", 9), ("biou", 9), ("hd95", 8), ("assd", 8)):
            line += f"{np.mean([m[k] for m in per_fold]):>{w_}.4f}"
        print(line)
        sd = "  " + " " * 5 + " " * 12 + "".join(
            f"±{np.std([m[k] for m in per_fold]):>8.4f}" for k in
            ("dice", "iou", "biou", "hd95", "assd"))
        print(sd)


if __name__ == "__main__":
    main()
