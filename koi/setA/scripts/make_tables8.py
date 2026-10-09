"""輸出八條 pipeline 的表一二三，OOF 與 holdout 各一套。

表一  分割品質    TP FP FN Dice IoU HD95 ASSD 面積ICC
表二  實例層級    TP FP FN Precision Recall F1
表三  像素層級    Precision Recall Acc(原圖) Acc(框內) Specificity

所有中位數與 ICC 都算在「八條共同命中」的牙上，否則比的是誰漏的牙比較難。
TP/FP/FN 則為各方法自身的全量統計。

--exclude：把指定影像整張排除（標註、預測、FP/FN 全部不算）後重算三張表。
以檔名主檔名比對，`89`、`89.jpg` 都可以。排除必須有事先講得出的理由（標註錯誤、
影像品質、植體等），不能只因為模型在上面表現差——論文裡要同時報排除前後。

用法：
    py scripts/make_tables8.py --split oof
    py scripts/make_tables8.py --split holdout
    py scripts/make_tables8.py --split oof --exclude 89 22 14 81
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
from icc import icc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL, ANN = ROOT / "eval", ROOT / "annotations"

OOF = [
    ("① Mask R-CNN 單獨", "maskrcnn_original_fold*.csv"),
    ("② MaskRCNN→HBB→HRNet", "e2e_oof_unet_tu-hrnet_w32_fold*.csv"),
    ("③ YOLO11-seg 單階段", "yolo_fold*.csv"),
    ("④ MaskRCNN→OBB→HRNet", "obb_maskrcnn_unet_tu-hrnet_w32_fold*.csv"),
    ("④j MaskRCNN→OBB→HRNet 擾動", "obb_maskrcnn_unet_tu-hrnet_w32_jit_fold*.csv"),
    ("⑤ YOLO11-OBB→HRNet", "obb_yolo11s_unet_tu-hrnet_w32_fold*.csv"),
    ("⑥ YOLOv8-OBB→HRNet", "obb_yolov8s_unet_tu-hrnet_w32_fold*.csv"),
    ("⑦ YOLO12-OBB→HRNet", "obb_yolo12s_unet_tu-hrnet_w32_fold*.csv"),
    ("⑧ YOLO26-OBB→HRNet", "obb_yolo26s_unet_tu-hrnet_w32_fold*.csv"),
]
HOLD = [
    ("① Mask R-CNN 單獨", "hold5_MaskRCNN_fold*.csv"),
    ("② MaskRCNN→HBB→HRNet", "hold5_MaskRCNN_HBB_HRNet_fold*.csv"),
    ("③ YOLO11-seg 單階段", "hold5_YOLOseg_fold*.csv"),
    ("④ MaskRCNN→OBB→HRNet", "hold5_MaskRCNN_OBB_HRNet_fold*.csv"),
    ("⑤ YOLO11-OBB→HRNet", "hold5_YOLOOBB_OBB_HRNet_fold*.csv"),
    ("⑥ YOLOv8-OBB→HRNet", "hold5_yolov8sOBB_OBB_HRNet_fold*.csv"),
    ("⑦ YOLO12-OBB→HRNet", "hold5_yolo12sOBB_OBB_HRNet_fold*.csv"),
    ("⑧ YOLO26-OBB→HRNet", "hold5_yolo26sOBB_OBB_HRNet_fold*.csv"),
]
EXCLUDE: set[str] = set()   # 由 --exclude 設定，存主檔名


def excluded(name: str) -> bool:
    return Path(name).stem in EXCLUDE


def gt_geometry(split: str) -> dict:
    """每顆標註牙的面積、影像面積、+20% 框面積。"""
    out = {}
    files = [ANN / "holdout.json"] if split == "holdout" else \
            [ANN / f"fold{k}_val.json" for k in range(5)]
    for fp in files:
        d = json.loads(fp.read_text(encoding="utf-8"))
        imgs = {i["id"]: i for i in d["images"]}
        per: dict = {}
        for a in d["annotations"]:
            if not a.get("iscrowd"):
                per.setdefault(a["image_id"], []).append(a)
        for iid, anns in per.items():
            im = imgs[iid]
            if excluded(im["file_name"]):
                continue
            h, w = im["height"], im["width"]
            for gi, a in enumerate(anns):
                m = np.zeros((h, w), np.uint8)
                for poly in a["segmentation"]:
                    cv2.fillPoly(m, [np.array(poly, np.int32).reshape(-1, 2)], 1)
                pts = np.array(a["segmentation"][0], np.int32).reshape(-1, 2)
                _, _, bw, bh = cv2.boundingRect(pts)
                out[(im["file_name"], str(gi))] = {
                    "A": float(m.sum()), "WH": float(h * w), "box": float(bw * bh) * 1.96}
    return out


def load(pattern: str):
    """回傳 [每折的 rows]；OOF 的五折是互斥的，holdout 的五折是重複評估。"""
    files = sorted(EVAL.glob(pattern))
    return [[r for r in csv.DictReader(f.open(encoding="utf-8")) if not excluded(r["image"])]
            for f in files]


def tp_map(rows):
    return {(r["image"], r["gt_idx"]): r for r in rows
            if r["kind"] == "TP" and r.get("dice")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["oof", "holdout"], default="oof")
    ap.add_argument("--merge", action="store_true", help="三張表併成一張")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="IMG",
                    help="整張排除的影像，例如 89 22 14 81 或 89.jpg")
    args = ap.parse_args()
    EXCLUDE.update(Path(x.strip()).stem for a in args.exclude for x in a.split(",") if x.strip())
    hold = args.split == "holdout"
    spec = HOLD if hold else OOF
    gt = gt_geometry(args.split)

    data = {}
    for name, pat in spec:
        folds = load(pat)
        if not folds:
            print(f"（略過 {name}：找不到 {pat}）")
            continue
        data[name] = folds

    # OOF：五折互斥，合併成一組；holdout：五折各自是完整的一次評估
    sets = []
    for name, folds in data.items():
        if hold:
            sets.append(set.intersection(*(set(tp_map(f)) for f in folds)))
        else:
            merged = {}
            for f in folds:
                merged |= tp_map(f)
            sets.append(set(merged))
    common = sorted(set.intersection(*sets) & set(gt))
    n_lab = len(gt)
    print(f"\n{args.split.upper()}　共同命中 n={len(common)}　標註總數 {n_lab}")
    if EXCLUDE:
        files = [ANN / "holdout.json"] if hold else [ANN / f"fold{k}_val.json" for k in range(5)]
        stems = {Path(i["file_name"]).stem for fp in files
                 for i in json.loads(fp.read_text(encoding="utf-8"))["images"]}
        print(f"已排除影像：{', '.join(sorted(EXCLUDE & stems)) or '（無）'}")
        if EXCLUDE - stems:
            print(f"⚠ 這些不在 {args.split} 裡，沒有作用：{', '.join(sorted(EXCLUDE - stems))}")
    print()

    def per_fold_tp(folds):
        return [tp_map(f) for f in folds]

    def counts(folds):
        if hold:
            f = lambda k: np.mean([sum(r["kind"] == k for r in x) for x in folds])
        else:
            f = lambda k: sum(sum(r["kind"] == k for r in x) for x in folds)
        return f("TP"), f("FP"), f("FN")

    def agg(folds, fn):
        """OOF 合併五折算一次；holdout 每折算一次再平均。"""
        if hold:
            return float(np.mean([fn(m) for m in per_fold_tp(folds)]))
        merged = {}
        for f in folds:
            merged |= tp_map(f)
        return float(fn(merged))

    if args.merge:
        H = ("Pipeline", "TP", "FP", "FN", "實例P", "實例R", "F1", "Dice", "IoU",
             "HD95", "ASSD", "面積ICC", "像素P", "像素R", "Acc原圖", "Acc框內", "Spec")
        W = (24, 6, 6, 6, 8, 8, 8, 8, 8, 7, 6, 9, 8, 8, 9, 8, 9)
        print("".join(h.ljust(w) if i == 0 else h.rjust(w)
                      for i, (h, w) in enumerate(zip(H, W))))
        print("─" * sum(W))
        for name, folds in data.items():
            tp, fp, fn = counts(folds)
            ip, ir = tp / (tp + fp), tp / (tp + fn)
            med = {k: agg(folds, lambda m, k=k: np.median([float(m[c][k]) for c in common]))
                   for k in ("dice", "iou", "hd95", "assd")}

            def _icc(m):
                g = np.array([gt[c]["A"] for c in common])
                pv = g * (1.0 + np.array([float(m[c]["rvd"]) for c in common]))
                return icc(np.stack([pv, g], 1), "absolute")["icc"]

            def px(m):
                P = []; R = []; A1 = []; A2 = []; SS = []
                for c in common:
                    se, pc, g = float(m[c]["sens"]), float(m[c]["prec"]), gt[c]
                    TP = se * g["A"]; FP = TP / pc - TP if pc > 0 else 0.0; FN = g["A"] - TP
                    for tot, acc in ((g["WH"], A1), (g["box"], A2)):
                        acc.append((TP + max(tot - TP - FP - FN, 0)) / tot)
                    TN = max(g["WH"] - TP - FP - FN, 0)
                    SS.append(TN / max(TN + FP, 1)); P.append(pc); R.append(se)
                return np.array([np.median(P), np.median(R), np.median(A1),
                                 np.median(A2), np.median(SS)])

            if hold:
                pv = np.mean([px(m) for m in per_fold_tp(folds)], axis=0)
            else:
                merged = {}
                for f in folds:
                    merged |= tp_map(f)
                pv = px(merged)
            v = [name, f"{tp:.0f}", f"{fp:.0f}", f"{fn:.1f}" if hold else f"{fn:.0f}",
                 f"{ip:.4f}", f"{ir:.4f}", f"{2 * ip * ir / (ip + ir):.4f}",
                 f"{med['dice']:.4f}", f"{med['iou']:.4f}", f"{med['hd95']:.2f}",
                 f"{med['assd']:.2f}", f"{agg(folds, _icc):.4f}",
                 f"{pv[0]:.4f}", f"{pv[1]:.4f}", f"{pv[2]:.5f}", f"{pv[3]:.4f}", f"{pv[4]:.5f}"]
            print("".join(x.ljust(w) if i == 0 else x.rjust(w)
                          for i, (x, w) in enumerate(zip(v, W))))
        print("\n※ 實例P／F1：FP 含「模型找到但未被標註」的真牙，被系統性低估。")
        print("※ 面積ICC：ICC(2,1) absolute agreement，受樣本異質性主導，不宜跨研究比較。")
        print("※ Acc／Spec：背景佔 99% 以上，各條之間無實質差異。")
        return

    print("【表一】分割品質")
    hdr = f"  {'Pipeline':<24}{'TP':>6}{'FP':>6}{'FN':>6}{'Dice':>9}{'IoU':>9}{'HD95':>8}{'ASSD':>7}{'面積ICC':>10}"
    print(hdr + "\n  " + "─" * (len(hdr) - 2))
    for name, folds in data.items():
        tp, fp, fn = counts(folds)
        med = {k: agg(folds, lambda m, k=k: np.median([float(m[c][k]) for c in common]))
               for k in ("dice", "iou", "hd95", "assd")}
        def _icc(m):
            g = np.array([gt[c]["A"] for c in common])
            p = g * (1.0 + np.array([float(m[c]["rvd"]) for c in common]))
            return icc(np.stack([p, g], 1), "absolute")["icc"]
        v = agg(folds, _icc)
        f_fmt = f"{fn:>6.1f}" if hold else f"{fn:>6.0f}"
        print(f"  {name:<24}{tp:>6.0f}{fp:>6.0f}{f_fmt}{med['dice']:>9.4f}{med['iou']:>9.4f}"
              f"{med['hd95']:>8.2f}{med['assd']:>7.2f}{v:>10.4f}")
    print("  ※ ICC(2,1) absolute agreement，算在牙齒面積上。ICC 受樣本異質性主導，"
          "不宜跨研究比較。")

    print("\n【表二】實例層級")
    hdr = f"  {'Pipeline':<24}{'TP':>7}{'FP':>7}{'FN':>7}{'Precision':>11}{'Recall':>9}{'F1':>8}"
    print(hdr + "\n  " + "─" * (len(hdr) - 2))
    for name, folds in data.items():
        tp, fp, fn = counts(folds)
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        print(f"  {name:<24}{tp:>7.1f}{fp:>7.1f}{fn:>7.1f}{pr:>11.4f}{rc:>9.4f}"
              f"{2 * pr * rc / (pr + rc):>8.4f}")
    print("  ※ FP 含「模型找到但未被標註」的真牙，precision 與 F1 因此被系統性低估。")

    print("\n【表三】像素層級")
    hdr = f"  {'Pipeline':<24}{'Precision':>11}{'Recall':>9}{'Acc(原圖)':>12}{'Acc(框內)':>12}{'Specificity':>13}"
    print(hdr + "\n  " + "─" * (len(hdr) - 2))
    for name, folds in data.items():
        def px(m):
            P = []; R = []; A1 = []; A2 = []; S = []
            for c in common:
                se, pc, g = float(m[c]["sens"]), float(m[c]["prec"]), gt[c]
                TP = se * g["A"]; FP = TP / pc - TP if pc > 0 else 0.0; FN = g["A"] - TP
                for tot, acc in ((g["WH"], A1), (g["box"], A2)):
                    acc.append((TP + max(tot - TP - FP - FN, 0)) / tot)
                TN = max(g["WH"] - TP - FP - FN, 0)
                S.append(TN / max(TN + FP, 1)); P.append(pc); R.append(se)
            return np.array([np.median(P), np.median(R), np.median(A1),
                             np.median(A2), np.median(S)])
        if hold:
            v = np.mean([px(m) for m in per_fold_tp(folds)], axis=0)
        else:
            merged = {}
            for f in folds:
                merged |= tp_map(f)
            v = px(merged)
        print(f"  {name:<24}{v[0]:>11.4f}{v[1]:>9.4f}{v[2]:>12.5f}{v[3]:>12.4f}{v[4]:>13.5f}")


if __name__ == "__main__":
    main()
