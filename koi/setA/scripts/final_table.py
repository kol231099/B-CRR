"""Table 1 加上融合法的列，評估規則與原表完全相同。

規則（照 doc/segmentation_benchmark.md 第 3 節）：
  * 共同命中：每條 pipeline 取「五折都判為 TP」的牙，再對 --ref 的五條取交集（n = 13）。
    新增的列不參與交集——原表的牙齒集合與數字因此一個都不變，新列只是算在同一批牙上。
  * 彙總：逐折取中位數，再平均五折；Area ICC 為逐折 ICC(2,1) absolute 再平均。
  * 配對檢定：同一顆牙五折平均後做 Wilcoxon 符號等級檢定。

**自我檢查**：--ref 那五列算出來必須和 Table 1 一模一樣（例如 Mask R-CNN HD95 9.97）。
對不上就代表 --ref 名稱或資料不對，新列的數字也不能用。

用法（在 koi/setA/final 底下）：
    python3 scripts/final_table.py --list                      # 列出 eval/ 裡有哪些 hold5_ 名稱
    python3 scripts/final_table.py --ref 名稱1 名稱2 名稱3 名稱4 名稱5
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from icc import icc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL, ANN = ROOT / "eval", ROOT / "annotations"
NEW = [("+ 框擾動 HRNet（單獨）", "FUS_OBBjit"),
       ("+ 融合（本方法）", "FUS_fuse"),
       ("Mask R-CNN +TTA", "FUS_MaskRCNN_tta"),
       ("+ 融合 +TTA", "FUS_fuse_tta")]


def names_in_eval():
    pat = re.compile(r"^hold5_(.+)_fold0\.csv$")
    return sorted(m.group(1) for f in EVAL.glob("hold5_*_fold0.csv") if (m := pat.match(f.name)))


def load(name):
    files = [EVAL / f"hold5_{name}_fold{k}.csv" for k in range(5)]
    miss = [f.name for f in files if not f.exists()]
    if miss:
        sys.exit(f"找不到：{', '.join(miss)}")
    return [list(csv.DictReader(f.open(encoding="utf-8"))) for f in files]


def tp_map(rows):
    return {(r["image"], r["gt_idx"]): r for r in rows if r["kind"] == "TP" and r.get("dice")}


def gt_area():
    d = json.loads((ANN / "holdout.json").read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in d["images"]}
    per: dict = {}
    for a in d["annotations"]:
        if not a.get("iscrowd"):
            per.setdefault(a["image_id"], []).append(a)
    out = {}
    for iid, anns in per.items():
        im = imgs[iid]
        for gi, a in enumerate(anns):
            m = np.zeros((im["height"], im["width"]), np.uint8)
            for poly in a["segmentation"]:
                cv2.fillPoly(m, [np.array(poly, np.int32).reshape(-1, 2)], 1)
            out[(im["file_name"], str(gi))] = float(m.sum())
    return out


def row_metrics(folds, keys, area):
    """逐折中位數再平均；ICC 逐折算再平均。缺牙（某折非 TP）的那折只用有的牙。"""
    per = {k: [] for k in ("dice", "iou", "hd95", "assd", "icc")}
    for f in folds:
        m = tp_map(f)
        ks = [k for k in keys if k in m]
        for k in ("dice", "iou", "hd95", "assd"):
            per[k].append(np.median([float(m[c][k]) for c in ks]))
        g = np.array([area[c] for c in ks])
        p = g * (1 + np.array([float(m[c]["rvd"]) for c in ks]))
        per["icc"].append(icc(np.stack([p, g], 1), "absolute")["icc"])
    return {k: float(np.mean(v)) for k, v in per.items()}


def tooth_mean(folds, keys, metric):
    out = {}
    for c in keys:
        v = [float(tp_map(f)[c][metric]) for f in folds if c in tp_map(f)]
        out[c] = np.mean(v) if v else np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--ref", nargs="+", help="Table 1 原本五條 pipeline 的 hold5_ 名稱，第一個必須是 Mask R-CNN")
    args = ap.parse_args()

    if args.list or not args.ref:
        print("eval/ 裡的 hold5_ 名稱：\n  " + "\n  ".join(names_in_eval()))
        if not args.ref:
            print("\n請用 --ref 指定 Table 1 的五條（第一個放 Mask R-CNN）。")
        return

    area = gt_area()
    ref = {n: load(n) for n in args.ref}
    sets = [set.intersection(*(set(tp_map(f)) for f in folds)) for folds in ref.values()]
    for n, s in zip(args.ref, sets):
        print(f"  {n:<40} 五折全命中 {len(s)}")
    common = sorted(set.intersection(*sets) & set(area))
    print(f"\n共同命中 n = {len(common)}（Table 1 為 13）\n")

    new = [(lab, n, load(n)) for lab, n in NEW if (EVAL / f"hold5_{n}_fold0.csv").exists()]
    hdr = f"  {'Pipeline':<34}{'DICE':>8}{'IOU':>8}{'HD95':>8}{'ASSD':>8}{'AreaICC':>9}{'缺牙':>6}"
    print(hdr + "\n  " + "─" * (len(hdr) - 2))
    for n, folds in ref.items():
        r = row_metrics(folds, common, area)
        print(f"  {n:<34}{r['dice']:>8.4f}{r['iou']:>8.4f}{r['hd95']:>8.2f}{r['assd']:>8.2f}"
              f"{r['icc']:>9.4f}{'':>6}")
    print("  " + "─" * (len(hdr) - 2))
    for lab, n, folds in new:
        r = row_metrics(folds, common, area)
        missing = sum(1 for f in folds for c in common if c not in tp_map(f))
        print(f"  {lab:<34}{r['dice']:>8.4f}{r['iou']:>8.4f}{r['hd95']:>8.2f}{r['assd']:>8.2f}"
              f"{r['icc']:>9.4f}{missing:>6}")
    print("\n  ※ 上半部必須與 Table 1 完全一致，否則下半部不可用。缺牙 = 共同命中的牙在該方法"
          "某折不是 TP 的次數（應為 0）。")

    # 配對檢定：n = 13（共同命中）與 Mask R-CNN 系列都命中的全部牙
    mr = ref[args.ref[0]]
    pairs = [("+ 融合", "FUS_fuse", mr, args.ref[0]),
             ("+ 框擾動 HRNet（單獨）", "FUS_OBBjit", mr, args.ref[0])]
    if (EVAL / "hold5_FUS_MaskRCNN_tta_fold0.csv").exists():
        pairs.append(("+ 融合 +TTA", "FUS_fuse_tta", load("FUS_MaskRCNN_tta"), "FUS_MaskRCNN_tta"))
    all_mr = set.intersection(*(set(tp_map(f)) for f in mr)) & set(area)
    print(f"\n配對 Wilcoxon（同一顆牙五折平均；勝 = 新方法較好的牙數）")
    for lab, n, base, bname in pairs:
        if not (EVAL / f"hold5_{n}_fold0.csv").exists():
            continue
        folds = load(n)
        for scope, keys in ((f"共同命中 n={len(common)}", common),
                            (f"Mask R-CNN 全命中 n={len(all_mr)}", sorted(all_mr))):
            line = f"  {lab} vs {bname}　{scope}　"
            for metric, better in (("dice", 1), ("hd95", -1)):
                a, b = tooth_mean(folds, keys, metric), tooth_mean(base, keys, metric)
                ks = [k for k in keys if np.isfinite(a[k]) and np.isfinite(b[k])]
                x, y = np.array([a[k] for k in ks]), np.array([b[k] for k in ks])
                win = int(((x - y) * better > 0).sum())
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        p = wilcoxon(x, y).pvalue
                except ValueError:
                    p = float("nan")
                line += f"{metric.upper()} 勝 {win}/{len(ks)} p={p:.3f}　"
            print(line)


if __name__ == "__main__":
    main()
