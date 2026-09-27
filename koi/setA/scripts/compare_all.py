"""彙整所有方法的評估結果，輸出論文用的比較表與配對檢定。

所有方法共用 metrics.py 的十個指標，並在**原圖座標**上計算，因此可直接並排。
配對以 (影像, 該影像內的牙齒序號) 為鍵——同一顆牙在不同方法上的分數是配對的，
用獨立樣本檢定會把「這顆牙本來就難」的變異算進誤差項，檢定力大幅下降。

用法：
    py scripts/compare_all.py            # 無 TTA
    py scripts/compare_all.py --tta
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

EVAL = Path(__file__).resolve().parent.parent / "eval"
COLS = [("Dice", "dice", "%.4f", False), ("IoU", "iou", "%.4f", False),
        ("Sens", "sens", "%.4f", False), ("Prec", "prec", "%.4f", False),
        ("B-IoU", "biou", "%.4f", False), ("HD95", "hd95", "%.1f", True),
        ("ASSD", "assd", "%.2f", True), ("NSD@3", "nsd3", "%.4f", False),
        ("NSD@5", "nsd5", "%.4f", False), ("RVD", "rvd", "%+.4f", None)]


def load(pattern: str) -> dict:
    d = {}
    for f in sorted(glob.glob(str(EVAL / pattern))):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["kind"] == "TP" and r.get("dice"):
                d[(r["image"], r["gt_idx"])] = r
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    sfx = "_tta" if args.tta else ""

    methods = [("Mask R-CNN（單階段）", f"maskrcnn_original{sfx}_fold*.csv")]
    for f in sorted(EVAL.glob(f"seg2_*{'_tta' if args.tta else ''}.csv")):
        name = f.stem[len("seg2_"):]
        if args.tta:
            if not name.endswith("_tta"):
                continue
            name = name[:-4]
        elif name.endswith("_tta"):
            continue
        methods.append((name, f.name))

    data = {n: load(p) for n, p in methods}
    data = {n: d for n, d in data.items() if d}
    if len(data) < 2:
        print("可比較的方法不足")
        return

    print(f"\n{'=' * 78}\n所有方法比較{'（含 TTA）' if args.tta else '（無 TTA）'}"
          f"　指標皆於原圖座標計算\n{'=' * 78}")
    hdr = f"{'方法':<24}{'n':>5}"
    for lab, *_ in COLS:
        hdr += f"{lab:>9}"
    print(hdr)
    print("-" * len(hdr))
    for n, d in data.items():
        line = f"{n:<24}{len(d):>5}"
        for lab, key, fmt, _ in COLS:
            v = np.array([float(r[key]) for r in d.values()])
            line += f"{fmt % np.median(v):>9}"
        print(line)

    ref = list(data)[0]
    print(f"\n各方法 vs {ref}　配對 Wilcoxon（Holm 校正）")
    print("-" * 78)
    for key, lab in (("biou", "Boundary IoU"), ("hd95", "HD95")):
        res = []
        for n, d in list(data.items())[1:]:
            common = sorted(set(data[ref]) & set(d))
            if len(common) < 10:
                continue
            a = np.array([float(d[k][key]) for k in common])
            b = np.array([float(data[ref][k][key]) for k in common])
            if np.allclose(a, b):
                continue
            res.append((n, len(common), float(np.median(a - b)), float(wilcoxon(a, b).pvalue)))
        res.sort(key=lambda r: r[3])
        print(f"\n  【{lab}】")
        for i, (n, cnt, diff, p) in enumerate(res):
            ph = min(1.0, p * (len(res) - i))
            print(f"    {n:<24}n={cnt:>4}　中位差 {diff:+8.4f}　p={p:.4f}　Holm={ph:.4f}"
                  f"　{'顯著' if ph < 0.05 else '不顯著'}")


if __name__ == "__main__":
    main()
