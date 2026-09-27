"""單階段 vs 兩階段的三方對照，全部端到端、全部五折 OOF、全部原圖座標。

compare_all.py 的第二階段欄位用的是標註框（oracle bbox），對兩階段有利；
這裡三條 pipeline 都自己偵測，因此「多一個階段值不值得」這個問題才答得準。

配對鍵是 (影像, 該影像內的牙齒序號)，只取三者都命中的牙，否則比的是
「誰漏的牙比較難」而不是分割品質。漏檢/多檢另外以 TP/FP/FN 報。

用法：
    py scripts/compare_stages.py
"""

from __future__ import annotations

import argparse
import csv
import glob
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

EVAL = Path(__file__).resolve().parent.parent / "eval"
COLS = [("Dice", "dice", "%.4f", False), ("IoU", "iou", "%.4f", False),
        ("B-IoU", "biou", "%.4f", False), ("HD95", "hd95", "%.1f", True),
        ("ASSD", "assd", "%.2f", True), ("NSD@3", "nsd3", "%.4f", False),
        ("NSD@5", "nsd5", "%.4f", False)]


def load(pattern: str) -> tuple[dict, dict]:
    d, cnt = {}, {"TP": 0, "FP": 0, "FN": 0}
    for f in sorted(glob.glob(str(EVAL / pattern))):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            cnt[r["kind"]] = cnt.get(r["kind"], 0) + 1
            if r["kind"] == "TP" and r.get("dice"):
                d[(r["image"], r["gt_idx"])] = r
    return d, cnt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-conf", default="", help="YOLO csv 後綴，例如 _c50")
    ap.add_argument("--tta", action="store_true", help="改讀 TTA 版的 csv")
    args = ap.parse_args()

    t = "_tta" if args.tta else ""
    methods = [
        ("1 Mask R-CNN 單獨", f"maskrcnn_original{t}_fold*.csv" if args.tta
                              else "maskrcnn_original_fold*.csv"),
        ("2 MaskR-CNN→HBB→HRNet", f"e2e_oof_unet_tu-hrnet_w32{t}_fold*.csv"),
        ("3 YOLO11-seg 單階段", f"yolo{args.yolo_conf}_fold*.csv"),
        ("4 MaskR-CNN→OBB→HRNet", f"obb_maskrcnn_unet_tu-hrnet_w32{t}_fold*.csv"),
        ("5 YOLO11-OBB→OBB→HRNet", f"obb_yoloobb_unet_tu-hrnet_w32{t}_fold*.csv"),
    ]
    data, counts = {}, {}
    for n, p in methods:
        d, c = load(p)
        if d:
            data[n], counts[n] = d, c
        else:
            print(f"（略過 {n}：找不到 {p}）")
    if len(data) < 2:
        print("可比較的方法不足")
        return

    print(f"\n{'=' * 92}\n端到端對照　五折 OOF　原圖座標　單模型　"
          f"{'含 TTA' if args.tta else '無 TTA'}\n{'=' * 92}")
    hdr = f"{'方法':<26}{'TP':>5}{'FP':>5}{'FN':>5}" + "".join(f"{l:>9}" for l, *_ in COLS)
    print(hdr + "\n" + "-" * 104)
    for n, d in data.items():
        c = counts[n]
        line = f"{n:<26}{c['TP']:>5}{c['FP']:>5}{c['FN']:>5}"
        for _, key, fmt, _hi in COLS:
            v = np.array([float(r[key]) for r in d.values()])
            line += f"{fmt % np.median(v):>9}"
        print(line)
    print("\n※ 上表中位數各自算在該方法自己的 TP 上；下面的檢定只用各方法共同命中的牙。")

    common = sorted(set.intersection(*(set(d) for d in data.values())))
    print(f"\n共同命中 n={len(common)}")
    for a_name, b_name in combinations(data, 2):
        print(f"\n{'-' * 92}\n{a_name}  vs  {b_name}　配對 Wilcoxon")
        for lab, key, _fmt, lower_better in COLS:
            a = np.array([float(data[a_name][k][key]) for k in common])
            b = np.array([float(data[b_name][k][key]) for k in common])
            if np.allclose(a, b):
                continue
            diff = a - b
            p = float(wilcoxon(a, b).pvalue)
            wins = int((diff < 0).sum() if lower_better else (diff > 0).sum())
            arrow = "↓越小越好" if lower_better else "↑越大越好"
            mark = "顯著" if p < 0.05 else "不顯著"
            better = a_name if (np.median(diff) < 0) == lower_better else b_name
            print(f"  {lab:<6}{arrow}　中位差(前-後) {np.median(diff):+9.4f}　"
                  f"前者勝 {wins:>3}/{len(common)}　p={p:.4f}　{mark}"
                  f"{'　→ ' + better.split('（')[0] if p < 0.05 else ''}")
    print("\n※ 中位差是「逐顆配對差值的中位數」，不等於上表兩個中位數相減。")


if __name__ == "__main__":
    main()
