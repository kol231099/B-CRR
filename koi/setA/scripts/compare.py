"""匯總所有方法的評估結果，做配對統計比較。

為什麼一定要配對
----------------
三條 pipeline 測的是**同一批 57 顆牙**，所以每顆牙在不同方法上的分數是配對的。
用獨立樣本檢定會把「這顆牙本來就難」的變異算進誤差項，檢定力大幅下降——牙齒
之間的差異（0.90~0.98）遠大於方法之間的差異（約 0.005），不配對幾乎測不出東西。
配對靠 (image, gt_idx)，由 metrics.match 寫入。

用 Wilcoxon signed-rank 而非成對 t 檢定：Dice 的分布左尾很長（少數幾顆牙特別
差），不符合常態假設。多重比較用 Holm 校正，比 Bonferroni 有檢定力且一樣嚴格。

漏檢與誤報不進配對檢定
----------------------
只有兩個方法都偵測到的牙才有分數可比。漏檢與誤報另外用計數報告——一個漏了牙
的方法，它的 Dice 反而會因為只留下容易的牙而變好看，這兩類數字必須並列。

用法：
    py koi/scripts/compare.py
    py koi/scripts/compare.py --metric hd95
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval"

METHODS = {
    "Mask R-CNN": "maskrcnn_fold{}.csv",
    "YOLOv11-seg": "yolo_fold{}.csv",
    "SAM2 (GT box)": "sam2_gt_fold{}.csv",
    "SAM2 (YOLO box)": "sam2_yolo_fold{}.csv",
    "MedSAM (GT box)": "medsam_gt_fold{}.csv",
    "MedSAM (YOLO box)": "medsam_yolo_fold{}.csv",
}


def load(pattern: str) -> tuple[dict, dict, int, int]:
    """回傳 {(image, gt_idx): dice}、{...: hd95}、FP 數、FN 數。"""
    dice, hd, n_fp, n_fn = {}, {}, 0, 0
    for k in range(5):
        f = EVAL / pattern.format(k)
        if not f.exists():
            return {}, {}, -1, -1
        for r in csv.DictReader(f.open(encoding="utf-8")):
            if r["kind"] == "TP":
                key = (r["image"], int(r["gt_idx"]))
                dice[key] = float(r["dice"])
                hd[key] = float(r["hd95"])
            elif r["kind"] == "FP":
                n_fp += 1
            else:
                n_fn += 1
    return dice, hd, n_fp, n_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=["dice", "hd95"], default="dice")
    args = ap.parse_args()

    data = {}
    for name, pat in METHODS.items():
        d, h, fp, fn = load(pat)
        if fp < 0:
            print(f"（{name} 尚未評估，略過）")
            continue
        data[name] = {"dice": d, "hd95": h, "fp": fp, "fn": fn}

    if len(data) < 1:
        return

    print(f"{'方法':<18}{'n':>4}{'Dice 中位':>11}{'IQR':>17}{'HD95 中位':>11}{'IQR':>15}"
          f"{'漏檢':>6}{'誤報':>6}")
    print("-" * 90)
    for name, v in data.items():
        d = np.array(list(v["dice"].values()))
        h = np.array(list(v["hd95"].values()))
        print(f"{name:<18}{len(d):>4}{np.median(d):>11.4f}"
              f"{f'{np.percentile(d,25):.4f}–{np.percentile(d,75):.4f}':>17}"
              f"{np.median(h):>11.1f}{f'{np.percentile(h,25):.1f}–{np.percentile(h,75):.1f}':>15}"
              f"{v['fn']:>6}{v['fp']:>6}")

    if len(data) < 2:
        print("\n（只有一個方法，無法做配對比較）")
        return

    print(f"\n配對 Wilcoxon signed-rank（{args.metric}），Holm 校正")
    print("-" * 90)
    results = []
    for a, b in itertools.combinations(data, 2):
        common = sorted(set(data[a][args.metric]) & set(data[b][args.metric]))
        if len(common) < 6:
            continue
        xa = np.array([data[a][args.metric][k] for k in common])
        xb = np.array([data[b][args.metric][k] for k in common])
        if np.allclose(xa, xb):
            continue
        stat, p = wilcoxon(xa, xb)
        results.append((a, b, len(common), float(np.median(xa - xb)), p))

    for i, (a, b, n, diff, p) in enumerate(sorted(results, key=lambda r: r[4])):
        p_holm = min(1.0, p * (len(results) - i))
        sig = "顯著" if p_holm < 0.05 else "不顯著"
        print(f"  {a} vs {b}　n={n}　中位差 {diff:+.4f}　p={p:.4f}　Holm p={p_holm:.4f}　{sig}")

    print("\n註一：以 GT box 當 prompt 的方法不做偵測，每個 box 必定輸出一個遮罩，所以它們的")
    print("      「漏檢」與「誤報」永遠相等，意思是**分割失敗率**（IoU < 0.5，遮罩差到配不上）,")
    print("      不是偵測失敗。這類方法只有 Dice 與 HD95 能跟偵測型方法相比。")
    print("註二：配對檢定只納入兩個方法都成功分割的牙齒。失敗率高的方法會因為只剩下容易的")
    print("      牙齒而讓 Dice 看起來偏好，所以務必連同 n 與失敗數一起看。")


if __name__ == "__main__":
    main()
