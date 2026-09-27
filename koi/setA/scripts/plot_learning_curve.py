"""評估學習曲線的各個模型並畫圖，回答「瓶頸是資料量還是模型」。

n=20 沿用 checkpoints/original/ 的完整模型——那就是用整個 fold 的 20 張訓練圖
訓出來的，不需要重跑。

每個資料量點取 3 個 fold，報中位數與範圍。單一 fold 的 val 只有 5 張圖，一個
點的雜訊會大到看不出趨勢。

判讀
----
    n=15→20 仍明顯上升      資料是瓶頸 → 先去標 300 張，改架構是浪費
    已平坦且遠低於人工一致性  模型是瓶頸 → PointRend / 56x56 遮罩頭才值得做
    已平坦且接近人工一致性    已到上限 → 收工

第三種要判定仍需人工標註一致性那個數字，那只能由人來測。

用法：
    py koi/scripts/plot_learning_curve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_maskrcnn import run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CKPT, EVAL = ROOT / "checkpoints", ROOT / "eval"
SIZES = [5, 10, 15, 20]
FOLDS = [0, 1, 2]


def main() -> None:
    results: dict[int, list[tuple[float, float, int, int]]] = {}
    for n in SIZES:
        tag = "original" if n == 20 else f"curve_n{n}"
        pts = []
        for k in FOLDS:
            if not (CKPT / tag / f"maskrcnn_fold{k}.pt").exists():
                continue
            rows = run(k, 0.35, False, "original", tag)
            tp = [r for r in rows if r["kind"] == "TP"]
            if not tp:
                continue
            d = np.array([float(r["dice"]) for r in tp])
            h = np.array([float(r["hd95"]) for r in tp])
            # 除了中位數也收集尾部：更多資料最可能改善的是困難案例，
            # 而中位數會把那個效果完全蓋掉。
            pts.append((float(np.median(d)), float(np.nanmedian(h)),
                        len(tp), sum(r["kind"] == "FN" for r in rows),
                        float(d.min()), float(np.nanmax(h)), int((d < 0.93).sum())))
        if pts:
            results[n] = pts
            print(f"  n={n:>2}  完成 {len(pts)} 個 fold", flush=True)

    if not results:
        print("尚無可評估的模型")
        return

    print(f"\n{'訓練張數':>8}{'Dice 中位':>11}{'fold 範圍':>17}{'最低 Dice':>11}"
          f"{'HD95 中位':>11}{'HD95 最高':>11}{'Dice<0.93':>11}{'漏檢':>6}")
    print("-" * 90)
    for n in SIZES:
        if n not in results:
            continue
        v = results[n]
        d = [p[0] for p in v]
        print(f"{n:>8}{np.median(d):>11.4f}{f'{min(d):.4f}–{max(d):.4f}':>17}"
              f"{min(p[4] for p in v):>11.4f}{np.median([p[1] for p in v]):>11.1f}"
              f"{max(p[5] for p in v):>11.1f}{sum(p[6] for p in v):>11}"
              f"{sum(p[3] for p in v):>6}")

    ns = [n for n in SIZES if n in results]
    if len(ns) >= 2:
        h0 = np.median([p[1] for p in results[ns[-2]]])
        h1 = np.median([p[1] for p in results[ns[-1]]])
        t0 = sum(p[6] for p in results[ns[-2]])
        t1 = sum(p[6] for p in results[ns[-1]])
        print(f"\n最後一段（n={ns[-2]}→{ns[-1]}）：HD95 中位 {h1 - h0:+.1f} px"
              f"　困難案例（Dice<0.93）{t0} → {t1} 顆")
        d0, d1 = np.median([p[0] for p in results[ns[-2]]]), np.median([p[0] for p in results[ns[-1]]])
        slope = d1 - d0
        print(f"\n最後一段（n={ns[-2]}→{ns[-1]}）Dice 變化：{slope:+.4f}")
        print("→ " + ("仍在明顯上升，資料量是瓶頸：先去標 300 張，改架構是浪費"
                      if slope > 0.008 else
                      "已趨平坦，加資料的邊際效益低。是否該改架構，取決於標註者"
                      "一致性——若人工一致性明顯高於此值，模型還有空間；若相當，就是到頂了"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, idx, lab, better in ((a1, 0, "Dice (median)", "higher"),
                                     (a2, 1, "HD95 px (median)", "lower")):
            med = [np.median([p[idx] for p in results[n]]) for n in ns]
            lo = [min(p[idx] for p in results[n]) for n in ns]
            hi = [max(p[idx] for p in results[n]) for n in ns]
            ax.fill_between(ns, lo, hi, alpha=0.18, color="#1c6479")
            ax.plot(ns, med, "o-", color="#1c6479", lw=2, ms=7)
            ax.set_xlabel("training images")
            ax.set_ylabel(f"{lab}  ({better} is better)")
            ax.set_xticks(ns)
            ax.grid(alpha=0.25)
        fig.suptitle("Mask R-CNN learning curve  (3 folds, shaded = fold range)")
        fig.tight_layout()
        EVAL.mkdir(parents=True, exist_ok=True)
        fig.savefig(EVAL / "learning_curve.png", dpi=150)
        print(f"\n圖 → {EVAL / 'learning_curve.png'}")
    except Exception as e:
        print(f"\n（畫圖略過：{type(e).__name__}: {e}）")


if __name__ == "__main__":
    main()
