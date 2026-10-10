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
    # +TTA 表：--ref 換成五條的 _tta 版本，--new 只列 TTA 版的新方法
    python3 scripts/final_table.py --ref MaskRCNN_tta ... --new FUS_fuse_tta
    # 論文表格加 p 值：每個 --compare 與 --anchor 逐顆配對（Wilcoxon，Holm 校正）
    python3 scripts/final_table.py --ref MaskRCNN MaskRCNN_OBB_HRNet yolov8sseg yolo11sseg yolo26sseg \
        --anchor FUS_fuse --compare MaskRCNN yolov8sseg yolo11sseg yolo26sseg
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
if ROOT.name != "final":
    sys.exit(f"⚠ 這支腳本必須放在 koi/setA/final/scripts/ 執行，目前的根目錄是 {ROOT}。請先 cd 到 koi/setA/final。")
EVAL, ANN = ROOT / "eval", ROOT / "annotations"
NEW = [("④ 重訓（同設定）", "FUS_OBBbase"),
       ("+ 融合（HRNet 不擾動）", "FUS_fuse_base"),
       ("+ 框擾動 HRNet（單獨）", "FUS_OBBjit"),
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
    ap.add_argument("--new", nargs="+", default=None,
                    help="下半部要列哪些新方法（hold5_ 名稱）；預設列出全部 FUS_*")
    ap.add_argument("--anchor", default=None, help="論文表格的主角（本方法），例如 FUS_fuse")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="要與 --anchor 比較的方法，例如 MaskRCNN yolov8sseg yolo11sseg yolo26sseg")
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

    labels = dict((n, lab) for lab, n in NEW)
    picked = args.new if args.new else [n for _, n in NEW]
    new = [(labels.get(n, n), n, load(n)) for n in picked if (EVAL / f"hold5_{n}_fold0.csv").exists()]
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

    if args.anchor:
        paper_table(args, common, area)
        return

    # 配對檢定：n = 13（共同命中）與 Mask R-CNN 系列都命中的全部牙
    mr = ref[args.ref[0]]
    if args.new:   # 自訂列：每個新方法都與 --ref 第一個（Mask R-CNN）配對
        pairs = [(labels.get(n, n), n, mr, args.ref[0]) for n in args.new]
    else:
        pairs = [("+ 融合", "FUS_fuse", mr, args.ref[0]),
                 ("+ 融合（HRNet 不擾動）", "FUS_fuse_base", mr, args.ref[0]),
                 ("+ 框擾動 HRNet（單獨）", "FUS_OBBjit", mr, args.ref[0])]
        if (EVAL / "hold5_FUS_OBBbase_fold0.csv").exists():
            pairs.append(("+ 融合", "FUS_fuse", load("FUS_OBBbase"), "④ 重訓"))
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


def holm(ps):
    """Holm–Bonferroni 校正，回傳與輸入同順序的校正後 p。"""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj, running = np.empty_like(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def fmt_p(p):
    return "—" if not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def paper_table(args, common, area):
    """論文用的表：每列的指標值，加上該列相對 --anchor 的配對 p 值（Holm 校正）。"""
    metrics = (("dice", "DICE"), ("iou", "IOU"), ("hd95", "HD95 (px)"), ("assd", "ASSD (px)"))
    anchor = load(args.anchor)
    names = args.compare + [args.anchor]
    data = {n: (anchor if n == args.anchor else load(n)) for n in names}
    vals = {n: row_metrics(f, common, area) for n, f in data.items()}

    raw = {m: [] for m, _ in metrics}
    for n in args.compare:
        for m, _ in metrics:
            a = tooth_mean(anchor, common, m)
            b = tooth_mean(data[n], common, m)
            ks = [k for k in common if np.isfinite(a[k]) and np.isfinite(b[k])]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = wilcoxon([a[k] for k in ks], [b[k] for k in ks]).pvalue
            except ValueError:
                p = float("nan")
            raw[m].append(p)
    adj = {m: holm(v) for m, v in raw.items()}

    print(f"\n論文表格（n = {len(common)} 共同命中；p = 與 {args.anchor} 的配對 Wilcoxon，"
          f"每顆牙先取五折平均；括號內為原始 p，括號外為 Holm 校正後）\n")
    head = "  " + f"{'Pipeline':<22}" + "".join(f"{lab:>12}{'p':>16}" for _, lab in metrics) + f"{'Area ICC':>10}"
    print(head + "\n  " + "─" * (len(head) - 2))
    out_rows = []
    for n in names:
        line = f"  {n:<22}"
        rec = {"pipeline": n}
        for j, (m, lab) in enumerate(metrics):
            v = vals[n][m]
            if n == args.anchor:
                ptxt = "（基準）"
                rec[f"{m}_p_holm"] = rec[f"{m}_p_raw"] = ""
            else:
                i = args.compare.index(n)
                ptxt = f"{fmt_p(adj[m][i])} ({fmt_p(raw[m][i])})"
                rec[f"{m}_p_holm"], rec[f"{m}_p_raw"] = adj[m][i], raw[m][i]
            rec[m] = v
            line += f"{v:>12.4f}" if m in ("dice", "iou") else f"{v:>12.2f}"
            line += f"{ptxt:>16}"
        rec["area_icc"] = vals[n]["icc"]
        print(line + f"{vals[n]['icc']:>10.4f}")
        out_rows.append(rec)
    out = EVAL / f"table_pvalues_{args.anchor}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        wr.writeheader()
        wr.writerows(out_rows)
    print(f"\n  ※ Area ICC 是整組牙的單一統計量，沒有逐顆配對值，不做檢定（表上填「—」）。")
    print(f"  ※ Holm 校正在每個指標內、對 {len(args.compare)} 個比較進行。")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
