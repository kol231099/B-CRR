"""final/ 的 OOF 實驗台：找出兩階段（改後）剩下的誤差從哪裡來，再決定下一步改什麼。

放在 koi/setA/final/scripts/ 執行。只用 OOF（93 張、五折，每張由沒看過它的那一折
推論），**不碰 holdout**——holdout 已用來確認過方法，不能再拿來挑參數。

做法同 hd95_lab.py：每顆牙只推論一次，把機率圖存成快取，之後所有分析與變體都從
快取重算（加 --reuse 只要幾十秒）。推論沿用 final 的 maskrcnn() 與 predict()，
與 Table 1 同一條路徑。

輸出三段：

A. 失敗型態：每顆牙的 HD95 來源（多圈 / 少圈 × 冠端 / 根端 / 側邊），以及三個
   假說各自能解釋多少「大錯」（HD95 ≥ 20 px）：
     H1 斷塊被丟　 第二階段輸出不連通，clean_mask 只留最大塊，丟掉的部分落在 GT 內
     H2 框外截斷　 GT 有一部分落在 crop 之外，第二階段根本看不到
     H3 補綴物　　 GT 內有一塊遠亮於牙本體的區域（金屬冠、填補物）

B. 誤差拆解：把每顆牙的錯誤像素分成「兩個模型都錯」與「只有一個錯」。
   前者是任何融合都救不回來的部分；後者是更聰明的組合方式還能拿回的上限。

C. 變體：融合權重、clean_mask 的替代規則，與 Mask R-CNN 逐顆配對比較。
   這裡挑出的設定之後要在 holdout 上只確認一次。

用法（在 koi/setA/final 底下）：
    python3 scripts/final_lab.py
    python3 scripts/final_lab.py --reuse
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_holdout_all as E  # noqa: E402  final 版
from eval_seg2_holdout import gt_mask, predict  # noqa: E402
from make_crops_obb import obb_of, warp_of  # noqa: E402
from metrics import assd, hd95  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, IMAGES, ROOT  # noqa: E402
from train_seg2 import build_seg2, split_tag  # noqa: E402

if ROOT.name != "final":
    sys.exit(f"⚠ 這支腳本必須放在 koi/setA/final/scripts/ 執行，目前的根目錄是 {ROOT}。")

PAD = getattr(E, "PAD", 0.2)
TAG = getattr(E, "TAG", "unet_tu-hrnet_w32")
THR, MARGIN, BIG = 0.35, 16, 20.0
OUT = ROOT / "eval" / "final_lab"


# ================================================================ 推論（只做一次）

def load_seg(ckdir, fold):
    arch, enc = split_tag(TAG)
    m = build_seg2(arch, enc, pretrained=False)
    m.load_state_dict(torch.load(ROOT / ckdir / "seg2" / TAG / f"fold{fold}.pt",
                                 map_location="cpu", weights_only=False)["model"])
    return m.eval()


def hr_prob(seg, gray, box):
    h, w = gray.shape
    M, cw, ch = warp_of(*box, PAD)
    prob = predict([seg], cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR), False)
    small = cv2.resize(prob, (cw, ch), interpolation=cv2.INTER_LINEAR)
    Mi = cv2.invertAffineTransform(M)
    back = cv2.warpAffine(small, Mi, (w, h), flags=cv2.INTER_LINEAR)
    inbox = cv2.warpAffine(np.ones((ch, cw), np.uint8), Mi, (w, h), flags=cv2.INTER_NEAREST)
    return back, inbox.astype(bool)


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


@torch.no_grad()
def infer_image(fold, seg_j, seg_b, gray, gts):
    t = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1)
    out = E.maskrcnn(fold)([t])[0]
    keep = out["scores"].numpy() >= THR
    pm = out["masks"].numpy()[keep, 0]
    bm = [np.asarray(clean_mask(p > 0.5), bool) for p in pm]
    recs = []
    for gt in gts:
        ious = [iou(b, gt) for b in bm]
        if not ious or max(ious) < 0.5:
            recs.append(None)
            continue
        i = int(np.argmax(ious))
        box = obb_of(bm[i].astype(np.uint8))
        p_j, inbox = hr_prob(seg_j, gray, box)
        p_b, _ = hr_prob(seg_b, gray, box)
        maps = {"gt": gt.astype(np.float32), "p_m": pm[i], "p_j": p_j, "p_b": p_b,
                "inbox": inbox.astype(np.float32)}
        any_ = gt.copy()
        for k in ("p_m", "p_j", "p_b"):
            any_ |= maps[k] > 0.05
        ys, xs = np.nonzero(any_)
        y0, y1 = max(ys.min() - MARGIN, 0), min(ys.max() + MARGIN + 1, gt.shape[0])
        x0, x1 = max(xs.min() - MARGIN, 0), min(xs.max() + MARGIN + 1, gt.shape[1])
        rec = {k: v[y0:y1, x0:x1].astype(np.float16) for k, v in maps.items()}
        rec["gray"] = gray[y0:y1, x0:x1].copy()
        recs.append(rec)
    return recs


def build_cache(ck_jit, ck_base):
    teeth, t0 = [], time.time()
    jobs = []
    for fold in range(5):
        coco = json.loads((ANN / f"fold{fold}_val.json").read_text(encoding="utf-8"))
        imgs = {i["id"]: i for i in coco["images"]}
        per: dict[int, list] = {}
        for a in coco["annotations"]:
            if not a.get("iscrowd"):
                per.setdefault(a["image_id"], []).append(a)
        jobs += [(fold, imgs[i], anns) for i, anns in sorted(per.items())]
    print(f"OOF：{len(jobs)} 張、{sum(len(a) for _, _, a in jobs)} 顆標註牙", flush=True)
    cur = -1
    for n, (fold, im, anns) in enumerate(jobs, 1):
        if fold != cur:
            seg_j, seg_b, cur = load_seg(ck_jit, fold), load_seg(ck_base, fold), fold
        gray = cv2.imread(str(IMAGES / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  ⚠ 讀不到 {IMAGES / im['file_name']}")
            continue
        h, w = gray.shape
        gts = np.stack([gt_mask(a, h, w) for a in anns])
        for gi, rec in enumerate(infer_image(fold, seg_j, seg_b, gray, gts)):
            teeth.append({"fold": fold, "image": im["file_name"], "tooth": gi, "rec": rec})
        el = time.time() - t0
        print(f"  推論 {n}/{len(jobs)}　fold {fold}　{im['file_name']}　已花 {el:.0f}s　"
              f"預估剩 {el / n * (len(jobs) - n):.0f}s", flush=True)
    return teeth


# ================================================================ 後處理規則

def keep_largest(m):
    return np.asarray(clean_mask(m), bool)


def fill_holes(m):
    """只填內部孔洞，不丟任何連通塊（clean_mask 的後半段）。"""
    m = m.astype(np.uint8)
    h, w = m.shape
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return (m | (1 - ff)).astype(bool)


def keep_anchored(m, anchor, min_frac=0.05):
    """保留與 anchor（Mask R-CNN 遮罩）重疊的所有連通塊，再填洞。用來檢驗 H1。"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 2:
        return keep_largest(m)
    total = stats[1:, cv2.CC_STAT_AREA].sum()
    keep = np.zeros(m.shape, bool)
    for k in range(1, n):
        comp = lab == k
        if (comp & anchor).any() and stats[k, cv2.CC_STAT_AREA] >= min_frac * total:
            keep |= comp
    return fill_holes(keep) if keep.any() else keep_largest(m)


def variants():
    """名稱 → 函式(rec) → 二值遮罩。"""
    def fuse(r, w):
        return w * r["p_j"] + (1 - w) * r["p_m"]
    V = {
        "M  Mask R-CNN": lambda r: keep_largest(r["p_m"] > 0.5),
        "B  兩階段改前（重訓）": lambda r: keep_largest(r["p_b"] > 0.5),
        "J  框擾動 HRNet": lambda r: keep_largest(r["p_j"] > 0.5),
        "F  融合 0.5（現行）": lambda r: keep_largest(fuse(r, 0.5) > 0.5),
        "J  框擾動 + 錨定保留": lambda r: keep_anchored(r["p_j"] > 0.5, r["p_m"] > 0.5),
        "F  融合 0.5 + 錨定保留": lambda r: keep_anchored(fuse(r, 0.5) > 0.5, r["p_m"] > 0.5),
    }
    for w in (0.3, 0.4, 0.6, 0.7):
        V[f"F  融合 w_HR={w}"] = lambda r, w=w: keep_largest(fuse(r, w) > 0.5)
    return V


# ================================================================ 診斷

def where(pred, gt, gray):
    from scipy.ndimage import distance_transform_edt
    dp, dg = distance_transform_edt(~pred), distance_transform_edt(~gt)
    ep = cv2.Canny(pred.astype(np.uint8) * 255, 100, 200) > 0
    eg = cv2.Canny(gt.astype(np.uint8) * 255, 100, 200) > 0
    if not ep.any() or not eg.any():
        return "?", "?"
    over, under = dg[ep], dp[eg]
    if np.percentile(over, 95) >= np.percentile(under, 95):
        d, pts, side = "多圈", np.argwhere(ep), over
    else:
        d, pts, side = "少圈", np.argwhere(eg), under
    bad = pts[side >= np.percentile(side, 95)]
    ys, xs = np.nonzero(gt)
    c = np.array([ys.mean(), xs.mean()])
    u = np.linalg.svd(np.stack([ys, xs], 1) - c, full_matrices=False)[2][0]
    proj = (ys - c[0]) * u[0] + (xs - c[1]) * u[1]
    lo, hi = proj.min(), proj.max()
    t = ((bad[:, 0] - c[0]) * u[0] + (bad[:, 1] - c[1]) * u[1] - lo) / max(hi - lo, 1e-6)
    tg = (proj - lo) / max(hi - lo, 1e-6)
    g = gray[ys, xs].astype(np.float32)
    crown_hi = g[tg > 0.8].mean() > g[tg < 0.2].mean()
    pos = np.where(t > 0.8, "高", np.where(t < 0.2, "低", "側邊"))
    pos = np.where(pos == "高", "冠端" if crown_hi else "根端",
                   np.where(pos == "低", "根端" if crown_hi else "冠端", pos))
    vals, cnt = np.unique(pos, return_counts=True)
    return d, str(vals[cnt.argmax()])


def hypotheses(r, gt):
    """回傳三個假說的量化值。"""
    raw = r["p_j"] > 0.5
    kept = keep_largest(raw)
    dropped = raw & ~kept
    h1 = (dropped & gt).sum() / max(gt.sum(), 1)                 # 被 clean_mask 丟掉、其實是牙的比例
    h2 = (gt & ~(r["inbox"] > 0.5)).sum() / max(gt.sum(), 1)      # GT 落在 crop 外的比例
    g = r["gray"].astype(np.float32)
    body = np.median(g[gt]) if gt.any() else 0.0
    bright = gt & (g > max(body + 60, np.percentile(g[gt], 99) if gt.any() else 255))
    h3 = bright.sum() / max(gt.sum(), 1)                          # GT 內遠亮於牙本體的比例
    return float(h1), float(h2), float(h3)


def shared_error(m, j, gt):
    """錯誤像素拆解：兩者都錯 / 只有 Mask R-CNN 錯 / 只有 HRNet 錯。"""
    em, ej = m ^ gt, j ^ gt
    return int((em & ej).sum()), int((em & ~ej).sum()), int((~em & ej).sum())


# ================================================================ 主程式

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-jit", default="checkpoints_obb_jit")
    ap.add_argument("--ckpt-base", default="checkpoints_obb_base")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "cache_oof.pkl.gz"
    if args.reuse and cache.exists():
        teeth = pickle.load(gzip.open(cache, "rb"))
        print(f"讀取快取：{len(teeth)} 筆")
    else:
        teeth = build_cache(args.ckpt_jit, args.ckpt_base)
        with gzip.open(cache, "wb") as fh:
            pickle.dump(teeth, fh)

    ok = [t for t in teeth if t["rec"] is not None]
    print(f"\n{len(teeth)} 顆牙，Mask R-CNN 配到 {len(ok)}，漏檢 {len(teeth) - len(ok)}")
    V = variants()
    rows, t0 = [], time.time()
    for n, t in enumerate(ok, 1):
        r = {k: (v if k == "gray" else np.asarray(v, np.float32)) for k, v in t["rec"].items()}
        gt = r["gt"] > 0.5
        row = {"fold": t["fold"], "image": t["image"], "tooth": t["tooth"]}
        preds = {}
        for name, fn in V.items():
            p = fn(r)
            preds[name] = p
            row[f"{name}|dice"] = 2 * (p & gt).sum() / max(p.sum() + gt.sum(), 1)
            row[f"{name}|hd95"] = hd95(p, gt)
            row[f"{name}|assd"] = assd(p, gt)
        for name in ("M  Mask R-CNN", "J  框擾動 HRNet", "F  融合 0.5（現行）"):
            row[f"{name}|where"] = "/".join(where(preds[name], gt, r["gray"]))
        row["H1"], row["H2"], row["H3"] = hypotheses(r, gt)
        row["both"], row["onlyM"], row["onlyJ"] = shared_error(
            preds["M  Mask R-CNN"], preds["J  框擾動 HRNet"], gt)
        row["errM"] = int((preds["M  Mask R-CNN"] ^ gt).sum())
        row["errF"] = int((preds["F  融合 0.5（現行）"] ^ gt).sum())
        rows.append(row)
        if n % 25 == 0 or n == len(ok):
            print(f"  分析 {n}/{len(ok)}　已花 {time.time() - t0:.0f}s", flush=True)

    with (OUT / "teeth_oof.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def col(k):
        return np.array([r[k] for r in rows], float)

    F0, M0 = "F  融合 0.5（現行）", "M  Mask R-CNN"

    # ---------------- A
    print(f"\n{'=' * 92}\nA. 失敗型態（OOF，n = {len(rows)}）\n{'=' * 92}")
    for k in (M0, "J  框擾動 HRNet", F0):
        vals, cnt = np.unique([r[f"{k}|where"] for r in rows], return_counts=True)
        o = np.argsort(-cnt)
        print(f"  {k}：" + "　".join(f"{vals[i]} {cnt[i]}" for i in o))
    big = [r for r in rows if r[f"{F0}|hd95"] >= BIG]
    print(f"\n  現行融合 HD95 ≥ {BIG:.0f} px 的大錯：{len(big)} 顆（佔 {len(big) / len(rows):.0%}）")
    if big:
        h1 = np.array([r["H1"] for r in big])
        h2 = np.array([r["H2"] for r in big])
        h3 = np.array([r["H3"] for r in big])
        print(f"    H1 斷塊被丟（被丟掉的真牙 ≥ 1% GT）　{(h1 >= 0.01).sum():>3} 顆")
        print(f"    H2 框外截斷（GT 有 ≥ 0.5% 在 crop 外） {(h2 >= 0.005).sum():>3} 顆")
        print(f"    H3 補綴物（GT 內有 ≥ 2% 異常亮區）　　 {(h3 >= 0.02).sum():>3} 顆")
        print(f"    三者皆否　　　　　　　　　　　　　　　 "
              f"{((h1 < 0.01) & (h2 < 0.005) & (h3 < 0.02)).sum():>3} 顆")
        print("    最差 10 顆：")
        for r in sorted(big, key=lambda r: -r[f"{F0}|hd95"])[:10]:
            print(f"      fold {r['fold']}  {r['image']}  #{r['tooth']}　HD95 ① {r[M0 + '|hd95']:.1f} → "
                  f"融合 {r[F0 + '|hd95']:.1f}（{r[F0 + '|where']}）　"
                  f"H1 {r['H1']:.1%} H2 {r['H2']:.1%} H3 {r['H3']:.1%}")

    # ---------------- B
    both, om, oj = col("both").sum(), col("onlyM").sum(), col("onlyJ").sum()
    tot = both + om + oj
    print(f"\n{'=' * 92}\nB. 誤差拆解（Mask R-CNN vs 框擾動 HRNet，所有錯誤像素加總）\n{'=' * 92}")
    print(f"  兩者都錯　　{both / tot:6.1%}　← 任何「兩模型組合」都救不回的部分")
    print(f"  只有 ① 錯　 {om / tot:6.1%}")
    print(f"  只有 HRNet 錯 {oj / tot:6.1%}")
    eM, eF = col("errM").sum(), col("errF").sum()
    print(f"\n  錯誤像素總數：① {eM:.0f}　現行融合 {eF:.0f}　理想組合下限 {both:.0f}")
    gain = (eM - both)
    print(f"  → 相對 ①，理想組合最多可減少 {gain / eM:.0%} 的錯誤像素；"
          f"現行融合實際減少 {(eM - eF) / eM:.0%}，"
          f"拿到可得空間的 {(eM - eF) / gain:.0%}" if gain > 0 else "  → 兩者錯誤完全重疊")
    print("  判讀：拿到的比例低 → 更聰明的組合方式（依信心、依位置加權）值得做；"
          "\n        「兩者都錯」佔多數 → 組合類方法已到頂，要改模型本身或檢查標註。")

    # ---------------- C
    print(f"\n{'=' * 92}\nC. 變體（與 ① 逐顆配對；勝率 = HD95 比 ① 低的牙所佔比例）\n{'=' * 92}")
    print(f"  {'變體':<24}{'Dice中位':>9}{'HD95中位':>10}{'HD95平均':>10}{'ASSD中位':>10}"
          f"{'勝率':>7}{'p(HD95)':>10}{'大錯數':>8}")
    b = col(f"{M0}|hd95")
    for name in V:
        h = col(f"{name}|hd95")
        okm = np.isfinite(h) & np.isfinite(b)
        d = h[okm] - b[okm]
        try:
            p = wilcoxon(h[okm], b[okm]).pvalue if name != M0 and np.any(d != 0) else np.nan
        except ValueError:
            p = np.nan
        print(f"  {name:<24}{np.nanmedian(col(name + '|dice')):>9.4f}{np.nanmedian(h):>10.2f}"
              f"{np.nanmean(h):>10.2f}{np.nanmedian(col(name + '|assd')):>10.2f}"
              f"{(d < 0).mean():>7.0%}{'' if np.isnan(p) else f'{p:.4f}':>10}"
              f"{int((h >= BIG).sum()):>8}")
    print(f"\n逐顆明細 → {OUT / 'teeth_oof.csv'}")


if __name__ == "__main__":
    main()
