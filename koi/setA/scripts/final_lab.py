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

--extra 名稱=權重目錄[@高,寬]：再加一個第二階段版本一起比（例如框擾動 + 邊界損失、
框擾動 + 768×384），會多出「X 名稱」與「F 融合 名稱」兩列。加了 --extra 會重新推論。

用法（在 koi/setA/final 底下）：
    python3 scripts/final_lab.py
    python3 scripts/final_lab.py --reuse
    python3 scripts/final_lab.py --extra BL=checkpoints_obb_jit_bl --extra R768=checkpoints_obb_jit_r768@768,384
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
import eval_seg2_holdout as ESH  # noqa: E402
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


def hr_prob(seg, gray, box, size=None):
    """size：第二階段輸入解析度 (高, 寬)；None 用 final 預設。暫時改 predict() 讀的 SIZE。"""
    h, w = gray.shape
    M, cw, ch = warp_of(*box, PAD)
    old = ESH.SIZE
    if size:
        ESH.SIZE = size
    try:
        prob = predict([seg], cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR), False)
    finally:
        ESH.SIZE = old
    small = cv2.resize(prob, (cw, ch), interpolation=cv2.INTER_LINEAR)
    Mi = cv2.invertAffineTransform(M)
    back = cv2.warpAffine(small, Mi, (w, h), flags=cv2.INTER_LINEAR)
    inbox = cv2.warpAffine(np.ones((ch, cw), np.uint8), Mi, (w, h), flags=cv2.INTER_NEAREST)
    return back, inbox.astype(bool)


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


@torch.no_grad()
def infer_image(fold, seg_j, seg_b, gray, gts, extras=()):
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
        for name, seg_x, size in extras:
            maps[f"p_x:{name}"] = hr_prob(seg_x, gray, box, size)[0]
        any_ = gt.copy()
        for k in maps:
            if k.startswith("p_"):
                any_ |= maps[k] > 0.05
        ys, xs = np.nonzero(any_)
        y0, y1 = max(ys.min() - MARGIN, 0), min(ys.max() + MARGIN + 1, gt.shape[0])
        x0, x1 = max(xs.min() - MARGIN, 0), min(xs.max() + MARGIN + 1, gt.shape[1])
        rec = {k: v[y0:y1, x0:x1].astype(np.float16) for k, v in maps.items()}
        rec["gray"] = gray[y0:y1, x0:x1].copy()
        recs.append(rec)
    return recs


def parse_extra(spec):
    name, rest = spec.split("=", 1)
    ck, size = (rest.split("@") + [None])[:2]
    return name, ck, (tuple(int(v) for v in size.split(",")) if size else None)


def build_cache(ck_jit, ck_base, extras=()):
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
            segx = [(n_, load_seg(ck, fold), sz) for n_, ck, sz in extras]
        gray = cv2.imread(str(IMAGES / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  ⚠ 讀不到 {IMAGES / im['file_name']}")
            continue
        h, w = gray.shape
        gts = np.stack([gt_mask(a, h, w) for a in anns])
        for gi, rec in enumerate(infer_image(fold, seg_j, seg_b, gray, gts, segx)):
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


def variants(extra_names=()):
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
    for n in extra_names:
        V[f"X  {n}"] = lambda r, n=n: keep_largest(r[f"p_x:{n}"] > 0.5)
        V[f"F  融合 {n}"] = lambda r, n=n: keep_largest(0.5 * r[f"p_x:{n}"] + 0.5 * r["p_m"] > 0.5)
    return V


# ================================================================ 學習式融合（只用推論時拿得到的資訊）

def axis_t(r):
    """沿 Mask R-CNN 遮罩主軸的位置 t（0 = 根端、1 = 冠端，冠端以較亮的一端判定）。不用 GT。"""
    mm = r["p_m"] > 0.5
    if mm.sum() < 20:
        return np.full(mm.shape, 0.5, np.float32)
    ys, xs = np.nonzero(mm)
    c = np.array([ys.mean(), xs.mean()])
    u = np.linalg.svd(np.stack([ys, xs], 1) - c, full_matrices=False)[2][0]
    yy, xx = np.mgrid[0:mm.shape[0], 0:mm.shape[1]]
    proj = (yy - c[0]) * u[0] + (xx - c[1]) * u[1]
    lo, hi = proj[mm].min(), proj[mm].max()
    t = (proj - lo) / max(hi - lo, 1e-6)
    g = r["gray"].astype(np.float32)
    if g[mm & (t > 0.8)].mean() < g[mm & (t < 0.2)].mean():
        t = 1 - t
    return np.clip(t, -0.3, 1.3).astype(np.float32)


def region_of(t):
    return np.where(t > 0.8, 2, np.where(t < 0.2, 0, 1))   # 0 根端、1 側邊、2 冠端


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def features(r, t):
    g = r["gray"].astype(np.float32) / 255.0
    sob = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    lm, lj = logit(r["p_m"]), logit(r["p_j"])
    tc = np.clip(t, 0, 1)
    return np.stack([np.ones_like(lm), lm, lj, g, sob, tc, tc ** 2, lm * tc, lj * tc,
                     np.abs(lm - lj)], -1).astype(np.float32)


def band(r, width=10):
    """兩個模型邊界附近的像素（推論時可得，不用 GT），學習只在這裡做。"""
    out = np.zeros(r["p_m"].shape, bool)
    for k in ("p_m", "p_j"):
        m = (r[k] > 0.5).astype(np.uint8)
        e = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        d = cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, 3)
        out |= d <= width
    return out


def fit_logistic(X, y, l2=1e-3):
    from scipy.optimize import minimize

    def f(w):
        z = X @ w
        loss = np.mean(np.logaddexp(0, z) - y * z) + l2 * (w[1:] ** 2).sum()
        p = 1 / (1 + np.exp(-z))
        grad = X.T @ (p - y) / len(y)
        grad[1:] += 2 * l2 * w[1:]
        return loss, grad
    w0 = np.zeros(X.shape[1])
    return minimize(f, w0, jac=True, method="L-BFGS-B").x


def cross_fit(prepped, rng):
    """以 fold 為單位交叉擬合：每折用其他四折的牙學參數。回傳 {(fold,image,tooth): 遮罩}。"""
    out_stack, out_region = {}, {}
    folds = sorted({p["fold"] for p in prepped})
    grid = np.linspace(0, 1, 11)
    for f in folds:
        tr = [p for p in prepped if p["fold"] != f]
        # 邏輯迴歸：每顆牙在邊界帶內最多抽 1500 點
        Xs, ys = [], []
        for p in tr:
            idx = np.flatnonzero(p["band"].ravel())
            if len(idx) > 1500:
                idx = rng.choice(idx, 1500, replace=False)
            Xs.append(p["X"].reshape(-1, p["X"].shape[-1])[idx])
            ys.append(p["gt"].ravel()[idx].astype(np.float32))
        w = fit_logistic(np.concatenate(Xs), np.concatenate(ys))
        # 區域權重：每個區域挑使錯誤像素最少的 w_HR
        best = {}
        for reg in (0, 1, 2):
            errs = []
            for wj in grid:
                e = 0
                for p in tr:
                    msk = p["reg"] == reg
                    pred = (wj * p["p_j"] + (1 - wj) * p["p_m"]) > 0.5
                    e += int(((pred ^ p["gt"]) & msk).sum())
                errs.append(e)
            best[reg] = grid[int(np.argmin(errs))]
        for p in [q for q in prepped if q["fold"] == f]:
            key = (p["fold"], p["image"], p["tooth"])
            z = p["X"] @ w
            prob = 1 / (1 + np.exp(-z))
            fused = 0.5 * p["p_j"] + 0.5 * p["p_m"]
            prob = np.where(p["band"], prob, fused)          # 帶外照舊融合
            out_stack[key] = keep_largest(prob > 0.5)
            wmap = np.vectorize(best.get)(p["reg"]).astype(np.float32)
            out_region[key] = keep_largest((wmap * p["p_j"] + (1 - wmap) * p["p_m"]) > 0.5)
        print(f"  fold {f}：區域權重 w_HR 根 {best[0]:.1f} / 側 {best[1]:.1f} / 冠 {best[2]:.1f}", flush=True)
    return out_stack, out_region


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
    ap.add_argument("--extra", action="append", default=[], metavar="名稱=目錄[@高,寬]")
    args = ap.parse_args()
    extras = [parse_extra(x) for x in args.extra]
    warnings.filterwarnings("ignore")

    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / ("cache_oof" + "".join(f"_{n}" for n, _, _ in extras) + ".pkl.gz")
    if args.reuse and cache.exists():
        teeth = pickle.load(gzip.open(cache, "rb"))
        print(f"讀取快取：{len(teeth)} 筆")
    else:
        teeth = build_cache(args.ckpt_jit, args.ckpt_base, extras)
        with gzip.open(cache, "wb") as fh:
            pickle.dump(teeth, fh)

    ok = [t for t in teeth if t["rec"] is not None]
    print(f"\n{len(teeth)} 顆牙，Mask R-CNN 配到 {len(ok)}，漏檢 {len(teeth) - len(ok)}")
    V = variants([n for n, _, _ in extras])

    # 學習式融合（交叉擬合，不用 GT 以外的未來資訊）
    print("\n學習式融合：交叉擬合中……", flush=True)
    rng = np.random.default_rng(0)
    prepped = []
    for t in ok:
        r = {k: (v if k == "gray" else np.asarray(v, np.float32)) for k, v in t["rec"].items()}
        tt = axis_t(r)
        prepped.append({"fold": t["fold"], "image": t["image"], "tooth": t["tooth"],
                        "p_m": r["p_m"], "p_j": r["p_j"], "gt": r["gt"] > 0.5,
                        "X": features(r, tt), "band": band(r), "reg": region_of(tt), "t": tt})
    stack, regw = cross_fit(prepped, rng)
    pk = {(p["fold"], p["image"], p["tooth"]): p for p in prepped}

    def conf_w(r):
        cm, cj = np.abs(r["p_m"] - 0.5), np.abs(r["p_j"] - 0.5)
        return keep_largest((cm * r["p_m"] + cj * r["p_j"]) / (cm + cj + 1e-6) > 0.5)
    V["F  依信心加權"] = conf_w
    V["F  依區域加權（學習）"] = None
    V["F  邏輯迴歸融合（學習）"] = None

    rows, t0 = [], time.time()
    reg_err = np.zeros((3, 3))     # [區域, 兩者都錯/只有①錯/只有HRNet錯]
    for n, t in enumerate(ok, 1):
        r = {k: (v if k == "gray" else np.asarray(v, np.float32)) for k, v in t["rec"].items()}
        gt = r["gt"] > 0.5
        row = {"fold": t["fold"], "image": t["image"], "tooth": t["tooth"]}
        preds = {}
        key = (t["fold"], t["image"], t["tooth"])
        for name, fn in V.items():
            if name == "F  依區域加權（學習）":
                p = regw[key]
            elif name == "F  邏輯迴歸融合（學習）":
                p = stack[key]
            else:
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
        em_, ej_ = preds["M  Mask R-CNN"] ^ gt, preds["J  框擾動 HRNet"] ^ gt
        reg = pk[key]["reg"]
        for k_ in (0, 1, 2):
            msk = reg == k_
            reg_err[k_] += [(em_ & ej_ & msk).sum(), (em_ & ~ej_ & msk).sum(), (~em_ & ej_ & msk).sum()]
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
    print("\n  依區域（以 Mask R-CNN 遮罩主軸判定，較亮端為冠）：")
    for k_, lab in ((2, "冠端"), (1, "側邊"), (0, "根端")):
        a, m_, j_ = reg_err[k_]
        s_ = a + m_ + j_
        print(f"    {lab}　佔全部錯誤 {s_ / reg_err.sum():5.1%}　其中 兩者都錯 {a / max(s_, 1):5.1%}　"
              f"只有①錯 {m_ / max(s_, 1):5.1%}　只有HRNet錯 {j_ / max(s_, 1):5.1%}")
    sh = np.sort(col("both"))[::-1]
    k10 = max(1, len(sh) // 10)
    print(f"\n  「兩者都錯」的集中度：最差 10%（{k10} 顆）的牙貢獻了 {sh[:k10].sum() / max(sh.sum(), 1):.0%}")
    worst = sorted(rows, key=lambda r: -r["both"])[:k10]
    print("    " + "、".join(f"{r['image']}#{r['tooth']}" for r in worst))
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
    print(f"\n  與現行融合配對（HD95）：")
    bf = col(f"{F0}|hd95")
    for name in ["F  依信心加權", "F  依區域加權（學習）", "F  邏輯迴歸融合（學習）",
                 "F  融合 w_HR=0.4"] + [f"F  融合 {n}" for n, _, _ in extras]:
        h = col(f"{name}|hd95")
        okm = np.isfinite(h) & np.isfinite(bf)
        d = h[okm] - bf[okm]
        try:
            p = wilcoxon(h[okm], bf[okm]).pvalue if np.any(d != 0) else np.nan
        except ValueError:
            p = np.nan
        print(f"    {name:<22} 平均差 {d.mean():+.2f} px　勝 {(d < 0).sum()}/負 {(d > 0).sum()}　"
              f"p={'' if np.isnan(p) else f'{p:.4f}'}　Dice 平均差 "
              f"{(col(name + '|dice') - col(F0 + '|dice')).mean():+.4f}")
    print(f"\n逐顆明細 → {OUT / 'teeth_oof.csv'}")


if __name__ == "__main__":
    main()
