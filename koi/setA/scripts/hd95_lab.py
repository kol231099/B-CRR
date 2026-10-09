"""HD95 實驗台：診斷 Mask R-CNN→OBB→HRNet 的 HD95 輸在哪，並一次比較所有推論端改法。

背景
----
holdout 上 ④ 的 ASSD（3.54）已經比 ① Mask R-CNN（3.57）好，HD95 卻輸（11.76 vs 9.97）。
平均邊界較準、最差 5% 的邊界較差——問題在「尾巴」，不在整體。這支腳本回答兩件事：

  A. 診斷：每顆牙的 HD95 是「多圈」還是「少圈」造成的、落在牙冠端、根尖端還是側邊。
     （本專案的 hd95 是區域式：預測邊界到 GT 區域、GT 邊界到預測區域的距離，
     所以前者 = 多圈，後者 = 少圈。）
  B. 變體：只改推論、不重新訓練的各種做法，跟 ① 逐顆牙配對比較。

做法：每顆牙只推論一次，把 Mask R-CNN 與 HRNet 的**機率圖**（未二值化）存成快取，
所有變體都從快取重算，不再跑模型。第二次執行（加 --reuse）只要幾秒。

配對：每顆 GT 牙取 IoU 最高（≥0.5）的那個 Mask R-CNN 偵測；①與④共用這個偵測，
因此兩者永遠比同一批牙——與 make_tables8 的「共同命中」同一精神。

指標一律用 metrics.py 的 hd95 / assd（在牙齒周圍的 ROI 上算；ROI 留 16 px 邊，
兩個遮罩都完整落在 ROI 內，距離與全圖計算相同）。

**選變體只看 OOF。** holdout 只有 26 顆牙，在上面挑最好的等於拿測試集調參。
流程：先跑 --split oof 選定一個變體，再跑 --split holdout 只為了確認。

用法：
    python3 scripts/hd95_lab.py --split oof --ckpt-dir checkpoints_obb_jit
    python3 scripts/hd95_lab.py --split oof --ckpt-dir checkpoints_obb_jit --reuse
    python3 scripts/hd95_lab.py --split holdout --ckpt-dir checkpoints_obb_jit --only 114 122 ...
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_seg2_holdout import HOLD, gt_mask  # noqa: E402
from make_crops_obb import obb_of, warp_of  # noqa: E402
from metrics import assd, hd95  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, CKPT, ROOT, ToothDataset, build_model, collate  # noqa: E402
from train_seg2 import SIZE, build_seg2, split_tag  # noqa: E402
from tta import predict_tta  # noqa: E402

PAD, MARGIN, THR = 0.2, 16, 0.35
VIEWS = [(False, False), (True, False), (False, True), (True, True)]


# ================================================================ 推論（只做一次）

def load_mrcnn(fold):
    ck = torch.load(CKPT / "original" / f"maskrcnn_fold{fold}.pt", map_location="cpu",
                    weights_only=False)
    m = build_model(False, ck.get("mask_res", 28))
    m.load_state_dict(ck["model"])
    return m.eval()


def load_seg(ckdir, tag, fold, device):
    arch, enc = split_tag(tag)
    m = build_seg2(arch, enc, pretrained=False)
    m.load_state_dict(torch.load(ROOT / ckdir / "seg2" / tag / f"fold{fold}.pt",
                                 map_location="cpu", weights_only=False)["model"])
    return m.eval().to(device)


@torch.no_grad()
def hrnet_probs(seg, gray, box, device):
    """回傳原圖座標的 HRNet 機率圖：(單次, 四向翻轉平均)。"""
    h, w = gray.shape
    M, cw, ch = warp_of(*box, PAD)
    crop = cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR)
    base = cv2.resize(crop, SIZE[::-1], interpolation=cv2.INTER_AREA)
    views = []
    for fh, fv in VIEWS:
        v = base[:, ::-1] if fh else base
        views.append(np.ascontiguousarray(v[::-1] if fv else v))
    x = torch.from_numpy(np.stack(views)).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
    out = torch.sigmoid(seg(x.to(device)))[:, 0].cpu().numpy()
    probs = []
    for o, (fh, fv) in zip(out, VIEWS):
        o = o[::-1] if fv else o
        probs.append(np.ascontiguousarray(o[:, ::-1] if fh else o))
    Mi = cv2.invertAffineTransform(M)

    def back(p):
        small = cv2.resize(p, (cw, ch), interpolation=cv2.INTER_LINEAR)
        return cv2.warpAffine(small, Mi, (w, h), flags=cv2.INTER_LINEAR)
    return back(probs[0]), back(np.mean(probs, 0))


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


@torch.no_grad()
def infer_image(mr, seg, gray, gts, device, use_tta):
    """每顆 GT 牙回傳一筆 ROI 快取；沒配到偵測的回傳 None（漏檢）。"""
    t = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1)
    out = mr([t])[0]
    keep = out["scores"].numpy() >= THR
    pm = out["masks"].numpy()[keep, 0]
    bm = [np.asarray(clean_mask(p > 0.5), bool) for p in pm]
    if use_tta:
        pt, _, _ = predict_tta(mr, gray, THR)
        bt = [p > 0.5 for p in pt]

    recs = []
    for gt in gts:
        ious = [iou(b, gt) for b in bm]
        if not ious or max(ious) < 0.5:
            recs.append(None)
            continue
        i = int(np.argmax(ious))
        box = obb_of(bm[i].astype(np.uint8))
        p_e, p_et = hrnet_probs(seg, gray, box, device)
        maps = {"gt": gt.astype(np.float32), "p_m": pm[i], "p_e": p_e, "p_et": p_et}
        if use_tta:
            j = int(np.argmax([iou(b, gt) for b in bt])) if bt else -1
            maps["p_mt"] = pt[j] if j >= 0 and iou(bt[j], gt) >= 0.5 else pm[i]
        # ROI：所有機率 > 0.05 的範圍，加邊
        any_ = np.zeros(gt.shape, bool)
        for k, v in maps.items():
            any_ |= v > 0.05
        ys, xs = np.nonzero(any_)
        y0, y1 = max(ys.min() - MARGIN, 0), min(ys.max() + MARGIN + 1, gt.shape[0])
        x0, x1 = max(xs.min() - MARGIN, 0), min(xs.max() + MARGIN + 1, gt.shape[1])
        rec = {k: v[y0:y1, x0:x1].astype(np.float16) for k, v in maps.items()}
        rec["gray"] = gray[y0:y1, x0:x1].copy()
        recs.append(rec)
    return recs


def iter_split(split, only):
    if split == "oof":
        for fold in range(5):
            ds = ToothDataset(ANN / f"fold{fold}_val.json", train=False, enhance="original")
            for imgs, targets in DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate):
                t = targets[0]
                yield [fold], t["_name"], (imgs[0][0].numpy() * 255).astype(np.uint8), \
                    t["masks"].numpy().astype(bool)
        return
    coco = json.loads((ANN / "holdout.json").read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    per: dict[int, list] = {}
    for a in coco["annotations"]:
        if not a.get("iscrowd"):
            per.setdefault(a["image_id"], []).append(a)
    for iid, anns in sorted(per.items()):
        im = imgs[iid]
        if only and Path(im["file_name"]).stem not in only:
            continue
        gray = cv2.imread(str(HOLD / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h, w = gray.shape
        # holdout 不屬於任何一折：五折權重各評一次（與 eval_holdout_all 相同）
        yield list(range(5)), im["file_name"], gray, np.stack([gt_mask(a, h, w) for a in anns])


def build_cache(args, device):
    by_fold: dict[int, list] = {}
    items = list(iter_split(args.split, args.only))
    for folds, name, gray, gts in items:
        for f in folds:
            by_fold.setdefault(f, []).append((name, gray, gts))
    teeth, t0, done = [], time.time(), 0
    total = sum(len(v) for v in by_fold.values())
    for fold in sorted(by_fold):
        mr, seg = load_mrcnn(fold), load_seg(args.ckpt_dir, args.model, fold, device)
        for name, gray, gts in by_fold[fold]:
            for gi, rec in enumerate(infer_image(mr, seg, gray, gts, device, args.tta)):
                teeth.append({"fold": fold, "image": name, "tooth": gi, "rec": rec})
            done += 1
            el = time.time() - t0
            print(f"  推論 {done}/{total}　fold {fold}　{name}　已花 {el:.0f}s　"
                  f"預估剩 {el / done * (total - done):.0f}s", flush=True)
    return teeth


# ================================================================ 變體（從快取重算）

def gauss(p, s):
    return cv2.GaussianBlur(p, (0, 0), s) if s else p


def variants(has_tta):
    """名稱 → 函式(rec) → 機率圖。全部以 0.5 二值化後再 clean_mask，除非名稱帶 t。"""
    V = {
        "M  Mask R-CNN（①）": lambda r: (r["p_m"], 0.5),
        "E  HRNet（④）": lambda r: (r["p_e"], 0.5),
        "E  翻轉TTA": lambda r: (r["p_et"], 0.5),
    }
    for t in (0.3, 0.4, 0.6, 0.7):
        V[f"E  門檻 {t}"] = lambda r, t=t: (r["p_e"], t)
    for s in (1, 2, 3):
        V[f"E  高斯 σ{s}"] = lambda r, s=s: (gauss(r["p_e"], s), 0.5)
    for w in (0.25, 0.5, 0.75):
        V[f"F  融合 w_E={w}"] = lambda r, w=w: (w * r["p_e"] + (1 - w) * r["p_m"], 0.5)
    V["F  融合 0.5 + 翻轉TTA"] = lambda r: (0.5 * r["p_et"] + 0.5 * r["p_m"], 0.5)
    V["F  融合 0.5 + 高斯 σ2"] = lambda r: (gauss(0.5 * r["p_e"] + 0.5 * r["p_m"], 2), 0.5)
    if has_tta:
        V["M  Mask R-CNN TTA（對照）"] = lambda r: (r["p_mt"], 0.5)
        V["F  融合 0.5 雙TTA"] = lambda r: (0.5 * r["p_et"] + 0.5 * r["p_mt"], 0.5)
    return V


def binarize(fn, rec):
    p, t = fn(rec)
    return np.asarray(clean_mask(np.asarray(p, np.float32) > t), bool)


# ================================================================ 診斷：HD95 落在哪

def where(pred, gt, gray):
    """回傳 (方向, 位置)：方向 = 多圈 / 少圈；位置 = 冠端 / 根端 / 側邊。"""
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

    # GT 主軸；較亮的一端是牙冠（琺瑯質）
    ys, xs = np.nonzero(gt)
    c = np.array([ys.mean(), xs.mean()])
    u = np.linalg.svd(np.stack([ys, xs], 1) - c, full_matrices=False)[2][0]
    proj = (np.stack([ys, xs], 1) - c) @ u
    lo, hi = proj.min(), proj.max()
    t = ((bad - c) @ u - lo) / max(hi - lo, 1e-6)          # 0..1
    tg = (proj - lo) / max(hi - lo, 1e-6)
    g = gray[ys, xs].astype(np.float32)
    crown_hi = g[tg > 0.8].mean() > g[tg < 0.2].mean()
    pos = np.where(t > 0.8, "高端", np.where(t < 0.2, "低端", "側邊"))
    pos = np.where(pos == "高端", "冠端" if crown_hi else "根端",
                   np.where(pos == "低端", "根端" if crown_hi else "冠端", pos))
    vals, cnt = np.unique(pos, return_counts=True)
    return d, str(vals[cnt.argmax()])


# ================================================================ 主程式

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["oof", "holdout"], default="oof")
    ap.add_argument("--only", nargs="+", default=[], metavar="IMG",
                    help="holdout 只算這些影像（主檔名）")
    ap.add_argument("--model", default="unet_tu-hrnet_w32")
    ap.add_argument("--ckpt-dir", default="checkpoints_obb_jit")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-tta", dest="tta", action="store_false",
                    help="不算 Mask R-CNN 的 TTA 對照（省時間）")
    ap.add_argument("--reuse", action="store_true", help="直接讀上次的快取，不重新推論")
    args = ap.parse_args()
    args.only = {Path(x.strip()).stem for a in args.only for x in a.split(",") if x.strip()}

    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu")
    out = ROOT / "eval" / "hd95_lab"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}_{args.ckpt_dir}" + (f"_only{len(args.only)}" if args.only else "")
    cache = out / f"cache_{tag}.pkl.gz"

    if args.reuse and cache.exists():
        teeth = pickle.load(gzip.open(cache, "rb"))
        print(f"讀取快取 {cache.name}：{len(teeth)} 筆")
    else:
        print(f"HRNet 跑在 {device}，Mask R-CNN 用 CPU" +
              ("，含 Mask R-CNN TTA 對照（較慢，可加 --no-tta）" if args.tta else ""), flush=True)
        teeth = build_cache(args, device)
        with gzip.open(cache, "wb") as fh:
            pickle.dump(teeth, fh)

    ok = [t for t in teeth if t["rec"] is not None]
    has_tta = bool(ok) and "p_mt" in ok[0]["rec"]
    V = variants(has_tta)
    print(f"\n{len(teeth)} 筆牙（{args.split}"
          f"{'，五折各評一次' if args.split == 'holdout' else ''}），"
          f"配到偵測 {len(ok)}、漏檢 {len(teeth) - len(ok)}")

    # 逐顆牙、逐變體算指標
    rows = []
    for t in ok:
        # 快取存 float16 省空間；OpenCV 的模糊不吃 float16，先轉回 float32
        r = {k: (v if k == "gray" else np.asarray(v, np.float32)) for k, v in t["rec"].items()}
        gt = r["gt"] > 0.5
        row = {"fold": t["fold"], "image": t["image"], "tooth": t["tooth"]}
        for name, fn in V.items():
            pred = binarize(fn, r)
            k = name.split("（")[0].strip()
            inter = (pred & gt).sum()
            row[f"{k}|dice"] = 2 * inter / max(pred.sum() + gt.sum(), 1)
            row[f"{k}|hd95"] = hd95(pred, gt)
            row[f"{k}|assd"] = assd(pred, gt)
            if name.startswith(("M  Mask R-CNN（", "E  HRNet")):
                row[f"{k}|where"] = "/".join(where(pred, gt, r["gray"]))
        rows.append(row)

    with (out / f"teeth_{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def col(k):
        return np.array([r[k] for r in rows], float)

    def agg(k, f=np.nanmedian):
        """OOF：全部合併；holdout：每折算一次再平均（與 make_tables8 一致）。"""
        if args.split == "oof":
            return float(f(col(k)))
        return float(np.mean([f(np.array([r[k] for r in rows if r["fold"] == fo], float))
                              for fo in sorted({r["fold"] for r in rows})]))

    base = "M  Mask R-CNN"
    keys = [n.split("（")[0].strip() for n in V]
    print(f"\n{'=' * 96}\nB. 變體比較（與 ① 逐顆牙配對；勝率 = HD95 比 ① 低的牙所佔比例）\n{'=' * 96}")
    print(f"  {'變體':<26}{'Dice中位':>9}{'HD95中位':>10}{'HD95平均':>10}{'ASSD中位':>10}"
          f"{'ΔHD95中位':>11}{'勝率':>7}{'Wilcoxon p':>12}")
    def paired(k):
        """逐顆牙的 HD95；holdout 先把同一顆牙的五折平均，配對檢定才不會把五折當成獨立樣本。"""
        acc: dict = {}
        for r in rows:
            acc.setdefault((r["image"], r["tooth"]), []).append(r[k])
        return np.array([np.nanmean(v) for v in acc.values()], float)

    b = paired(f"{base}|hd95")
    for k in keys:
        h = paired(f"{k}|hd95")
        ok_ = ~np.isnan(h) & ~np.isnan(b)
        d = h[ok_] - b[ok_]
        try:
            p = wilcoxon(h[ok_], b[ok_]).pvalue if k != base and np.any(d != 0) else np.nan
        except ValueError:
            p = np.nan
        print(f"  {k:<26}{agg(k + '|dice'):>9.4f}{agg(k + '|hd95'):>10.2f}"
              f"{agg(k + '|hd95', np.nanmean):>10.2f}{agg(k + '|assd'):>10.2f}"
              f"{np.median(d):>+11.2f}{(d < 0).mean():>7.0%}"
              f"{'' if np.isnan(p) else f'{p:.4f}':>12}")

    print(f"\n{'=' * 96}\nA. 診斷：HD95 是怎麼來的（每顆牙取決定 HD95 的那一側、最差 5% 邊界點的主要位置）\n"
          f"{'=' * 96}")
    for k in (base, "E  HRNet"):
        w = [r[f"{k}|where"] for r in rows]
        vals, cnt = np.unique(w, return_counts=True)
        order = np.argsort(-cnt)
        print(f"  {k}：" + "　".join(f"{vals[i]} {cnt[i]}" for i in order))
    worse = [r for r in rows if r["E  HRNet|hd95"] - r[f"{base}|hd95"] > 3]
    print(f"\n  ④ 比 ① 差 3 px 以上的牙：{len(worse)} 顆（共 {len(rows)}）")
    if worse:
        vals, cnt = np.unique([r["E  HRNet|where"] for r in worse], return_counts=True)
        print("    它們 ④ 的 HD95 來源：" + "　".join(f"{v} {c}" for v, c in
                                                     sorted(zip(vals, cnt), key=lambda x: -x[1])))
        for r in sorted(worse, key=lambda r: r[f"{base}|hd95"] - r["E  HRNet|hd95"])[:10]:
            print(f"    fold {r['fold']}  {r['image']}  #{r['tooth']}  "
                  f"① {r[base + '|hd95']:.1f}（{r[base + '|where']}）→ "
                  f"④ {r['E  HRNet|hd95']:.1f}（{r['E  HRNet|where']}）")
    print(f"\n逐顆牙明細 → {out / f'teeth_{tag}.csv'}")
    if args.split == "holdout":
        print("※ holdout 只用來確認 OOF 選出的變體，不要在這裡挑最好的。")


if __name__ == "__main__":
    main()
