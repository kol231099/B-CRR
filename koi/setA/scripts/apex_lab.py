"""根尖診斷：預測遮罩的根尖跟 GT 差多少、偏短還是偏長、是哪一段流程造成的。

放在 koi/setA/final/scripts/ 執行，只做推論，不訓練。

為什麼要先診斷
--------------
CRR 量測的 G 點（根尖）完全由遮罩決定：取遮罩沿長軸最靠根尖 1% 像素的中位數
（scripts/find_axis_paper.py 的 apex_point）。所以根尖標錯只有三種可能：

    A. 遮罩在根尖段偏短（最常見：根尖段對比低，網路提早收尾）
    B. 遮罩偏長或長歪（吃到骨頭、鄰牙或另一個牙根）
    C. GT 本身在根尖就標短了（模型只是學到標註的習慣）

而遮罩又是兩個模型的融合 mask = clean(0.5·P_HRNet + 0.5·P_MaskRCNN > 0.5)。
Mask R-CNN 的遮罩頭只有 28×28：一顆 800 px 長的牙，每格約 29 px，細窄的根尖會被
平均掉、機率掉到 0.5 以下；融合時只要 Mask R-CNN 在根尖說「不是」，HRNet 說「是」
也只剩 0.5，過不了門檻——**融合可能正是把根尖切短的元兇**。本腳本把三個來源拆開量：

    M     Mask R-CNN 單獨
    E     框擾動 HRNet 單獨（同一個斜框）
    F     融合（目前論文的方法）
    FA    根尖感知融合（候選修正，不需重訓）：根尖段只聽 HRNet，其餘照 F 融合。
          根尖段由 F 自己的長軸決定（從根尖起算 25% 牙長全用 HRNet，25–40% 線性過渡），
          不使用任何 GT 資訊

根尖誤差（以 GT 遮罩的長軸為準，兩邊用同一條軸，只比遮罩本身的差異）：
    axial    沿長軸的誤差，**正 = 預測根尖比 GT 靠牙冠（遮罩偏短）**，負 = 偏長
    lateral  垂直長軸的偏移
    dist     歐氏距離

**選變體只看 OOF**（--split oof），holdout 只在最後確認一次。

用法（在 koi/setA/final 底下）：
    python3 scripts/apex_lab.py --split oof
    python3 scripts/apex_lab.py --split oof --vis 20            # 另存最差 20 顆的根尖放大圖
    python3 scripts/apex_lab.py --split oof --only 1 13 17 --vis 99
    python3 scripts/apex_lab.py --split holdout
    python3 scripts/apex_lab.py --split oof --full --teeth 81_0 22_1 1_1   # 整張原圖，不裁切不調對比
輸出：eval/apex_<split>.csv（逐顆）、eval/apex_vis_<split>/（放大圖，含病人影像，只存本機）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_fuse_eval as F  # noqa: E402  沿用 final 的推論路徑（與 Table 1 相同）
from eval_seg2_holdout import HOLD, gt_mask  # noqa: E402
from make_crops_obb import obb_of  # noqa: E402
from metrics import match  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, ROOT  # noqa: E402

if ROOT.name != "final":
    sys.exit(f"⚠ 這支腳本必須放在 koi/setA/final/scripts/ 執行，目前的根目錄是 {ROOT}。")
EVAL = ROOT / "eval"
VARIANTS = ["M", "E", "F", "FA"]
NAMES = {"M": "Mask R-CNN", "E": "HRNet 單獨", "F": "融合（現行）", "FA": "根尖感知融合"}
APEX_FULL, APEX_RAMP = 0.25, 0.40   # FA：根尖 25% 全用 HRNet，25–40% 過渡


# ───────────────────────── 幾何 ─────────────────────────
def axis_of(m):
    """回傳 (形心, 指向牙冠的單位長軸)。牙冠端 = 較寬的一半（與 stratify.py 同一判準）。"""
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    mu = pts.mean(0)
    d = pts - mu
    c = np.array([[np.mean(d[:, 0] * d[:, 0]), np.mean(d[:, 0] * d[:, 1])],
                  [np.mean(d[:, 0] * d[:, 1]), np.mean(d[:, 1] * d[:, 1])]])
    ax = np.linalg.eigh(c)[1][:, -1]
    proj = d[:, 0] * ax[0] + d[:, 1] * ax[1]
    lat = -d[:, 0] * ax[1] + d[:, 1] * ax[0]
    if (proj < 0).sum() > 10 and (proj >= 0).sum() > 10 and lat[proj < 0].std() > lat[proj >= 0].std():
        ax = -ax
    return mu, ax


def apex_of(m, mu, ax, frac=0.01):
    """根尖：沿 ax 最小（最靠根尖）的 1% 像素之中位數，與 find_axis_paper.apex_point 同法。"""
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    d = pts - mu
    proj = d[:, 0] * ax[0] + d[:, 1] * ax[1]
    k = max(1, int(len(pts) * frac))
    return np.median(pts[np.argsort(proj)[:k]], axis=0)


def apex_error(pred, gt):
    mu, ax = axis_of(gt)
    g, p = apex_of(gt, mu, ax), apex_of(pred, mu, ax)
    d = p - g
    axial = float(d[0] * ax[0] + d[1] * ax[1])
    lateral = float(-d[0] * ax[1] + d[1] * ax[0])
    return axial, lateral, float(np.hypot(*d)), p, g


def apex_aware(p_mr, p_hr, fused_mask):
    """根尖段只用 HRNet 機率，其餘照原融合；根尖段由融合遮罩自己的長軸決定。"""
    fused = 0.5 * p_hr + 0.5 * p_mr
    if fused_mask.sum() < 50:
        return fused_mask
    mu, ax = axis_of(fused_mask)
    ys, xs = np.nonzero(fused_mask)
    proj = (xs - mu[0]) * ax[0] + (ys - mu[1]) * ax[1]
    tip, length = proj.min(), proj.max() - proj.min()
    h, w = fused.shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = ((gx - mu[0]) * ax[0] + (gy - mu[1]) * ax[1] - tip) / max(length, 1.0)
    wgt = np.clip((APEX_RAMP - t) / (APEX_RAMP - APEX_FULL), 0, 1).astype(np.float32)
    prob = wgt * p_hr + (1 - wgt) * fused
    return np.asarray(clean_mask(prob > 0.5), bool)


# ───────────────────────── 資料 ─────────────────────────
def samples(split, fold, only):
    """回傳 [(影像路徑, 影像資訊, [GT 標註])]。OOF 用該折驗證影像；holdout 每折都評。"""
    if split == "holdout":
        coco, img_dir = json.loads((ANN / "holdout.json").read_text(encoding="utf-8")), HOLD
    else:
        coco = json.loads((ANN / f"fold{fold}_val.json").read_text(encoding="utf-8"))
        img_dir = ROOT / "images"
    per: dict = {}
    for a in coco["annotations"]:
        if not a.get("iscrowd"):
            per.setdefault(a["image_id"], []).append(a)
    out = []
    for im in coco["images"]:
        if only and Path(im["file_name"]).stem not in only:
            continue
        if im["id"] in per:
            out.append((img_dir / im["file_name"], im, per[im["id"]]))
    return out


# ───────────────────────── 主程式 ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["oof", "holdout"], default="oof")
    ap.add_argument("--ckpt-dir", default="checkpoints_obb_jit")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--only", nargs="+", default=[], metavar="IMG", help="只跑這些影像（主檔名）")
    ap.add_argument("--vis", type=int, default=0, help="另存最差 N 顆（依 F 的根尖距離）的放大圖")
    ap.add_argument("--full", action="store_true",
                    help="改存整張原圖（不裁切、不放大、不調對比），把所有輪廓與根尖點畫上去")
    ap.add_argument("--teeth", nargs="+", default=[], metavar="IMG_IDX",
                    help="只畫這幾顆牙，例如 81_0 22_1（影像主檔名_牙號）；不受 --vis 數量限制")
    args = ap.parse_args()

    rows, vis_items = [], []
    for fold in range(5):
        items = samples(args.split, fold, set(args.only))
        if not items:
            continue
        seg = F.load_jit(args.ckpt_dir, fold)
        print(f"fold {fold}：{len(items)} 張", flush=True)
        for path, im, anns in items:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"  ⚠ 讀不到 {path}")
                continue
            h, w = gray.shape
            gt = np.stack([gt_mask(a, h, w) for a in anns])
            prob, sc = F.mr_probs(fold, gray, args.thr, False)
            masks = {v: [] for v in VARIANTS}
            probs, keep = [], []
            for i, p in enumerate(prob):
                cm = np.asarray(clean_mask(p > 0.5), bool)
                if not cm.any():
                    continue
                pe = F.hr_prob(seg, gray, obb_of(cm.astype(np.uint8)), False)
                fm = np.asarray(clean_mask(0.5 * pe + 0.5 * p > 0.5), bool)
                masks["M"].append(cm)
                masks["E"].append(np.asarray(clean_mask(pe > 0.5), bool))
                masks["F"].append(fm)
                masks["FA"].append(apex_aware(p, pe, fm))
                keep.append(i)
            if not keep:
                continue
            for v in VARIANTS:
                arr = np.array(masks[v], bool).reshape(-1, h, w)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rr, matched = match(arr, sc[keep], gt, im["file_name"])
                tp = {r["gt_idx"]: r for r in rr if r["kind"] == "TP"}
                for pi, gi in matched.items():
                    if not arr[pi].any():
                        continue
                    axial, lateral, dist, pa, ga = apex_error(arr[pi], gt[gi])
                    r = tp.get(gi, {})
                    rows.append({"split": args.split, "fold": fold, "image": im["file_name"], "gt_idx": gi,
                                 "variant": v, "axial": round(axial, 2), "lateral": round(lateral, 2),
                                 "dist": round(dist, 2), "dice": r.get("dice", ""), "hd95": r.get("hd95", "")})
                    tid = f"{Path(im['file_name']).stem}_{gi}"
                    want = (tid in args.teeth) if args.teeth else bool(args.vis)
                    if want and v == "F" and args.full:
                        # 整張原圖模式：只存輪廓與點（不存整張遮罩，避免吃光記憶體）
                        mu, ax = axis_of(gt[gi])
                        cont = {k: cv2.findContours(masks[k][pi].astype(np.uint8), cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)[0] for k in VARIANTS}
                        cont["GT"] = cv2.findContours(gt[gi].astype(np.uint8), cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_NONE)[0]
                        pts = {k: apex_of(masks[k][pi], mu, ax) for k in VARIANTS}
                        vis_items.append((dist, fold, im["file_name"], gi, ga, path, cont, pts))
                    elif want and v == "F":
                        # 只存根尖附近 ±140 px 的小塊，避免整張遮罩吃光記憶體
                        x0, y0 = int(max(ga[0] - 140, 0)), int(max(ga[1] - 140, 0))
                        x1, y1 = int(min(ga[0] + 140, w)), int(min(ga[1] + 140, h))
                        mu, ax = axis_of(gt[gi])
                        pts = {k: apex_of(masks[k][pi], mu, ax) - (x0, y0) for k in VARIANTS}
                        vis_items.append((dist, fold, im["file_name"], gi, ga - (x0, y0),
                                          gray[y0:y1, x0:x1].copy(), gt[gi][y0:y1, x0:x1].copy(),
                                          {k: masks[k][pi][y0:y1, x0:x1].copy() for k in VARIANTS}, pts))
    if not rows:
        sys.exit("沒有任何結果；請確認影像路徑（OOF 用 images/，holdout 用 holdout/）。")

    EVAL.mkdir(exist_ok=True)
    # 有 --only / --teeth 時另存檔名，不覆蓋完整的結果
    partial = "_partial" if (args.only or args.teeth) else ""
    out = EVAL / f"apex_{args.split}{partial}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    summarize(rows, args.split)
    if vis_items and args.full:
        save_full(vis_items, args.split, len(vis_items) if args.teeth else args.vis)
    elif vis_items:
        save_vis(vis_items, args.split, len(vis_items) if args.teeth else args.vis)
    print(f"\n逐顆結果 → {out}")


def summarize(rows, split):
    """holdout 同一顆牙五折都有，先取五折平均再統計，避免把五折當成獨立樣本。"""
    from scipy.stats import wilcoxon

    per: dict = {}
    for r in rows:
        per.setdefault(r["variant"], {}).setdefault((r["image"], r["gt_idx"]), []).append(r)
    keys = set.intersection(*(set(per[v]) for v in VARIANTS if v in per))
    mean = lambda v, k, f: float(np.mean([float(x[f]) for x in per[v][k] if x[f] != ""]))  # noqa: E731
    print(f"\n根尖誤差（{split}，{len(keys)} 顆牙，四個變體都偵測到者；正 axial = 預測偏短）\n")
    print(f"  {'變體':<14}{'|axial| 中位':>11}{'axial 平均':>10}{'偏短>10px':>10}{'偏短>20px':>10}"
          f"{'dist 中位':>10}{'dist P90':>9}{'Dice 中位':>10}{'HD95 中位':>10}")
    stat = {}
    for v in VARIANTS:
        ax_ = np.array([mean(v, k, "axial") for k in keys])
        di = np.array([mean(v, k, "dist") for k in keys])
        dc = np.array([mean(v, k, "dice") for k in keys])
        hd = np.array([mean(v, k, "hd95") for k in keys])
        stat[v] = di
        print(f"  {NAMES[v]:<12}{np.median(np.abs(ax_)):>11.2f}{ax_.mean():>10.2f}"
              f"{(ax_ > 10).mean():>10.0%}{(ax_ > 20).mean():>10.0%}{np.median(di):>10.2f}"
              f"{np.percentile(di, 90):>9.2f}{np.nanmedian(dc):>10.4f}{np.nanmedian(hd):>10.2f}")
    print("\n  配對比較（根尖距離 dist，每顆牙一筆；勝 = 前者較小）")
    for a, b in (("FA", "F"), ("E", "F"), ("F", "M")):
        x, y = stat[a], stat[b]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                p = wilcoxon(x, y).pvalue
            except ValueError:
                p = float("nan")
        print(f"    {NAMES[a]} vs {NAMES[b]}：勝 {int((x < y).sum())}/{len(x)}　"
              f"中位差 {np.median(x - y):+.2f} px　p = {p:.4f}")
    print("\n  判讀：F 的 axial 平均明顯為正、且 E 比 F 小 → 融合把根尖切短，FA 是對症的修正；"
          "\n        E 也一樣偏短 → 兩個模型都短，要看 GT（放大圖）或改模型本身。")


def save_full(items, split, n):
    """整張原圖：原始灰階、原始尺寸，不做任何對比調整，只把輪廓與根尖點畫上去。"""
    out = EVAL / f"apex_full_{split}"
    out.mkdir(exist_ok=True)
    color = {"GT": (0, 200, 0), "M": (255, 255, 0), "E": (0, 0, 255), "F": (0, 255, 255), "FA": (255, 0, 255)}
    for dist, fold, name, gi, g, path, cont, pts in sorted(items, key=lambda x: -x[0])[:n]:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        for k in ("M", "E", "FA", "F", "GT"):        # GT 最後畫，疊在最上層
            cv2.drawContours(img, cont[k], -1, color[k], 2 if k in ("GT", "F") else 1)
        for k in ("M", "E", "FA", "F"):
            cv2.circle(img, (int(round(pts[k][0])), int(round(pts[k][1]))), 4, color[k], -1)
        cv2.circle(img, (int(round(g[0])), int(round(g[1]))), 5, color["GT"], -1)
        cv2.imwrite(str(out / f"{dist:06.1f}_{Path(name).stem}_{gi}_f{fold}.png"), img)
    print(f"\n整張原圖 → {out}/（綠 GT、黃 融合（論文方法）、青 Mask R-CNN、紅 HRNet、洋紅 根尖感知融合；點 = 各自的根尖）")


def save_vis(items, split, n):
    out = EVAL / f"apex_vis_{split}"
    out.mkdir(exist_ok=True)
    color = {"M": (255, 255, 0), "E": (0, 0, 255), "F": (0, 255, 255), "FA": (255, 0, 255)}
    for dist, fold, name, gi, g, gray, gt, ms, pts in sorted(items, key=lambda x: -x[0])[:n]:
        img = cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        draw = lambda m, c, t: cv2.drawContours(  # noqa: E731
            img, [c2 * 2 for c2 in cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)[0]], -1, c, t)
        draw(gt, (0, 200, 0), 3)
        for v in VARIANTS:
            draw(ms[v], color[v], 1)
            cv2.circle(img, (int(pts[v][0] * 2), int(pts[v][1] * 2)), 5, color[v], -1)
        cv2.circle(img, (int(g[0] * 2), int(g[1] * 2)), 6, (0, 200, 0), -1)
        cv2.putText(img, f"{Path(name).stem} #{gi} f{fold} dist {dist:.0f}px", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(out / f"{dist:06.1f}_{Path(name).stem}_{gi}_f{fold}.png"), img)
    print(f"\n根尖放大圖 → {out}/（綠 GT、青 Mask R-CNN、紅 HRNet、黃 融合、洋紅 根尖感知融合；點 = 各自的根尖）")


if __name__ == "__main__":
    main()
