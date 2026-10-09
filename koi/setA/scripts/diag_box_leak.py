"""診斷：OBB→HRNet 的 oracle 優勢，是 HRNet 的能力，還是框洩漏了答案？

背景：Oracle OBB→HRNet（Dice 0.9782）明顯優於 Mask R-CNN（0.9739），但把框換成
Mask R-CNN 推出的斜框後，端到端只剩 0.9734，幾乎與 Mask R-CNN 本身一模一樣。

懷疑的原因：make_crops_obb.py 的框是 GT 遮罩的 minAreaRect，再固定外擴 pad=0.2，
train_seg2.py 又沒有任何幾何增強（只有翻轉與亮度）。於是每一張訓練 crop 裡，
牙齒的上、下、左、右四個極點都落在**完全相同的正規化位置**
（pad / (1 + 2·pad) ≈ 14.3% 與 85.7%）。網路不必看影像就知道牙冠尖與根尖在哪，
只要學「把框內縮 14.3% 填滿」再修一下輪廓。

若是如此：
  * oracle 的分數包含 GT 洩漏，不是 HRNet 的能力上限；
  * 換成 Mask R-CNN 的框時，HRNet 會照抄框的四個邊界，等於複製 Mask R-CNN 的
    長度/寬度誤差——所以端到端 ≈ Mask R-CNN；
  * 改 padding 只是換一個常數，洩漏仍在；改第一階段或框的導出方式也救不了，
    因為第二階段本來就只會照抄框。

不需重新訓練，三個測試：

  1. 模板基線：完全不看影像，只把「訓練集平均遮罩」依 GT 斜框貼回去。
     這個分數就是框本身帶有的資訊量。
  2. 邊界跟隨斜率：把 GT 框的某一條邊往外/往內推 d·邊長，看預測遮罩的極點跟著動
     多少。斜率 ≈ 1 表示網路照抄框；≈ 0 表示網路真的在看影像找邊界。
  3. 抖動 oracle：GT 框加上與 Mask R-CNN 框誤差同量級的雜訊，看 oracle 掉多少。
     並實測 Mask R-CNN 框相對 GT 框的誤差分布，以及 HRNet 輸出與 Mask R-CNN
     遮罩的一致度（若遠高於兩者各自對 GT 的 Dice，代表第二階段在複製第一階段）。

用法：
    py scripts/diag_box_leak.py --all
    py scripts/diag_box_leak.py --fold 0 --max-images 20
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_seg2_holdout import predict  # noqa: E402
from make_crops_obb import obb_of, warp_of  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, CKPT, ROOT, ToothDataset, build_model, collate  # noqa: E402
import train_seg2  # noqa: E402
from train_seg2 import SIZE, build_seg2, split_tag  # noqa: E402

train_seg2.CROPS = ROOT / "crops_obb"
EVAL, PAD = ROOT / "eval", 0.2
DELTAS = (-0.06, -0.03, 0.03, 0.06)        # 邊界推移量，佔該邊長的比例
JITTER = (0.01, 0.02, 0.04, 0.08)          # 抖動 oracle 的相對雜訊 σ
EDGES = ("bottom", "top", "right", "left")  # crop 座標裡的四條邊


def dice(a, b):
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def axes(box):
    """crop 的 x、y 軸在原圖中的單位向量。"""
    M, _, _ = warp_of(*box, PAD)
    R = M[:, :2]
    return R.T @ np.array([1.0, 0.0]), R.T @ np.array([0.0, 1.0])


def seg_box(seg, gray, box, hw, template=None):
    """依框裁切 → 第二階段（或模板）→ 貼回原圖，與 eval_e2e_obb.py 一致。"""
    h, w = hw
    M, cw, ch = warp_of(*box, PAD)
    if template is None:
        crop = cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR)
        prob = predict([seg], crop, False)
    else:
        prob = template
    small = cv2.resize(prob, (cw, ch), interpolation=cv2.INTER_LINEAR)
    back = cv2.warpAffine(small, cv2.invertAffineTransform(M), (w, h),
                          flags=cv2.INTER_LINEAR) > 0.5
    return clean_mask(back)


def push_edge(box, edge, d):
    """把框的一條邊往外推 d·邊長（d<0 往內），其餘三條邊不動。"""
    cx, cy, rw, rh, ang = box
    ux, uy = axes(box)
    if edge in ("bottom", "top"):
        s = 1 if edge == "bottom" else -1
        c = np.array([cx, cy]) + s * uy * d * rh / 2
        return c[0], c[1], rw, rh * (1 + d), ang
    s = 1 if edge == "right" else -1
    c = np.array([cx, cy]) + s * ux * d * rw / 2
    return c[0], c[1], rw * (1 + d), rh, ang


def extreme(mask, box, edge):
    """遮罩在 GT 框某條邊方向上的極點位置（px，以 GT 框中心為原點）。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.nan
    ux, uy = axes(box)
    p = np.stack([xs - box[0], ys - box[1]], 1)
    proj = {"bottom": p @ uy, "top": -(p @ uy), "right": p @ ux, "left": -(p @ ux)}[edge]
    return float(np.percentile(proj, 99.5))   # 99.5% 而非 max，避開單一雜點


def jitter(box, sigma, rng):
    cx, cy, rw, rh, ang = box
    ux, uy = axes(box)
    c = np.array([cx, cy]) + ux * rng.normal(0, sigma) * rw + uy * rng.normal(0, sigma) * rh
    return (c[0], c[1], rw * (1 + rng.normal(0, sigma)), rh * (1 + rng.normal(0, sigma)),
            ang + np.degrees(rng.normal(0, sigma)))


def box_err(pred, gt):
    """預測框相對 GT 框的誤差，全部以 GT 邊長正規化。"""
    ux, uy = axes(gt)
    dc = np.array([pred[0] - gt[0], pred[1] - gt[1]])
    da = ((pred[4] - gt[4] + 90) % 180) - 90
    return {"e_cx": float(dc @ ux / gt[2]), "e_cy": float(dc @ uy / gt[3]),
            "e_rw": pred[2] / gt[2] - 1, "e_rh": pred[3] / gt[3] - 1, "e_ang": float(da)}


def load_seg(tag, fold):
    arch, enc = split_tag(tag)
    m = build_seg2(arch, enc, pretrained=False)
    m.load_state_dict(torch.load(ROOT / "checkpoints_obb" / "seg2" / tag / f"fold{fold}.pt",
                                 map_location="cpu", weights_only=False)["model"])
    return m.eval()


def load_mrcnn(fold):
    ck = torch.load(CKPT / "original" / f"maskrcnn_fold{fold}.pt", map_location="cpu",
                    weights_only=False)
    m = build_model(False, ck.get("mask_res", 28))
    m.load_state_dict(ck["model"])
    return m.eval()


def mean_template(fold):
    """該折訓練 crop 的平均遮罩（512x256），不含驗證影像。"""
    tr, _ = train_seg2.fold_ids(fold)
    acc = np.zeros(SIZE, np.float64)
    for cid in tr:
        m = cv2.imread(str(train_seg2.CROPS / "masks" / f"{cid}.png"), cv2.IMREAD_GRAYSCALE)
        acc += cv2.resize(m, SIZE[::-1], interpolation=cv2.INTER_AREA) / 255.0
    return (acc / max(len(tr), 1)).astype(np.float32)


@torch.no_grad()
def run(fold, tag, thr, max_images, rng):
    seg, mr, tmpl = load_seg(tag, fold), load_mrcnn(fold), mean_template(fold)
    ds = ToothDataset(ANN / f"fold{fold}_val.json", train=False, enhance="original")
    rows = []
    for k, (imgs, targets) in enumerate(DataLoader(ds, batch_size=1, shuffle=False,
                                                   collate_fn=collate)):
        if max_images and k >= max_images:
            break
        t = targets[0]
        gts = t["masks"].numpy().astype(bool)
        hw = gts.shape[1:]
        gray = (imgs[0][0].numpy() * 255).astype(np.uint8)

        out = mr([imgs[0]])[0]
        keep = out["scores"].numpy() >= thr
        mmasks = [np.asarray(clean_mask(x), bool) for x in out["masks"].numpy()[keep, 0] > 0.5]

        for gi, gt in enumerate(gts):
            if gt.sum() < 50:
                continue
            gbox = obb_of(gt.astype(np.uint8))
            r = {"fold": fold, "image": t["_name"], "gt_idx": gi}

            oracle = seg_box(seg, gray, gbox, hw)
            r["d_oracle"] = dice(oracle, gt)
            r["d_template"] = dice(seg_box(None, gray, gbox, hw, tmpl), gt)

            # 2. 邊界跟隨斜率：極點位移 ÷ 框邊位移
            for edge in EDGES:
                side = gbox[3] if edge in ("bottom", "top") else gbox[2]
                e0 = extreme(oracle, gbox, edge)
                xs, ys = [], []
                for d in DELTAS:
                    pm = seg_box(seg, gray, push_edge(gbox, edge, d), hw)
                    xs.append(d * side)
                    ys.append(extreme(pm, gbox, edge) - e0)
                xs, ys = np.array(xs), np.array(ys)
                ok = np.isfinite(ys)
                r[f"slope_{edge}"] = float((xs[ok] @ ys[ok]) / max(xs[ok] @ xs[ok], 1e-9))

            # 3. 抖動 oracle
            for s in JITTER:
                r[f"d_jit{s:g}"] = dice(seg_box(seg, gray, jitter(gbox, s, rng), hw), gt)

            # Mask R-CNN 的框與遮罩
            ious = [(m & gt).sum() / max((m | gt).sum(), 1) for m in mmasks]
            if ious and max(ious) >= 0.5:
                mm = mmasks[int(np.argmax(ious))]
                mbox = obb_of(mm.astype(np.uint8))
                e2e = seg_box(seg, gray, mbox, hw)
                r.update(box_err(mbox, gbox))
                r["d_mrcnn"] = dice(mm, gt)
                r["d_e2e"] = dice(e2e, gt)
                r["d_e2e_vs_mrcnn"] = dice(e2e, mm)   # 第二階段有多像第一階段
            rows.append(r)
    return rows


def report(rows):
    def col(k):
        v = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])], float)
        return v

    def line(lab, k):
        v = col(k)
        print(f"  {lab:<34}{np.median(v):>9.4f}{v.mean():>9.4f}   n={len(v)}")

    print(f"\n{'=' * 72}\n1. 框本身帶多少資訊（Dice，中位 / 平均）\n{'=' * 72}")
    line("模板基線（不看影像）", "d_template")
    line("Oracle OBB→HRNet", "d_oracle")
    line("Mask R-CNN", "d_mrcnn")
    line("Mask R-CNN→OBB→HRNet", "d_e2e")
    line("HRNet 輸出 vs Mask R-CNN 遮罩", "d_e2e_vs_mrcnn")

    print(f"\n{'=' * 72}\n2. 邊界跟隨斜率（1 = 照抄框，0 = 看影像）\n{'=' * 72}")
    for e in EDGES:
        v = col(f"slope_{e}")
        print(f"  {e:<10} 中位 {np.median(v):.3f}　四分位 [{np.percentile(v, 25):.3f}, "
              f"{np.percentile(v, 75):.3f}]")

    print(f"\n{'=' * 72}\n3. Mask R-CNN 框誤差（相對 GT 邊長；角度為度）\n{'=' * 72}")
    for k in ("e_cx", "e_cy", "e_rw", "e_rh", "e_ang"):
        v = col(k)
        print(f"  {k:<8} 中位 {np.median(v):+.4f}　std {v.std():.4f}　"
              f"|誤差| 中位 {np.median(np.abs(v)):.4f}")
    print("\n  抖動 oracle：")
    line("σ = 0（原 oracle）", "d_oracle")
    for s in JITTER:
        line(f"σ = {s:g}", f"d_jit{s:g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="unet_tu-hrnet_w32")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--max-images", type=int, default=0, help="每折最多幾張，0 = 全部")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    for f in (range(5) if args.all else [args.fold]):
        rows += run(f, args.model, args.thr, args.max_images, rng)
    EVAL.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("fold", "image", "gt_idx"), k))
    with (EVAL / f"diag_box_leak_{args.model}.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    report(rows)


if __name__ == "__main__":
    main()
