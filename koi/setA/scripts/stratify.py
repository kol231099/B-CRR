"""把預測誤差沿牙齒長軸、以及依影像品質拆開，判斷天花板來自模型還是影像。

為什麼要拆
----------
整體 HD95 中位 11 px 是一個混合的數字。如果誤差平均分布在整顆牙上，那是模型
的解析度或判斷力問題，值得繼續調；如果集中在根尖段，那是影像本身沒有邊界——
實測人工標註輪廓上的梯度，根尖端只有牙冠端的 46%，CNR 甚至低於 1——那麼再怎麼
改模型都突破不了，該停手。

做法
----
對每一對（預測遮罩、真實遮罩），用真實遮罩的 PCA 長軸把輪廓點正規化到 0（根尖端）
到 1（牙冠端），計算每個輪廓點到對方輪廓的距離，再依位置分段統計。這是把 HD95
拆解到解剖位置上，不是重新定義指標。

用法：
    py koi/scripts/stratify.py
    py koi/scripts/stratify.py --tag mask56
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, CKPT, IMAGES, build_model  # noqa: E402

BINS = [(0.0, 0.2, "根尖端"), (0.2, 0.4, ""), (0.4, 0.6, "中段"),
        (0.6, 0.8, ""), (0.8, 1.0, "牙冠端")]


def contour(m: np.ndarray) -> np.ndarray:
    c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(c, key=cv2.contourArea).reshape(-1, 2).astype(np.float64) if c else np.zeros((0, 2))


def axis_of(m: np.ndarray):
    """回傳長軸方向（指向牙冠）與形心。牙冠端的判準沿用 find_axis：較寬短的一半。"""
    pts = np.column_stack(np.nonzero(m)[::-1]).astype(np.float64)
    mu = pts.mean(0)
    ax = np.linalg.eigh(np.cov((pts - mu).T))[1][:, -1]
    perp = np.array([-ax[1], ax[0]])
    proj = (pts - mu) @ ax
    ratio = lambda h: (((h - mu) @ perp).var()) / max((((h - mu) @ ax).var()), 1e-6)
    if ratio(pts[proj < 0]) > ratio(pts[proj >= 0]):
        ax = -ax
    return ax, mu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="original")
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    seg = {i: [] for i in range(len(BINS))}
    by_img: dict[str, list[float]] = {}

    for k in range(5):
        f = CKPT / args.tag / f"maskrcnn_fold{k}.pt"
        if not f.exists():
            continue
        m = build_model(False, torch.load(f, map_location="cpu", weights_only=False).get("mask_res", 28))
        m.load_state_dict(torch.load(f, map_location="cpu", weights_only=False)["model"])
        m.eval()
        coco = json.loads((ANN / f"fold{k}_val.json").read_text(encoding="utf-8"))
        anns: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)

        for im in coco["images"]:
            h, w = im["height"], im["width"]
            gray = cv2.imread(str(IMAGES / im["file_name"]), cv2.IMREAD_GRAYSCALE)
            gts = [cv2.fillPoly(np.zeros((h, w), np.uint8),
                                [np.array(a["segmentation"][0], np.int32).reshape(-1, 2)], 1).astype(bool)
                   for a in anns.get(im["id"], [])]
            t = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1)
            with torch.no_grad():
                o = m([t])[0]
            keep = o["scores"].numpy() >= args.conf
            preds = [clean_mask(x) for x in o["masks"].numpy()[keep, 0] > 0.5]

            for gt in gts:
                best = max(preds, key=lambda p: np.logical_and(p, gt).sum(), default=None)
                if best is None or np.logical_and(best, gt).sum() / max(np.logical_or(best, gt).sum(), 1) < 0.5:
                    continue
                cg = contour(gt)
                if len(cg) < 20:
                    continue
                # 每個真實輪廓點到預測輪廓的距離
                dp = distance_transform_edt(~best)
                dist = dp[np.clip(cg[:, 1], 0, h - 1).astype(int), np.clip(cg[:, 0], 0, w - 1).astype(int)]
                ax, mu = axis_of(gt)
                tt = (cg - mu) @ ax
                tt = (tt - tt.min()) / max(np.ptp(tt), 1e-6)
                for i, (lo, hi, _) in enumerate(BINS):
                    s = (tt >= lo) & (tt < hi) if hi < 1.0 else (tt >= lo)
                    if s.any():
                        seg[i].append(float(np.percentile(dist[s], 95)))
                by_img.setdefault(im["file_name"], []).append(float(np.percentile(dist, 95)))
        print(f"  fold{k} 完成", flush=True)

    print(f"\n=== 邊界誤差沿長軸的分布（{args.tag}）===")
    print("每段真實輪廓點到預測輪廓距離的 95 百分位，單位 px\n")
    ref = np.median(seg[4]) if seg[4] else 1
    for i, (lo, hi, lab) in enumerate(BINS):
        v = np.array(seg[i])
        if not len(v):
            continue
        bar = "█" * int(np.median(v) / max(ref, 1e-6) * 26)
        name = lab or f"({lo:.0%}–{hi:.0%})"
        print(f"  {name:<10}{np.median(v):6.1f} px  {bar}")
    if seg[0] and seg[4]:
        print(f"\n  根尖端誤差是牙冠端的 {np.median(seg[0]) / np.median(seg[4]):.2f} 倍")

    print("\n=== 依影像清晰度分層 ===")
    rows = []
    for name, v in by_img.items():
        g = cv2.imread(str(IMAGES / name), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mag = float(np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0, 5) ** 2
                            + cv2.Sobel(g, cv2.CV_32F, 0, 1, 5) ** 2).mean())
        rows.append((mag, float(np.median(v)), name))
    rows.sort()
    n = len(rows) // 3
    for lab, grp in (("最模糊 1/3", rows[:n]), ("中間 1/3", rows[n:2 * n]), ("最清晰 1/3", rows[2 * n:])):
        if not grp:
            continue
        print(f"  {lab:<12}梯度 {np.mean([r[0] for r in grp]):5.1f}"
              f"　邊界誤差中位 {np.median([r[1] for r in grp]):5.1f} px　({len(grp)} 張)")


if __name__ == "__main__":
    main()
