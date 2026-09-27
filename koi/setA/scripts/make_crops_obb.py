"""OBB 版的單顆牙裁切：先把牙轉正，再裁。

與 make_crops.py 的差別只有「框會旋轉」。裁切規則（padding 0.2、保持原解析度、
不正方形化）刻意完全一致，這樣 OBB vs HBB 的比較才只差在框的方向。

轉正後牙齒一律是「高的那一邊朝上」。實測長寬比中位 3.34，比 HBB 的 3.05 更遠離
第二階段輸入的 2.00——斜牙的水平框被傾角撐寬，反而更接近 2:1。也就是說縮放到
512x256 時 OBB crop 被壓扁得更多，這是 OBB 的一個小劣勢，不是優勢。

manifest 記下 (cx, cy, rw, rh, ang, cw, ch)：推論時要靠這組參數把遮罩轉回原圖。

用法：
    py scripts/make_crops_obb.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGES, ANN = ROOT / "images", ROOT / "annotations"
CROPS = ROOT / "crops_obb"


def obb_of(mask: np.ndarray) -> tuple[float, float, float, float, float]:
    """回傳 (cx, cy, 短邊, 長邊, 角度)，角度使長邊轉正後朝上。"""
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    if rw > rh:
        rw, rh, ang = rh, rw, ang + 90
    return cx, cy, rw, rh, ang


def warp_of(cx, cy, rw, rh, ang, pad):
    """回傳把原圖座標轉成「牙擺正」crop 座標的仿射矩陣，與 crop 尺寸。"""
    cw = max(8, int(round(rw * (1 + 2 * pad))))
    ch = max(8, int(round(rh * (1 + 2 * pad))))
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    M[0, 2] += cw / 2 - cx
    M[1, 2] += ch / 2 - cy
    return M, cw, ch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pad", type=float, default=0.2)
    args = ap.parse_args()

    data = json.loads((ANN / "instances_all.json").read_text(encoding="utf-8"))
    imgs = {im["id"]: im for im in data["images"]}
    by: dict[int, list[dict]] = {}
    for a in data["annotations"]:
        by.setdefault(a["image_id"], []).append(a)

    for d in (CROPS / "images", CROPS / "masks"):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.png"):
            old.unlink()

    rows, tilts = [], []
    for iid, anns in sorted(by.items()):
        im = imgs[iid]
        gray = cv2.imread(str(IMAGES / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(IMAGES / im["file_name"])
        for a in anns:
            if a["iscrowd"]:
                continue
            full = np.zeros(gray.shape, np.uint8)
            cv2.fillPoly(full, [np.array(a["segmentation"][0], np.int32).reshape(-1, 2)], 255)
            cx, cy, rw, rh, ang = obb_of(full)
            M, cw, ch = warp_of(cx, cy, rw, rh, ang, args.pad)
            crop = cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR)
            # 遮罩用 NEAREST：插值會在邊界產生灰階值，二值化後邊界會漂移
            mask = cv2.warpAffine(full, M, (cw, ch), flags=cv2.INTER_NEAREST)

            cid = f"{Path(im['file_name']).stem}_t{a['id']:04d}"
            cv2.imwrite(str(CROPS / "images" / f"{cid}.png"), crop)
            cv2.imwrite(str(CROPS / "masks" / f"{cid}.png"), mask)
            tilts.append(abs(((ang + 90) % 180) - 90))
            rows.append({"crop_id": cid, "image": im["file_name"], "ann_id": a["id"],
                         "img_w": im["width"], "img_h": im["height"],
                         "cx": round(cx, 2), "cy": round(cy, 2),
                         "rw": round(rw, 2), "rh": round(rh, 2), "ang": round(ang, 4),
                         "crop_w": cw, "crop_h": ch, "pad": args.pad,
                         "mask_px": int((mask > 0).sum())})

    with (CROPS / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    ar = np.array([r["crop_h"] / r["crop_w"] for r in rows])
    fill = np.array([r["mask_px"] / (r["crop_w"] * r["crop_h"]) for r in rows])
    print(f"{len(rows)} 個 OBB crop → {CROPS}")
    print(f"  傾角中位 {np.median(tilts):.1f}°　長寬比中位 {np.median(ar):.2f}"
          f"（第二階段輸入為 2.00）　填充率中位 {100 * np.median(fill):.1f}%")


if __name__ == "__main__":
    main()
