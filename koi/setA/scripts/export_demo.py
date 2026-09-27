"""把一張測試片的幾何結果匯出成 JSON，供網站的 demo 動畫用。

網站上那段「逐步顯現」的動畫需要的是**座標**，不是圖片：旋轉框要能一條一條
描出來、遮罩外要能漸漸淡掉、特徵點要一顆一顆亮起、講 CRR 時要能讓 J、L、K
單獨發光。這些都得用向量畫，所以這支把下列東西通通輸出成同一個座標系下的
數值，前端再照著畫：

    obb    旋轉框的四個角
    poly   牙齒遮罩的輪廓
    pts    A–G 七個量到的點
    lv     H I J K L R Q S 的軸上位置，以及各自那條平行 CD 的高度線
    axis   主軸的兩個端點

用法：
    py koi/setA/scripts/export_demo.py --image 113 --out demo.json --png pa.webp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from make_steps import (                                            # noqa: E402
    DEFAULTS, detect_obb, load_detector, load_segmenters, segment_in_obb,
)
from train_maskrcnn import ROOT                                     # noqa: E402

from scripts.measure import measure, mask_points_of                 # noqa: E402


def contour_of(mask: np.ndarray, eps: float = 1.2) -> list:
    """遮罩最外層輪廓，稍微簡化以免 JSON 太肥。"""
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, eps, True)
    return [[round(float(p[0][0]), 1), round(float(p[0][1]), 1)] for p in c]


def geometry(img: np.ndarray, mask: np.ndarray, box) -> dict:
    """一顆牙的全部幾何。座標一律是原圖像素。"""
    r = measure("demo", img, mask, **DEFAULTS)
    frame, slope = r.axis.frame, r.axis.cd_slope
    points = mask_points_of(mask)
    xs, ys = frame.to_frame(points)
    half = float(np.abs(xs).max()) * 1.30

    def xy(p):
        return [round(float(p[0]), 1), round(float(p[1]), 1)]

    def level(key):
        v = r.levels[key]
        ends = frame.to_image(np.array([-half, half]),
                             np.array([v - half * slope, v + half * slope]))
        return {"p": xy(frame.point_at(v)), "line": [xy(ends[0]), xy(ends[1])]}

    quad = cv2.boxPoints(((box[0], box[1]), (box[2], box[3]), box[4]))
    tip, tail = frame.point_at(float(ys.max())), frame.point_at(float(ys.min()))
    return {
        "obb": [xy(p) for p in quad],
        "poly": contour_of(mask),
        "axis": [xy(tail), xy(tip)],
        "pts": {k: xy(v) for k, v in r.landmarks.items()},
        "lv": {k: level(k) for k in ("H", "I", "J", "K", "L", "R", "Q", "S")},
        "crr": round(r.crr, 4),
        "ablr": round(r.ablr, 4),
        "bcrr": round(r.b_crr, 4),
        "max_blr": round(r.max_blr, 4),
    }


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="113")
    ap.add_argument("--out", required=True, help="JSON 輸出路徑")
    ap.add_argument("--png", help="順便把原圖存成 webp 給網站用")
    ap.add_argument("--width", type=int, default=1100, help="原圖輸出寬度")
    args = ap.parse_args()

    src = ROOT / "testset" / f"{args.image}.jpg"
    gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"讀不到 {src}")

    print("載入模型…")
    det, segs = load_detector(), load_segmenters()
    boxes = sorted(detect_obb(det, gray), key=lambda b: b[0])
    teeth = []
    for i, b in enumerate(boxes):
        m = segment_in_obb(segs, gray, b)
        if not m.any():
            continue
        try:
            teeth.append(geometry(gray, m, b))
            print(f"  牙 {i + 1}: CRR {teeth[-1]['crr']}  B-CRR {teeth[-1]['bcrr']}")
        except Exception as exc:                      # noqa: BLE001
            print(f"  牙 {i + 1}: 幾何推導失敗 — {exc}")

    h, w = gray.shape
    data = {"image": args.image, "size": [w, h], "teeth": teeth}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"{out}  {out.stat().st_size // 1024} KB  {len(teeth)} 顆牙")

    if args.png:
        s = args.width / w
        im = cv2.resize(gray, (args.width, int(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(args.png, im, [cv2.IMWRITE_WEBP_QUALITY, 90])
        print(f"{args.png}  {args.width}x{int(h * s)}")


if __name__ == "__main__":
    main()
