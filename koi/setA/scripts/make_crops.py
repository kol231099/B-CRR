"""從整顆牙的多邊形裁出單顆牙的 ROI 小圖與遮罩。

Mask R-CNN **不需要**這些小圖——它吃整張原圖，靠 RPN 與 ROIAlign 自己在圖上
取 ROI。這裡產出的 crop 是給第二階段用的：nnU-Net 與 MedSAM / SAM 2 都是在
「一張圖只有一顆主體」的前提下做二元前景／背景。

裁切規則（三個模型共用，改動會直接改變分數，所以固定在這裡）：

    padding       往外擴 pad_frac。牙根尖與牙冠邊界常常貼著 bbox，零 padding
                  會讓模型永遠看不到牙齒外圍的骨頭對比。
    原尺寸        預設不 resize。牙齒又高又窄（實測 bbox 寬高比 0.22~0.68），
                  縮到固定邊長會壓掉邊界細節，而 HD95 量的正是邊界。
    不正方形化    曾經試過補成正方形：牙齒只佔 crop 面積的 22~68%（平均 35%），
                  其餘全是黑邊，等於把解析度浪費在空白上。需要正方形輸入的
                  MedSAM / SAM 2 自己會 letterbox，那一步留在推論腳本裡做。
    clip 到邊界   框超出影像時直接切齊，不補黑也不把框推回影像內。

manifest.csv 記下每個 crop 的來源與幾何，推論結果要靠它貼回原圖座標。評估
一律在原圖座標上算：在 crop 內算 Dice 會虛高，因為背景已經被裁掉了。

輸出到 koi/crops/：images/ 是小圖、masks/ 是對應遮罩，兩邊同名，命名為
<圖名>_t<標註id>.png。

用法：
    py koi/scripts/make_crops.py
    py koi/scripts/make_crops.py --pad 0.15
    py koi/scripts/make_crops.py --size 512     # 需要固定尺寸時才用
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "images"
ANN = ROOT / "annotations"
CROPS = ROOT / "crops"


def crop_box(bbox: list[float], pad_frac: float, w: int, h: int) -> tuple[int, int, int, int]:
    """回傳 clip 到影像範圍內的 (x0, y0, x1, y1)。"""
    x, y, bw, bh = bbox
    px, py = bw * pad_frac, bh * pad_frac
    return (
        max(0, int(round(x - px))),
        max(0, int(round(y - py))),
        min(w, int(round(x + bw + px))),
        min(h, int(round(y + bh + py))),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pad", type=float, default=0.2, help="往外擴的比例，預設 0.2")
    ap.add_argument("--size", type=int, default=0,
                    help="長邊縮到這個大小（等比例、不補邊）。0 表示保持原尺寸")
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

    rows, touched = [], 0
    for iid, anns in sorted(by.items()):
        im = imgs[iid]
        gray = cv2.imread(str(IMAGES / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(IMAGES / im["file_name"])

        for a in anns:
            if a["iscrowd"]:  # ignore 區不產生 crop
                continue
            full = np.zeros(gray.shape, np.uint8)
            cv2.fillPoly(full, [np.array(a["segmentation"][0], np.int32).reshape(-1, 2)], 255)

            x0, y0, x1, y1 = crop_box(a["bbox"], args.pad, im["width"], im["height"])
            crop, mask = gray[y0:y1, x0:x1], full[y0:y1, x0:x1]
            touched += bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())

            scale = 1.0
            if args.size:
                scale = args.size / max(crop.shape)
                wh = (int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale)))
                # 遮罩用 INTER_NEAREST：插值會在邊界產生灰階值，二值化後邊界會漂移
                crop = cv2.resize(crop, wh, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, wh, interpolation=cv2.INTER_NEAREST)

            stem = f"{Path(im['file_name']).stem}_t{a['id']:04d}"
            cv2.imwrite(str(CROPS / "images" / f"{stem}.png"), crop)
            cv2.imwrite(str(CROPS / "masks" / f"{stem}.png"), mask)
            rows.append(
                {
                    "crop_id": stem,
                    "image": im["file_name"],
                    "ann_id": a["id"],
                    "img_w": im["width"],
                    "img_h": im["height"],
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "crop_w": crop.shape[1],
                    "crop_h": crop.shape[0],
                    "scale": round(scale, 6),
                    "mask_px": int((mask > 0).sum()),
                }
            )

    with (CROPS / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fill = [r["mask_px"] / (r["crop_w"] * r["crop_h"]) for r in rows]
    print(f"產生 {len(rows)} 個 crop（pad {args.pad:.0%}"
          f"{'，長邊 ' + str(args.size) if args.size else '，原尺寸'}）")
    print(f"尺寸 {min(r['crop_w'] for r in rows)}x{min(r['crop_h'] for r in rows)} ~ "
          f"{max(r['crop_w'] for r in rows)}x{max(r['crop_h'] for r in rows)}")
    print(f"牙齒佔 crop 面積 {min(fill):.0%}~{max(fill):.0%}（平均 {sum(fill) / len(fill):.0%}）")
    print(f"{touched} 個 crop 的牙齒碰到邊界（padding 被影像邊界切掉）")
    print(f"輸出 → {CROPS}/")


if __name__ == "__main__":
    main()
