"""比較各種影像增強對 SAM 2 分割品質的影響。

SAM 2 是零訓練的，11 種增強共用同一份權重——所以這個實驗**不需要訓練**，
但結果仍然會不同，因為餵進去的影像不同。這是整個增強比較裡唯一免費的一半。

用 GT bbox 當 prompt（oracle 條件），把偵測階段的變異隔離掉，純粹測「影像處理
有沒有讓 SAM 2 的邊界更準」。指標與其他實驗共用 metrics.py，一律在原圖座標上算。

用法：
    py koi/scripts/eval_enhance_sam2.py
    py koi/scripts/eval_enhance_sam2.py --methods clahe,clahe+unsharp
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weights import weight  # noqa: E402
from enhance import METHODS  # noqa: E402
from eval_sam2 import boxes_from_gt  # noqa: E402
from metrics import hd95  # noqa: E402
from postprocess import clean_mask  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMAGES, ANN, EVAL = ROOT / "images", ROOT / "annotations", ROOT / "eval"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default="", help="逗號分隔，預設全部")
    args = ap.parse_args()

    from ultralytics import SAM

    names = [s.strip() for s in args.methods.split(",") if s.strip()] or list(METHODS)
    sam = SAM(weight("sam2.1_b.pt"))
    data = json.loads((ANN / "instances_all.json").read_text(encoding="utf-8"))
    by: dict[int, list[dict]] = {}
    for a in data["annotations"]:
        by.setdefault(a["image_id"], []).append(a)

    tmp = ROOT / "_enh_tmp.png"
    rows, per_tooth = [], []
    for name in names:
        fn = METHODS[name]
        dices, hds, fails = [], [], 0
        for im in data["images"]:
            h, w = im["height"], im["width"]
            anns = [a for a in by.get(im["id"], []) if not a["iscrowd"]]
            gt = np.stack([cv2.fillPoly(np.zeros((h, w), np.uint8),
                                        [np.array(a["segmentation"][0], np.int32).reshape(-1, 2)], 1).astype(bool)
                           for a in anns])
            boxes, _ = boxes_from_gt(anns)
            cv2.imwrite(str(tmp), fn(cv2.imread(str(IMAGES / im["file_name"]), cv2.IMREAD_GRAYSCALE)))
            r = sam.predict(str(tmp), bboxes=boxes.tolist(), verbose=False)[0]
            m = r.masks.data.cpu().numpy() > 0.5
            pred = (np.stack([cv2.resize(x.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                              for x in m]) if m.shape[1:] != (h, w) else m)
            for i in range(min(len(pred), len(gt))):
                p = clean_mask(pred[i])
                inter = np.logical_and(p, gt[i]).sum()
                iou = inter / max(np.logical_or(p, gt[i]).sum(), 1)
                if iou < 0.5:
                    fails += 1
                    per_tooth.append(dict(method=name, image=im["file_name"], gt_idx=i,
                                          dice="", hd95=""))
                    continue
                d = 2 * inter / (p.sum() + gt[i].sum())
                hv = hd95(p, gt[i])
                dices.append(d); hds.append(hv)
                per_tooth.append(dict(method=name, image=im["file_name"], gt_idx=i,
                                      dice=round(float(d), 5), hd95=round(float(hv), 3)))
        d, hh = np.array(dices), np.array(hds)
        rows.append(dict(method=name, n=len(d), fails=fails,
                         dice_med=float(np.median(d)), dice_mean=float(d.mean()),
                         hd95_med=float(np.nanmedian(hh))))
        print(f"  {name:<18} Dice {np.median(d):.4f}　HD95 {np.nanmedian(hh):5.1f} px"
              f"　失敗 {fails}", flush=True)
    tmp.unlink(missing_ok=True)

    EVAL.mkdir(parents=True, exist_ok=True)
    with (EVAL / "enhance_sam2.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (EVAL / "enhance_sam2_per_tooth.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "image", "gt_idx", "dice", "hd95"])
        w.writeheader()
        w.writerows(per_tooth)

    base = next(r for r in rows if r["method"] == "original")
    print(f"\n{'方法':<18}{'Dice':>9}{'vs 原圖':>11}{'HD95':>9}{'vs 原圖':>11}{'失敗':>6}")
    print("-" * 66)
    for r in sorted(rows, key=lambda r: -r["dice_med"]):
        print(f"{r['method']:<18}{r['dice_med']:>9.4f}{r['dice_med']-base['dice_med']:>+11.4f}"
              f"{r['hd95_med']:>9.1f}{r['hd95_med']-base['hd95_med']:>+11.1f}{r['fails']:>6}")


if __name__ == "__main__":
    main()
