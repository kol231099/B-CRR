"""評估某個 fold 的 YOLOv11-seg，指標與 eval_maskrcnn.py 完全共用 metrics.py。

遮罩取 results.masks.xy 的多邊形而非 masks.data 的張量：後者是模型輸入解析度
（1024）下的結果，要自己還原到原圖尺寸；前者 ultralytics 已經換算回原圖座標，
少一次插值也少一個出錯的地方。

用法：
    py koi/scripts/eval_yolo.py --fold 0
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
from metrics import FIELDS, match, summarize  # noqa: E402
from postprocess import clean_mask  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMAGES, ANN, EVAL = ROOT / "images", ROOT / "annotations", ROOT / "eval"


def run(fold: int, conf: float, figures: bool) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(ROOT / "yolo_runs" / f"fold{fold}" / "weights" / "best.pt")
    coco = json.loads((ANN / f"fold{fold}_val.json").read_text(encoding="utf-8"))
    by: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by.setdefault(a["image_id"], []).append(a)

    out_dir = EVAL / f"yolo_fold{fold}"
    if figures:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for im in coco["images"]:
        h, w = im["height"], im["width"]
        gt_list = [
            cv2.fillPoly(np.zeros((h, w), np.uint8),
                         [np.array(a["segmentation"][0], np.int32).reshape(-1, 2)], 1).astype(bool)
            for a in by.get(im["id"], []) if not a["iscrowd"]
        ]
        gt = np.stack(gt_list) if gt_list else np.zeros((0, h, w), bool)

        res = model.predict(str(IMAGES / im["file_name"]), conf=conf, imgsz=1024, verbose=False)[0]
        if res.masks is None or len(res.masks.xy) == 0:
            pred, scores = np.zeros((0, h, w), bool), np.zeros(0)
        else:
            pred = np.stack([
                cv2.fillPoly(np.zeros((h, w), np.uint8), [p.astype(np.int32)], 1).astype(bool)
                for p in res.masks.xy
            ])
            scores = res.boxes.conf.cpu().numpy()

        pred = np.array([clean_mask(m) for m in pred], bool).reshape(-1, h, w)
        r, matched = match(pred, scores, gt, im["file_name"])
        rows += r

        if figures:
            vis = cv2.cvtColor(cv2.imread(str(IMAGES / im["file_name"]), cv2.IMREAD_GRAYSCALE),
                               cv2.COLOR_GRAY2BGR)
            for g in gt:
                cv2.drawContours(vis, cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_NONE)[0], -1, (255, 255, 255), 3)
            for pi, p in enumerate(pred):
                col = (0, 255, 80) if pi in matched else (60, 60, 255)
                cv2.drawContours(vis, cv2.findContours(p.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_NONE)[0], -1, col, 3)
            cv2.imwrite(str(out_dir / im["file_name"].replace(".jpg", ".png")), vis)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--conf", type=float, default=0.5, help="與 Mask R-CNN 的 score 門檻一致")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    rows = run(args.fold, args.conf, not args.no_figures)
    EVAL.mkdir(parents=True, exist_ok=True)
    with (EVAL / f"yolo_fold{args.fold}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    summarize(rows, f"yolo fold {args.fold}")
    print(f"  → {EVAL}/yolo_fold{args.fold}.csv")


if __name__ == "__main__":
    main()
