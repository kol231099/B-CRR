"""用 SAM 2 做第二階段分割，以 box 當 prompt，零訓練。

這是第 3 條 pipeline。SAM 2 不需要訓練，所以它衡量的是「一個通用分割模型在
給定正確 ROI 的前提下，能把牙齒邊界切得多準」。

兩種 prompt，回答兩個不同的問題
--------------------------------
    --prompt gt      用 ground-truth bbox 當 prompt。這是 oracle：假設 ROI 完美，
                     純粹測 SAM 2 的分割能力。回答「裁 ROI 到底有沒有幫助」。
    --prompt yolo    用 YOLOv11-seg 預測的 bbox。這是端到端的真實表現。
                     兩者的差距就是偵測器造成的損失。

只跑其中一種會分不清失敗是分割爛還是偵測歪，所以兩種都要跑。

指標與 Mask R-CNN、YOLO 共用 metrics.py，一律在原圖座標上計算。

用法：
    py koi/scripts/eval_sam2.py --fold 0 --prompt gt
    py koi/scripts/eval_sam2.py --fold 0 --prompt yolo
    py koi/scripts/eval_sam2.py --fold 0 --prompt gt --limit 2   # 冒煙測試
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
from metrics import FIELDS, match, summarize  # noqa: E402
from postprocess import clean_mask  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMAGES, ANN, EVAL = ROOT / "images", ROOT / "annotations", ROOT / "eval"


def boxes_from_gt(anns: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    b = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                  for a in anns if not a["iscrowd"]], np.float32).reshape(-1, 4)
    return b, np.ones(len(b), np.float32)


def boxes_from_yolo(fold: int, path: Path, conf: float):
    from ultralytics import YOLO

    if not hasattr(boxes_from_yolo, "_cache"):
        boxes_from_yolo._cache = {}
    if fold not in boxes_from_yolo._cache:
        boxes_from_yolo._cache[fold] = YOLO(ROOT / "yolo_runs" / f"fold{fold}" / "weights" / "best.pt")
    r = boxes_from_yolo._cache[fold].predict(str(path), conf=conf, imgsz=1024, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()


def run(fold: int, prompt: str, model_name: str, conf: float, limit: int, figures: bool) -> list[dict]:
    from ultralytics import SAM

    sam = SAM(model_name)
    coco = json.loads((ANN / f"fold{fold}_val.json").read_text(encoding="utf-8"))
    by: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by.setdefault(a["image_id"], []).append(a)

    out_dir = EVAL / f"sam2_{prompt}_fold{fold}"
    if figures:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for im in coco["images"][: limit or None]:
        h, w = im["height"], im["width"]
        path = IMAGES / im["file_name"]
        anns = [a for a in by.get(im["id"], []) if not a["iscrowd"]]
        g = [cv2.fillPoly(np.zeros((h, w), np.uint8),
                          [np.array(a["segmentation"][0], np.int32).reshape(-1, 2)], 1).astype(bool)
             for a in anns]
        gt = np.stack(g) if g else np.zeros((0, h, w), bool)

        boxes, scores = (boxes_from_gt(anns) if prompt == "gt"
                         else boxes_from_yolo(fold, path, conf))
        if len(boxes) == 0:
            rows += match(np.zeros((0, h, w), bool), np.zeros(0), gt, im["file_name"])[0]
            continue

        res = sam.predict(str(path), bboxes=boxes.tolist(), verbose=False)[0]
        m = res.masks.data.cpu().numpy() > 0.5
        pred = np.stack([cv2.resize(x.astype(np.uint8), (w, h), cv2.INTER_NEAREST).astype(bool)
                         for x in m]) if m.shape[1:] != (h, w) else m

        pred = np.array([clean_mask(m) for m in pred], bool).reshape(-1, h, w)
        r, matched = match(pred, scores[: len(pred)], gt, im["file_name"])
        rows += r

        if figures:
            vis = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
            for x in gt:
                cv2.drawContours(vis, cv2.findContours(x.astype(np.uint8), cv2.RETR_EXTERNAL,
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
    ap.add_argument("--prompt", choices=["gt", "yolo"], required=True)
    ap.add_argument("--model", default="sam2.1_b.pt")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 張，0 表示全部")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    rows = run(args.fold, args.prompt, args.model, args.conf, args.limit, not args.no_figures)
    if not args.limit:
        EVAL.mkdir(parents=True, exist_ok=True)
        with (EVAL / f"sam2_{args.prompt}_fold{args.fold}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    summarize(rows, f"SAM2 ({args.prompt} prompt) fold {args.fold}")


if __name__ == "__main__":
    main()
