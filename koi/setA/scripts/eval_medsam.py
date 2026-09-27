"""用 MedSAM 做第二階段分割，以 box 當 prompt，零訓練。

MedSAM 是 SAM (ViT-B) 在大量醫學影像上微調的版本。它與 SAM 2 的對照正是這個
比較的重點之一：**在醫學影像上微調過的通用模型，是否真的比一般的通用模型好**。
兩者都不需要訓練，所以差異純粹來自預訓練資料。

與 eval_sam2.py 完全對稱：同樣的兩種 prompt、同樣的 metrics.py、同樣的原圖座標。

    --prompt gt      GT bbox 當 prompt（oracle），純測分割能力
    --prompt yolo    YOLOv11-seg 預測的 bbox，端到端表現

實作用 transformers 的 SamModel 而非 MedSAM 官方 repo：官方 repo 需要另外安裝
segment_anything 並從 Google Drive 抓權重，而 wanglab/medsam-vit-base 是官方在
HuggingFace 上的轉檔，用已經裝好的 transformers 就能跑，不必動環境。

用法：
    py koi/scripts/eval_medsam.py --fold 0 --prompt gt
    py koi/scripts/eval_medsam.py --fold 0 --prompt gt --limit 2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_sam2 import boxes_from_gt, boxes_from_yolo  # noqa: E402
from metrics import FIELDS, match, summarize  # noqa: E402
from postprocess import clean_mask  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMAGES, ANN, EVAL = ROOT / "images", ROOT / "annotations", ROOT / "eval"
MODEL_ID = "wanglab/medsam-vit-base"


@torch.no_grad()
def run(fold: int, prompt: str, conf: float, limit: int, figures: bool) -> list[dict]:
    from transformers import SamModel, SamProcessor

    model = SamModel.from_pretrained(MODEL_ID).eval()
    processor = SamProcessor.from_pretrained(MODEL_ID)

    coco = json.loads((ANN / f"fold{fold}_val.json").read_text(encoding="utf-8"))
    by: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by.setdefault(a["image_id"], []).append(a)

    out_dir = EVAL / f"medsam_{prompt}_fold{fold}"
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

        # 根尖片是灰階，SAM 的影像編碼器吃三通道，複製灰階即可
        rgb = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
        inputs = processor(rgb, input_boxes=[boxes.tolist()], return_tensors="pt")
        out = model(**inputs, multimask_output=False)
        masks = processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0]
        pred = masks.squeeze(1).numpy().astype(bool)

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
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    rows = run(args.fold, args.prompt, args.conf, args.limit, not args.no_figures)
    if not args.limit:
        EVAL.mkdir(parents=True, exist_ok=True)
        with (EVAL / f"medsam_{args.prompt}_fold{args.fold}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    summarize(rows, f"MedSAM ({args.prompt} prompt) fold {args.fold}")


if __name__ == "__main__":
    main()
