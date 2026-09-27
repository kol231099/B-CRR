"""把 COCO 多邊形標註轉成 YOLO11-OBB 的資料格式，每個 fold 一份。

OBB 的標註是每行一個實例的四個角點（順序沿著矩形，非軸對齊）：

    <class> x1 y1 x2 y2 x3 y3 x4 y4      座標除以影像寬高正規化到 0~1

角點由多邊形的 cv2.minAreaRect 算出——**不需要重新標註**，斜框是從既有的
逐顆牙多邊形推導的，因此與 Mask R-CNN 吃的是同一份人工標註，兩者可公平比較。

其餘規則（符號連結、類別從 0 起、iscrowd 跳過）與 make_yolo.py 完全一致。

用法：
    py scripts/make_yolo_obb.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGES, ANN = ROOT / "images", ROOT / "annotations"
YOLO = ROOT / "yolo_obb"


def write_split(fold_dir: Path, split: str, coco: dict) -> int:
    img_dir, lbl_dir = fold_dir / "images" / split, fold_dir / "labels" / split
    for d in (img_dir, lbl_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            old.unlink()

    by: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by.setdefault(a["image_id"], []).append(a)

    n_skipped = 0
    for im in coco["images"]:
        (img_dir / im["file_name"]).symlink_to((IMAGES / im["file_name"]).resolve())
        lines = []
        for a in by.get(im["id"], []):
            if a["iscrowd"]:
                n_skipped += 1
                continue
            pts = np.array(a["segmentation"][0], np.float32).reshape(-1, 2)
            box = cv2.boxPoints(cv2.minAreaRect(pts))          # 4x2，順時針
            box[:, 0] = np.clip(box[:, 0] / im["width"], 0, 1)
            box[:, 1] = np.clip(box[:, 1] / im["height"], 0, 1)
            lines.append("0 " + " ".join(f"{v:.6f}" for v in box.reshape(-1)))
        (lbl_dir / f"{Path(im['file_name']).stem}.txt").write_text("\n".join(lines) + "\n", "utf-8")
    return n_skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    total_skipped = 0
    for k in range(args.folds):
        fold_dir = YOLO / f"fold{k}"
        for split, name in (("train", f"fold{k}_train.json"), ("val", f"fold{k}_val.json")):
            total_skipped += write_split(
                fold_dir, split, json.loads((ANN / name).read_text(encoding="utf-8")))
        (fold_dir / "data.yaml").write_text(
            f"path: {fold_dir.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: tooth\n",
            "utf-8")
        n_tr = len(list((fold_dir / "images" / "train").iterdir()))
        n_va = len(list((fold_dir / "images" / "val").iterdir()))
        n_inst = sum(len(p.read_text().split("\n")) - 1
                     for p in (fold_dir / "labels").rglob("*.txt"))
        print(f"fold{k}: train {n_tr} 張 / val {n_va} 張，共 {n_inst} 個實例")
    print(f"ignore 區跳過 {total_skipped} 個")
    print(f"輸出 → {YOLO}/")


if __name__ == "__main__":
    main()
