"""把 COCO 標註轉成 YOLOv11-seg 的資料格式，每個 fold 一份。

YOLO 的分割標註是每行一個實例：

    <class> x1 y1 x2 y2 ...      座標除以影像寬高正規化到 0~1

與 COCO 的差別除了格式，還有兩點要注意：

    座標正規化    YOLO 用相對座標，所以同一份標註在不同尺寸的圖上不用改。
                  這批資料混了 825x1200 與 1842x1324，正規化正好省事。
    類別從 0 起   COCO 的 category_id 從 1 起（0 慣例上留給背景），YOLO 沒有
                  背景類別，tooth 要編成 0。

影像用符號連結而非複製：5 個 fold 會引用同一批 25 張圖，複製等於存 5 份。

ignore 區
---------
YOLO 沒有 ignore region 的概念。iscrowd=1 的標註不寫進 label，但那樣它會變成
背景；正確做法是把該區域塗黑另存一份影像。目前標註檔還沒有 unclear，此處先
如實記錄有幾個被跳過，補標後再處理。

用法：
    py koi/scripts/make_yolo.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "images"
ANN = ROOT / "annotations"
YOLO = ROOT / "yolo"


def write_split(fold_dir: Path, split: str, coco: dict) -> int:
    img_dir = fold_dir / "images" / split
    lbl_dir = fold_dir / "labels" / split
    for d in (img_dir, lbl_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            old.unlink()

    by: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by.setdefault(a["image_id"], []).append(a)

    n_skipped = 0
    for im in coco["images"]:
        src = IMAGES / im["file_name"]
        link = img_dir / im["file_name"]
        link.symlink_to(src.resolve())

        lines = []
        for a in by.get(im["id"], []):
            if a["iscrowd"]:
                n_skipped += 1
                continue
            pts = a["segmentation"][0]
            xs = [f"{v / im['width']:.6f}" for v in pts[0::2]]
            ys = [f"{v / im['height']:.6f}" for v in pts[1::2]]
            coords = " ".join(f"{x} {y}" for x, y in zip(xs, ys))
            lines.append(f"0 {coords}")
        (lbl_dir / f"{Path(im['file_name']).stem}.txt").write_text("\n".join(lines) + "\n", "utf-8")
    return n_skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    total_skipped = 0
    for k in range(args.folds):
        fold_dir = YOLO / f"fold{k}"
        for split, name in (("train", f"fold{k}_train.json"), ("val", f"fold{k}_val.json")):
            coco = json.loads((ANN / name).read_text(encoding="utf-8"))
            total_skipped += write_split(fold_dir, split, coco)
        (fold_dir / "data.yaml").write_text(
            f"path: {fold_dir.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: tooth\n",
            "utf-8",
        )
        n_tr = len(list((fold_dir / "images" / "train").iterdir()))
        n_va = len(list((fold_dir / "images" / "val").iterdir()))
        n_inst = sum(len(p.read_text().split("\n")) - 1 for p in (fold_dir / "labels").rglob("*.txt"))
        print(f"fold{k}: train {n_tr} 張 / val {n_va} 張，共 {n_inst} 個實例")

    print(f"ignore 區跳過 {total_skipped} 個（YOLO 無 ignore region，補標後需另外塗黑）")
    print(f"輸出 → {YOLO}/")


if __name__ == "__main__":
    main()
