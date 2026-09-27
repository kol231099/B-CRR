"""訓練 YOLO11-OBB，超參與 train_yolo.py 逐項相同，只差偵測頭。

權重初始化刻意**不用**官方的 yolo11s-obb.pt：那是 DOTA 空拍資料集預訓練的，
而對照組 YOLO11-seg 用的是 COCO。兩者預訓練領域不同，OBB 若輸了會分不清是
「OBB 這個設計不好」還是「DOTA 與牙科 X 光差太遠」。

改成從本機既有的 yolo11s-seg.pt 轉移：YOLO11s 的 backbone 與 neck 兩者共用，
實測轉移 535/541 個張量（99%），只有 OBB 頭是隨機初始化。這樣 seg 與 OBB
兩條的起跑點完全相同，差異只剩頭部——那正是要比的東西。副作用是不必對外連線。

用法：
    py scripts/train_yolo_obb.py --fold 0 --device mps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weights import weight  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--arch", default="yolo11s-obb.yaml",
                    help="模型結構檔，套件內建（yolov8s-obb / yolo11s-obb / "
                         "yolo12s-obb / yolo26s-obb）")
    ap.add_argument("--init", default="yolo11s-seg.pt",
                    help="轉移來源。各架構用自己的 COCO 預訓練權重，"
                         "以免比較到的是預訓練領域而非架構")
    ap.add_argument("--tag", default="", help="輸出子目錄，預設由 --arch 推得")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.arch).load(weight(args.init))
    model.train(
        data=str(ROOT / "yolo_obb" / f"fold{args.fold}" / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "yolo_obb_runs" /
                     (args.tag or args.arch.replace("-obb.yaml", ""))),
        name=f"fold{args.fold}",
        exist_ok=True,
        seed=0,
        # 以下每一項都與 train_yolo.py 相同，理由見該檔註解
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.0,
        mixup=0.0,
        # degrees=0 對 OBB 尤其重要：旋轉增強會改變角度標籤，
        # 且 ultralytics 的仿射增強在細長多邊形上實測會讓訓練崩潰
        degrees=0.0,
        scale=0.3,
        hsv_h=0.0, hsv_s=0.0,
        hsv_v=0.4,
        patience=args.epochs,
    )


if __name__ == "__main__":
    main()
