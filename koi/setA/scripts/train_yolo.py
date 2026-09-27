"""訓練 YOLOv11-seg，一次一個 fold。

這個模型在整套比較裡身兼三職：

    實驗組      第 2、3 條 pipeline 的第一階段
    ROI 來源    它預測的 bbox 就是要餵給 nnU-Net / MedSAM 的裁切框
    標註工具    訓好之後拿去對剩下 275 張推論，人工修正比從頭畫快 5~10 倍

為什麼用 CPU
------------
與 Mask R-CNN 同一個理由：實測 MPS 上 torchvision 的偵測模型會卡死。ultralytics
自己支援 mps，但為了讓兩個模型的比較條件一致（同硬體、同精度），這裡也用 CPU。
要試 MPS 的話加 --device mps。

imgsz 的選擇
------------
預設 1024 而非 YOLO 慣用的 640。牙齒又高又窄，根尖那段的寬度只有幾十個像素，
640 會把它壓到十幾個像素，HD95 直接失真。原圖短邊是 825 與 1324，1024 是不放大
又不過度縮小的折衷。

用法：
    py koi/scripts/train_yolo.py --fold 0
    py koi/scripts/train_yolo.py --fold 0 --model yolo11s-seg.pt --epochs 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from weights import weight  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--model", default="yolo11s-seg.pt", help="n/s/m/l/x，預設 s")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(weight(args.model))
    model.train(
        data=str(ROOT / "yolo" / f"fold{args.fold}" / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "yolo_runs"),
        name=f"fold{args.fold}",
        exist_ok=True,
        seed=0,
        # 只有 20 張訓練圖，增強的權重很高。翻轉在解剖上都合理：資料裡本來
        # 就同時有左右側與上下顎的片子。
        fliplr=0.5,
        flipud=0.5,
        # 關掉會破壞牙齒形狀先驗或製造假的相鄰關係的增強
        mosaic=0.0,      # 把四張圖拼一起會產生不存在的牙齒排列
        mixup=0.0,       # 兩張根尖片疊加沒有物理意義
        # 旋轉關掉：實測會讓訓練崩潰（mAP50 中途掉到 0.001），ultralytics 的
        # 仿射增強在 augment.py 拋出 divide-by-zero，應是牙齒這種細長多邊形
        # 旋轉後偶爾退化成零面積。少一個增強換訓練穩定，划算。
        degrees=0.0,
        scale=0.3,
        hsv_h=0.0, hsv_s=0.0,   # 灰階影像，色相飽和度無意義
        hsv_v=0.4,              # 亮度抖動對應曝光差異
        # early stopping 關掉。val 只有 5 張圖，mAP 在 fold 間震盪極大——fold0 的
        # best.pt 曾被選在第 7 個 epoch，然後在 ep 57 被 patience 砍掉，等於拿一個
        # 沒訓練的模型去跟訓練完整的 Mask R-CNN 比。所有 fold 必須練滿相同 epoch。
        patience=args.epochs,
    )


if __name__ == "__main__":
    main()
