#!/bin/sh
# 依序跑 5 個 fold 的 YOLOv11-seg，fold 切分與 Mask R-CNN 完全相同。
set -e
cd "$(dirname "$0")"   # 腳本與資料同層，搬動整個資料夾也不會失效
for k in 0 1 2 3 4; do
  echo "===== yolo fold $k ====="
  python3 -u ./train_yolo.py --fold "$k" --epochs "${EPOCHS:-150}"
done
