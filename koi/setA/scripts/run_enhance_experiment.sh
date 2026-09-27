#!/bin/sh
# 對每一種影像增強訓練 Mask R-CNN 的 5 個 fold。
#
# original 那組已經訓練過，直接跳過——重跑會得到相同結果，浪費 2.1 小時。
# 每種方法約 2.1 小時，10 種約 21 小時。
#
# SAM 2 不在這裡：它零訓練，11 種增強共用同一份權重，比較由
# eval_enhance_sam2.py 直接推論完成，不產生任何 .pt。
set -e
cd "$(dirname "$0")"   # 腳本與資料同層，搬動整個資料夾也不會失效
METHODS="contrast_stretch hist_eq clahe clahe_strong gamma_0.6 unsharp clahe+unsharp bilateral+clahe homomorphic sobel"
for e in $METHODS; do
  if [ -f "koi/checkpoints/$e/maskrcnn_fold4.pt" ]; then
    echo "===== skip $e（已完成）====="; continue
  fi
  for k in 0 1 2 3 4; do
    echo "===== enhance=$e fold=$k ====="
    python3 -u ./train_maskrcnn.py --fold "$k" --enhance "$e" --epochs "${EPOCHS:-40}"
  done
done
echo "ALL_ENHANCE_DONE"
