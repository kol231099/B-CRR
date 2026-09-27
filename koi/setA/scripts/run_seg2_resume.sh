#!/bin/sh
# 接續中斷的訓練：deeplabv3p 的 fold4，以及尚未開始的兩個模型。
# 已完成的 fold 會跳過。set +e 讓單一 fold 失敗不會中斷整條鏈。
cd "$(dirname "$0")"
CK=../checkpoints/seg2
run() {
  for k in 0 1 2 3 4; do
    [ -f "$CK/$1_$2/fold$k.pt" ] && { echo "===== skip $1 × $2 fold $k（已完成）====="; continue; }
    echo "===== $1 × $2  fold $k ====="
    python3 -W ignore -u ./train_seg2.py --arch "$1" --encoder "$2" --fold "$k" --epochs 20 || \
      echo "!!!!! $1 × $2 fold $k 失敗，繼續下一個 !!!!!"
  done
}
run deeplabv3p resnet101
run unetpp     resnet34
run unet       mit_b0
echo "SEG2_DONE"
