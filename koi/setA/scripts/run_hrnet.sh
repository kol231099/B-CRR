#!/bin/sh
# HRNetV2 當第二階段編碼器。與其餘七個模型同樣的 crops、fold 切分、20 epochs，
# 因此可直接併入現有比較表。w32（36.0M）對應 U-Net×resnet50（32.5M）。
cd "$(dirname "$0")"
CK=../checkpoints/seg2
for enc in tu-hrnet_w32 tu-hrnet_w18; do
  for k in 0 1 2 3 4; do
    [ -f "$CK/unet_$enc/fold$k.pt" ] && { echo "===== skip unet × $enc fold $k ====="; continue; }
    echo "===== unet × $enc  fold $k ====="
    python3 -W ignore -u ./train_seg2.py --arch unet --encoder "$enc" --fold "$k" --epochs 20 || \
      echo "!!!!! unet × $enc fold $k 失敗 !!!!!"
  done
done
echo "HRNET_DONE"
