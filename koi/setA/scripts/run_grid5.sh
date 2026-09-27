#!/bin/sh
# 五格一次跑完，不用等待迴圈（前面的工作都已結束）。
#   前三格：解開 DeepLabv3+ 與 resnet101 的混淆
#   後兩格：解開 HRNet w32 與 U-Net 解碼器的混淆
cd "$(dirname "$0")"
CK=../checkpoints/seg2
for combo in "unet resnet101" "deeplabv3p resnet34" "deeplabv3p resnet50" \
             "unetpp tu-hrnet_w32" "deeplabv3p tu-hrnet_w32"; do
  set -- $combo
  for k in 0 1 2 3 4; do
    [ -f "$CK/$1_$2/fold$k.pt" ] && { echo "===== skip $1 × $2 fold $k ====="; continue; }
    echo "===== $1 × $2  fold $k ====="
    python3 -W ignore -u ./train_seg2.py --arch "$1" --encoder "$2" --fold "$k" --epochs 20 || \
      echo "!!!!! $1 × $2 fold $k 失敗 !!!!!"
  done
done
echo "===== 併入比較表 ====="
python3 -W ignore -u ./eval_seg2.py --all --tta
python3 -W ignore -u ./eval_seg2_holdout.py --all --tta
echo "GRID5_DONE"
