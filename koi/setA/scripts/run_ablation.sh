#!/bin/sh
# 預訓練與骨幹的三階梯消融（回答「為什麼要加 ResNet」）：
#   原始 U-Net → U-Net×r34 隨機初始化 → U-Net×r34 ImageNet → U-Net×HRNet-w32
# 前兩階是這裡要跑的，後兩階已完成。每一階只動一個因素。
cd "$(dirname "$0")"
while pgrep -f "run_grid5.sh" > /dev/null; do sleep 120; done
echo "五格結束 $(date '+%H:%M')"
CK=../checkpoints/seg2
for combo in "unetvanilla none" "unetscratch resnet34"; do
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
echo "ABLATION_DONE"
