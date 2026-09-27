#!/bin/sh
# 補上三格，解開 DeepLabv3+ 與 resnet101 的混淆。
# 順序按價值排：前兩格補齊「十字」的兩臂，第三格測交互作用。
cd "$(dirname "$0")"
while pgrep -f "run_hrnet.sh|run_bpr2.sh|train_seg2.py|train_bpr.py|bpr_dump.py" > /dev/null; do sleep 120; done
echo "前置作業結束 $(date '+%H:%M')"
CK=../checkpoints/seg2
for combo in "unet resnet101" "deeplabv3p resnet34" "deeplabv3p resnet50"; do
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
echo "GRID3_DONE"
