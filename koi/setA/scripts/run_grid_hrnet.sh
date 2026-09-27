#!/bin/sh
# 補上 HRNet w32 的解碼器那一臂。
# w32 目前只跟 U-Net 配過，因此「最好編碼器 × 最好解碼器」這一格從未測試。
# unetpp 先跑：固定 resnet34 時它是 B-IoU 最好的解碼器。
cd "$(dirname "$0")"
while pgrep -f "run_hrnet.sh|run_bpr2.sh|run_grid3.sh|train_seg2.py|train_bpr.py|bpr_dump.py|eval_bpr.py" > /dev/null; do sleep 120; done
echo "前置作業結束 $(date '+%H:%M')"
CK=../checkpoints/seg2
for combo in "unetpp tu-hrnet_w32" "deeplabv3p tu-hrnet_w32"; do
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
echo "GRID_HRNET_DONE"
