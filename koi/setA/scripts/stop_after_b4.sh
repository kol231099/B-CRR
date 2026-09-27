#!/bin/sh
# EfficientNet-B4 五折跑完就停掉 encoder 佇列，其餘 encoder 改到 Kaggle 跑。
# 停掉 run_enc8 之後 run_d2 會自動接手（PointRend／Transfiner 留在本機，
# 因為 detectron2 的環境已經裝好且能跑，搬到 Kaggle 要重來一次相依考古）。
cd "$(dirname "$0")"
while [ "$(grep -c 'unet_efficientnet-b4 最佳 val Dice' ../logs/enc8.log)" -lt 5 ]; do
  sleep 60
done
echo "efficientnet-b4 五折完成 $(date '+%H:%M')，停止 encoder 佇列"
pkill -f run_enc8.sh
sleep 2
pkill -f "train_seg2.py --arch unet --encoder efficientnet-b3"
echo "===== 產生 efficientnet-b4 的端到端指標 ====="
python3 -W ignore -u ./eval_e2e.py --model unet_efficientnet-b4
echo "ENC8_STOPPED"
