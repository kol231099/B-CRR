#!/bin/sh
# 接續 BPR：比較表已在前一輪完成，從步驟 1 開始。
cd "$(dirname "$0")"
echo "===== BPR 步驟 1／4：dump 訓練用粗糙遮罩 ====="
python3 -W ignore -u ./bpr_dump.py --model unetpp_resnet34 --split train --tta
echo "===== BPR 步驟 2／4：訓練五個精修網路 ====="
for k in 0 1 2 3 4; do
  [ -f ../checkpoints/bpr/fold$k.pt ] && { echo "skip fold $k"; continue; }
  python3 -W ignore -u ./train_bpr.py --fold "$k" --epochs 10 || echo "!!!!! BPR fold $k 失敗 !!!!!"
done
echo "===== BPR 步驟 3／4：dump 各模型的驗證集預測 ====="
python3 -W ignore -u ./bpr_dump.py --model maskrcnn --split val --tta
for d in ../checkpoints/seg2/*/; do
  m=$(basename "$d")
  [ -d "../bpr/${m}_tta/val" ] || python3 -W ignore -u ./bpr_dump.py --model "$m" --split val --tta
done
echo "===== BPR 步驟 4／4：套用精修並重算指標 ====="
python3 -W ignore -u ./eval_bpr.py --model maskrcnn_tta
for d in ../checkpoints/seg2/*/; do
  python3 -W ignore -u ./eval_bpr.py --model "$(basename "$d")_tta"
done
echo "BPR_ALL_DONE"
