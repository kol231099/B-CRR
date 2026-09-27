#!/bin/sh
# 第二階段的七個候選模型，各跑 5-fold。
#
# 20 epochs：實測 U-Net 在第 10 個 epoch 即達 val Dice 0.9687、第 15 個 0.9703，
# 之後幾乎不再上升。所有模型用相同輪數，比較才受控。
#
# 模型依牙齒分割文獻挑選：
#   unet × resnet34         所有相關論文的共同基準
#   unet × resnet50         與第一階段 Mask R-CNN 相同骨幹，可做跨階段對照
#   deeplabv3p × resnet101  Leite 等人於全景片報告 IoU 0.936、F1 0.966
#   unetpp × resnet34       牙齒分割實測 IoU 0.8619、Dice 0.9258
#   unet × efficientnet-b0  骨幹比較研究中精度／計算成本的最佳點
#   unet × mit_b0           SegFormer 編碼器，代表 Transformer 路線
#   fpn × resnet34          另一個解碼器家族，計算最省
set -e
cd "$(dirname "$0")"
run() {
  for k in 0 1 2 3 4; do
    echo "===== $1 × $2  fold $k ====="
    python3 -W ignore -u ./train_seg2.py --arch "$1" --encoder "$2" --fold "$k" --epochs "${EPOCHS:-20}"
  done
}
run unet       resnet34
run fpn        resnet34
run unet       resnet50
run unet       efficientnet-b0
run deeplabv3p resnet101
run unetpp     resnet34
run unet       mit_b0
echo "SEG2_DONE"
