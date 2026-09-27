#!/bin/sh
# 八個新 encoder，decoder 固定 U-Net。順序按投報率：
#   1-2 便宜且有明確假設（w48 延伸 w18→w32 趨勢；densenet 補齊 Tufts 基準家族）
#   3-4 檢查 efficientnet-b0 慘敗是變體問題還是家族問題
#   5-6 ConvNeXt-V2，文獻支持，且可回頭驗證 worktogether 不用預訓練是否吃虧
#   7-8 MobileNet，輕量對照
cd "$(dirname "$0")"
CK=../checkpoints/seg2
for enc in densenet121 tu-hrnet_w48 efficientnet-b4 efficientnet-b3 \
           tu-convnextv2_nano tu-convnextv2_tiny tu-mobilenetv3_large_100 mobilenet_v2; do
  for k in 0 1 2 3 4; do
    [ -f "$CK/unet_$enc/fold$k.pt" ] && { echo "===== skip unet × $enc fold $k ====="; continue; }
    echo "===== unet × $enc  fold $k ====="
    python3 -W ignore -u ./train_seg2.py --arch unet --encoder "$enc" --fold "$k" --epochs 20 || \
      echo "!!!!! unet × $enc fold $k 失敗 !!!!!"
  done
  echo "----- $enc 五折完成 $(date '+%H:%M') -----"
done
echo "===== 端到端評估 ====="
python3 -W ignore -u ./eval_e2e.py --all
echo "ENC8_DONE"
