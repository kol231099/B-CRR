#!/bin/sh
# DeepLabv3+ decoder 接 HRNet-w32，補上 smp 拒絕建構的那一格。
cd "$(dirname "$0")"
CK=../checkpoints/seg2
for k in 0 1 2 3 4; do
  [ -f "$CK/deeplabv3phr_tu-hrnet_w32/fold$k.pt" ] && { echo "===== skip fold $k ====="; continue; }
  echo "===== deeplabv3phr × tu-hrnet_w32  fold $k ====="
  python3 -W ignore -u ./train_seg2.py --arch deeplabv3phr --encoder tu-hrnet_w32 \
    --fold "$k" --epochs 20 || echo "!!!!! fold $k 失敗 !!!!!"
done
echo "===== 併入比較表 ====="
python3 -W ignore -u ./eval_seg2.py --all --tta
python3 -W ignore -u ./eval_seg2_holdout.py --all --tta
echo "DLHR_DONE"
