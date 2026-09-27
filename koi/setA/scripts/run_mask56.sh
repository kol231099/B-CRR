#!/bin/sh
# 遮罩頭 56x56 的 5-fold 訓練。與基準（28x28）唯一的差別就是這個參數，
# 資料、切分、超參數全部相同，所以差異可以歸因到遮罩解析度。
set -e
cd "$(dirname "$0")"   # 腳本與資料同層，搬動整個資料夾也不會失效
for k in 0 1 2 3 4; do
  echo "===== mask56 fold $k ====="
  python3 -u ./train_maskrcnn.py --fold "$k" --mask-res 56 \
      --tag mask56 --epochs "${EPOCHS:-40}"
done
echo "MASK56_DONE"
