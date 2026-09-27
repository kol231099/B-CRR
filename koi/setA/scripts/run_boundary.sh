#!/bin/sh
# 邊界感知損失的 5-fold 訓練。與基準唯一的差別是遮罩損失，
# 資料、切分、超參數、遮罩解析度全部相同。
set -e
cd "$(dirname "$0")"   # 腳本與資料同層，搬動整個資料夾也不會失效
for k in 0 1 2 3 4; do
  echo "===== boundary fold $k ====="
  python3 -u ./train_maskrcnn.py --fold "$k" --mask-loss boundary \
      --tag boundary --epochs "${EPOCHS:-40}"
done
echo "BOUNDARY_DONE"
