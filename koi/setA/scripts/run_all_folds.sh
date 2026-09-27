#!/bin/sh
# 依序跑 5 個 fold。25 張的單一 fold 只有 5 張 val，數字太吵，
# 必須跑滿 5 個 fold 讓 57 顆牙每顆都被測到一次，才能拿來比較三條 pipeline。
set -e
cd "$(dirname "$0")"   # 腳本與資料同層，搬動整個資料夾也不會失效
for k in 0 1 2 3 4; do
  echo "===== fold $k ====="
  python3 -u ./train_maskrcnn.py --fold "$k" --epochs "${EPOCHS:-40}"
done
