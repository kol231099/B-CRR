#!/bin/sh
# 學習曲線：用 5 / 10 / 15 張訓練影像各訓一次，n=20 沿用既有的完整模型。
#
# 回答的問題是「瓶頸是資料量還是模型」。曲線在 n=20 仍陡升 → 先去標 300 張，
# 改架構是浪費；曲線已平坦 → 資料夠了，才輪到 PointRend、56x56 遮罩頭那些。
#
# 跑 3 個 fold 而非 1 個：單一 fold 的 val 只有 5 張圖，一個點的雜訊會大到
# 看不出趨勢。3 個 fold 可以給每個點一個範圍。
set -e
cd "$(dirname "$0")"   # 腳本與資料同層，搬動整個資料夾也不會失效
for n in 5 10 15; do
  for k in 0 1 2; do
    if [ -f "koi/checkpoints/curve_n${n}/maskrcnn_fold${k}.pt" ]; then
      echo "===== skip n=$n fold=$k（已完成）====="; continue
    fi
    echo "===== n=$n fold=$k ====="
    python3 -u ./train_maskrcnn.py --fold "$k" --subset "$n" \
        --tag "curve_n${n}" --epochs "${EPOCHS:-40}"
  done
done
echo "LEARNING_CURVE_DONE"
