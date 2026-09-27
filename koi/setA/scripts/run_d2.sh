#!/bin/sh
# PointRend 與 Mask Transfiner 的五折訓練，等八個 encoder 跑完再開始。
# 兩者在各自獨立的 venv 裡跑（transfiner 自帶一份改過的 detectron2，會與官方版衝突）。
cd "$(dirname "$0")"
SCRATCH=/private/tmp/claude-501/-Users-kol-Downloads-CRR-PA-main/835c93ed-5aab-405d-9196-0e96fc2c612d
D2="$SCRATCH/d2env/bin/python"
TF="$SCRATCH/tfenv/bin/python"
D2DIR="$(pwd)/d2"
PR_ITERS=1500       # PointRend：訓練集 50 張、batch 2 → 約 60 epoch
TF_ITERS=1000       # Transfiner 每 iter 慢 2.8 倍，降到約 40 epoch 控制總時間

while pgrep -f "run_enc8.sh|train_seg2.py" > /dev/null; do sleep 120; done
echo "八個 encoder 結束 $(date '+%H:%M')"

echo "===== PointRend 五折 ====="
for k in 0 1 2 3 4; do
  [ -f ../d2out/pointrend_fold$k/model_final.pth ] && { echo "skip fold $k"; continue; }
  echo "----- pointrend fold $k -----"
  (cd d2 && "$D2" ./train_d2.py --arch pointrend --fold "$k" --iters $PR_ITERS) \
    || echo "!!!!! pointrend fold $k 失敗 !!!!!"
done
(cd d2 && "$D2" ./predict_d2.py --arch pointrend --split holdout)
python3 -W ignore ./eval_d2.py --arch pointrend --split holdout

echo "===== Mask Transfiner 五折 ====="
for k in 0 1 2 3 4; do
  [ -f ../d2out/transfiner_fold$k/model_final.pth ] && { echo "skip fold $k"; continue; }
  echo "----- transfiner fold $k -----"
  (cd "$SCRATCH/transfiner" && "$TF" "$D2DIR/train_d2.py" \
     --arch transfiner --fold "$k" --iters $TF_ITERS) \
    || echo "!!!!! transfiner fold $k 失敗 !!!!!"
done
echo "D2_ALL_DONE"
