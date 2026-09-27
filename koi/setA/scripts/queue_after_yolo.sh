#!/bin/sh
# 等 YOLO 的 5-fold 跑完再啟動增強實驗，避免兩者搶 CPU。
cd "$(dirname "$0")"   # 腳本與資料同層，搬動整個資料夾也不會失效
while pgrep -f "run_all_folds_yolo|train_yolo.py" > /dev/null; do sleep 120; done
echo "YOLO 已結束，開始增強實驗 $(date '+%H:%M')"
sh koi/scripts/run_enhance_experiment.sh
