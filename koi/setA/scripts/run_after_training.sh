#!/bin/sh
# 等第二階段訓練跑完，自動接上含 TTA 的完整評估與比較。
cd "$(dirname "$0")"
while pgrep -f "run_seg2.sh|run_seg2_resume.sh|train_seg2.py" > /dev/null; do sleep 120; done
echo "訓練結束 $(date '+%H:%M')，開始評估"

echo "===== Mask R-CNN + TTA（新指標）====="
for k in 0 1 2 3 4; do
  python3 -W ignore -u ./eval_maskrcnn.py --fold "$k" --tta --no-figures
done
echo "===== 保留測試集 + TTA ====="
python3 -W ignore -u ./eval_holdout.py --tta

echo "===== 第二階段全部 + TTA ====="
python3 -W ignore -u ./eval_seg2.py --all --tta
echo "===== 第二階段全部（無 TTA，作為對照）====="
python3 -W ignore -u ./eval_seg2.py --all

echo "===== 比較表 ====="
python3 -W ignore -u ./compare_all.py --tta
python3 -W ignore -u ./compare_all.py
echo "ALL_DONE"
