#!/usr/bin/env bash
# 一次跑完 Table 1 的五個模型（五折）＋ holdout 評估＋論文表格（含 p 值、Friedman）。
#
# 放在 koi/setA/final/scripts/，在 koi/setA/final 底下執行：
#     bash scripts/run_all_final.sh
#
# 說明文件：doc/retrain_all.md（B-CRR repo）
#
# 五個模型與設定（與 Table 1 相同；HRNet 為最佳版：框擾動訓練 + 機率融合）：
#   Mask R-CNN                     train_maskrcnn.py  40 epochs, batch 2, lr 1e-4
#   Mask R-CNN + U-Net-HRNet-w32   final_jit_train.py 40 epochs, batch 4, lr 1e-4, 輸入 512×256,
#                                  框擾動 旋轉 ±5°、平移 ±5%、縮放 ±8%；推論時與 Mask R-CNN 機率 0.5:0.5 融合
#   YOLOv8s-seg / YOLO11s-seg / YOLO26s-seg   train_yolo.py 300 epochs, batch 4, imgsz 1024
#
# 可中斷續跑：每一折完成會在 logs/run_all/ 留下 .done，重跑時自動跳過已完成的部分。
# 想從頭重跑某一步，刪掉對應的 .done 即可。

set -euo pipefail

# ─────────────────────────── 設定（只需要看這一段） ───────────────────────────
DEVICE="${DEVICE:-cuda}"            # HRNet：cuda / mps / cpu
MR_DEVICE="${MR_DEVICE:-$DEVICE}"   # Mask R-CNN：預設同 DEVICE；Mac 上請設 cpu（torchvision 偵測模型在 MPS 會卡住）
YOLO_DEVICE="${YOLO_DEVICE:-0}"     # YOLO：GPU 編號，或 mps / cpu
FOLDS="0 1 2 3 4"
MR_EPOCHS=40
HR_EPOCHS=40
YOLO_EPOCHS=300
YOLO_MODELS="yolov8s-seg yolo11s-seg yolo26s-seg"
# holdout 結果檔名（eval/hold5_<名稱>_fold{k}.csv），順序對應 YOLO_MODELS
YOLO_NAMES="yolov8sseg yolo11sseg yolo26sseg"
# ⚠ 必填：產生上面三個 YOLO holdout 結果檔的指令——請用原本 Table 1 產生 YOLO 結果的同一支程式，
#   例如 YOLO_EVAL='python3 scripts/<原本的 YOLO holdout 評估腳本>.py <參數>'
YOLO_EVAL="${YOLO_EVAL:-}"
# ──────────────────────────────────────────────────────────────────────────────

ROOT="$(pwd)"
LOG="$ROOT/logs/run_all"
TS="$(date +%Y%m%d_%H%M%S)"

die() { echo "✗ $*" >&2; exit 1; }
step() { echo; echo "════ $* ════ $(date '+%F %T')"; }
done_mark() { touch "$LOG/$1.done"; }
is_done() { [[ -f "$LOG/$1.done" ]]; }

# ─────────────────────────── 0. 開跑前檢查（任何一項不過就不開始訓練） ───────────────────────────
step "0. 開跑前檢查"
[[ "$(basename "$ROOT")" == "final" ]] || die "請在 koi/setA/final 底下執行（目前在 $ROOT）"
for f in train_maskrcnn.py train_yolo.py final_jit_train.py final_fuse_eval.py final_table.py \
         make_crops_obb.py make_yolo.py; do
  [[ -f "scripts/$f" ]] || die "找不到 scripts/$f"
done
for f in instances_all.json holdout.json fold0_train.json fold0_val.json; do
  [[ -f "annotations/$f" ]] || die "找不到 annotations/$f"
done
[[ -f crops_obb/manifest.csv ]] || die "找不到 crops_obb/manifest.csv（先跑 python3 scripts/make_crops_obb.py）"
[[ crops_obb/manifest.csv -nt annotations/instances_all.json ]] \
  || die "crops_obb 比標註舊，請先重跑 python3 scripts/make_crops_obb.py"
for k in $FOLDS; do
  [[ -d "yolo/fold$k" ]] || die "找不到 yolo/fold$k（先跑 python3 scripts/make_yolo.py）"
  [[ "yolo/fold$k/data.yaml" -nt annotations/instances_all.json ]] \
    || die "yolo/fold$k 比標註舊，請先重跑 python3 scripts/make_yolo.py"
done
[[ -n "$YOLO_EVAL" && "$YOLO_EVAL" != *"<"* ]] || die "YOLO_EVAL 未設定：請填入原本產生 YOLO holdout 結果檔的指令（見腳本最上方）"
read -r -a YM <<< "$YOLO_MODELS"; read -r -a YN <<< "$YOLO_NAMES"
[[ ${#YM[@]} -eq ${#YN[@]} ]] || die "YOLO_MODELS 與 YOLO_NAMES 數量不同"
python3 -c "import torch, ultralytics, scipy" || die "缺少 torch / ultralytics / scipy"
echo "  ✓ 檢查通過"

# 第一次執行：把舊的權重與結果改名保存，避免新舊混用
mkdir -p "$LOG"
if [[ ! -f "$LOG/started" ]]; then
  for d in checkpoints checkpoints_obb_jit yolo_runs eval; do
    if [[ -e "$d" ]]; then mv "$d" "${d}_bak_$TS"; echo "  舊的 $d → ${d}_bak_$TS"; fi
  done
  echo "$TS" > "$LOG/started"
fi
mkdir -p eval

# 記錄環境（論文 Methods 的硬體與軟體版本）
{
  set +e   # 記錄環境失敗不影響訓練
  echo "date: $(date)"; echo "host: $(hostname)"
  python3 -c "import sys, torch, torchvision, ultralytics, scipy; print('python', sys.version.split()[0]); print('torch', torch.__version__, '| cuda', torch.version.cuda); print('torchvision', torchvision.__version__); print('ultralytics', ultralytics.__version__); print('scipy', scipy.__version__); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" || true
  python3 -c "import segmentation_models_pytorch as s, timm; print('smp', s.__version__); print('timm', timm.__version__)" 2>/dev/null || true
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
  echo "images: $(python3 -c "import json;print(len(json.load(open('annotations/instances_all.json'))['images']))") train, $(python3 -c "import json;print(len(json.load(open('annotations/holdout.json'))['images']))") holdout"
} > "$LOG/env.txt" 2>&1 || true
echo "  環境 → $LOG/env.txt"

# ─────────────────────────── 1. Mask R-CNN ───────────────────────────
step "1. Mask R-CNN（${MR_EPOCHS} epochs × 5 折）"
for k in $FOLDS; do
  is_done "maskrcnn_fold$k" && { echo "  fold $k 已完成，跳過"; continue; }
  python3 scripts/train_maskrcnn.py --fold "$k" --epochs "$MR_EPOCHS" --device "$MR_DEVICE" \
    2>&1 | tee "$LOG/maskrcnn_fold$k.log"
  done_mark "maskrcnn_fold$k"
done

# ─────────────────────────── 2. U-Net-HRNet-w32（框擾動版，最佳設定） ───────────────────────────
step "2. U-Net-HRNet-w32 框擾動版（${HR_EPOCHS} epochs × 5 折）"
for k in $FOLDS; do
  is_done "hrnet_jit_fold$k" && { echo "  fold $k 已完成，跳過"; continue; }
  python3 scripts/final_jit_train.py --arch unet --encoder tu-hrnet_w32 --fold "$k" \
    --epochs "$HR_EPOCHS" --device "$DEVICE" 2>&1 | tee "$LOG/hrnet_jit_fold$k.log"
  done_mark "hrnet_jit_fold$k"
done

# ─────────────────────────── 3. 三個 YOLO-seg ───────────────────────────
step "3. YOLO-seg ×3（${YOLO_EPOCHS} epochs × 5 折）"
for m in "${YM[@]}"; do
  for k in $FOLDS; do
    is_done "${m}_fold$k" && { echo "  $m fold $k 已完成，跳過"; continue; }
    python3 scripts/train_yolo.py --fold "$k" --model "$m.pt" --epochs "$YOLO_EPOCHS" \
      --imgsz 1024 --batch 4 --device "$YOLO_DEVICE" 2>&1 | tee "$LOG/${m}_fold$k.log"
    done_mark "${m}_fold$k"
  done
done

# ─────────────────────────── 4. holdout 評估 ───────────────────────────
step "4a. Mask R-CNN 與本方法（融合）holdout 評估"
if ! is_done eval_fuse; then
  python3 scripts/final_fuse_eval.py --no-tta 2>&1 | tee "$LOG/eval_fuse.log"
  done_mark eval_fuse
fi

step "4b. YOLO holdout 評估"
if ! is_done eval_yolo; then
  bash -c "$YOLO_EVAL" 2>&1 | tee "$LOG/eval_yolo.log"
  for n in "${YN[@]}"; do
    for k in $FOLDS; do
      [[ -f "eval/hold5_${n}_fold$k.csv" ]] || die "YOLO 評估沒有產生 eval/hold5_${n}_fold$k.csv，請檢查 YOLO_EVAL / YOLO_NAMES"
    done
  done
  done_mark eval_yolo
fi

# ─────────────────────────── 5. 論文表格 ───────────────────────────
step "5. 論文表格（p 值、Holm、Friedman）"
python3 scripts/final_table.py --ref FUS_MaskRCNN FUS_fuse "${YN[@]}" \
  --anchor FUS_fuse --compare FUS_MaskRCNN "${YN[@]}" 2>&1 | tee "eval/paper_table.txt"

echo
echo "全部完成。"
echo "  論文表格　 eval/paper_table.txt（CSV：eval/table_pvalues_FUS_fuse.csv）"
echo "  環境版本　 $LOG/env.txt"
echo "  各步驟 log $LOG/"
