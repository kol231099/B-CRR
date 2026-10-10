#!/usr/bin/env bash
# RunPod 用：一次跑完 Table 1 的五個模型（各五折）＋ holdout 評估＋論文表格，最後打包所有權重與紀錄。
#
# 放在 koi/setA/final/scripts/，在 koi/setA/final 底下執行（建議在 tmux 裡）：
#     YOLO_EVAL='<原本產生 YOLO holdout 結果的指令>' bash scripts/run_all_final.sh 2>&1 | tee run_all.out
#
# 完整說明（給執行者看）：B-CRR repo 的 doc/retrain_all.md
#
# 五個模型（與 Table 1 相同的資料切分與超參數；HRNet 為最終版：框擾動訓練 + 機率融合）：
#   Mask R-CNN                     train_maskrcnn.py  40 epochs, batch 2, lr 1e-4, AdamW, mask head 28×28
#   Mask R-CNN + U-Net-HRNet-w32   第一階段 = 上面的 Mask R-CNN（不另訓）
#                                  第二階段 final_jit_train.py 40 epochs, batch 4, lr 1e-4, 輸入 512×256,
#                                  訓練裁切框加擾動：旋轉 ±5°、平移 ±5%、縮放 ±8%
#                                  推論：mask = clean_mask(0.5·P_HRNet + 0.5·P_MaskRCNN > 0.5)，無 TTA
#   YOLOv8s-seg / YOLO11s-seg / YOLO26s-seg   train_yolo.py 300 epochs, batch 4, imgsz 1024
#
# 可中斷續跑：每完成一步會在 logs/run_all/ 留下 .done，重跑同一行指令會跳過已完成的部分。

set -euo pipefail

# ─────────────────────────── 設定（只需要看這一段） ───────────────────────────
DEVICE="${DEVICE:-cuda}"            # HRNet：cuda / mps / cpu
MR_DEVICE="${MR_DEVICE:-$DEVICE}"   # Mask R-CNN：預設同 DEVICE（Mac 上須設 cpu，MPS 會卡住）
YOLO_DEVICE="${YOLO_DEVICE:-0}"     # YOLO：GPU 編號，或 mps / cpu
FOLDS="0 1 2 3 4"
MR_EPOCHS=40
HR_EPOCHS=40
YOLO_EPOCHS=300
YOLO_MODELS="yolov8s-seg yolo11s-seg yolo26s-seg"
# holdout 結果檔名（eval/hold5_<名稱>_fold{k}.csv），順序對應 YOLO_MODELS
YOLO_NAMES="yolov8sseg yolo11sseg yolo26sseg"
# ⚠ 必填：產生上面三個 YOLO holdout 結果檔的指令——必須是原本 Table 1 產生 YOLO 結果的同一支程式
YOLO_EVAL="${YOLO_EVAL:-}"
# ──────────────────────────────────────────────────────────────────────────────

ROOT="$(pwd)"
LOG="$ROOT/logs/run_all"
TS="$(date +%Y%m%d_%H%M%S)"

die() { echo "✗ $*" >&2; exit 1; }
step() { echo; echo "════ $* ════ $(date '+%F %T')"; }
done_mark() { touch "$LOG/$1.done"; }
is_done() { [[ -f "$LOG/$1.done" ]]; }
count_imgs() { python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['images']))" "$1"; }

# ─────────────────────────── 0. 開跑前檢查（任何一項不過就不開始訓練） ───────────────────────────
step "0. 開跑前檢查"
[[ "$(basename "$ROOT")" == "final" ]] || die "請在 koi/setA/final 底下執行（目前在 $ROOT）"
for f in train_maskrcnn.py train_yolo.py train_seg2.py final_jit_train.py final_fuse_eval.py \
         final_table.py make_crops_obb.py make_yolo.py eval_holdout_all.py eval_seg2_holdout.py \
         tta.py metrics.py postprocess.py; do
  [[ -f "scripts/$f" ]] || die "找不到 scripts/$f"
done
for f in instances_all.json holdout.json; do
  [[ -f "annotations/$f" ]] || die "找不到 annotations/$f"
done
for k in $FOLDS; do
  for s in train val; do [[ -f "annotations/fold${k}_$s.json" ]] || die "找不到 annotations/fold${k}_$s.json"; done
done
[[ -d images ]] || die "找不到 images/（訓練影像）"
[[ -d holdout ]] || die "找不到 holdout/（測試影像）"
python3 - <<'EOF' || die "影像與標註對不上（見上方）"
import json, pathlib, sys
bad = []
for ann, d in (("instances_all.json", "images"), ("holdout.json", "holdout")):
    for im in json.load(open(f"annotations/{ann}"))["images"]:
        if not (pathlib.Path(d) / im["file_name"]).exists():
            bad.append(f"{d}/{im['file_name']}")
for b in bad[:20]:
    print("  ✗ 找不到", b)
sys.exit(1 if bad else 0)
EOF
[[ -n "$YOLO_EVAL" && "$YOLO_EVAL" != *"<"* ]] \
  || die "YOLO_EVAL 未設定：請填入原本產生 YOLO holdout 結果檔的指令（見腳本最上方）"
read -r -a YM <<< "$YOLO_MODELS"; read -r -a YN <<< "$YOLO_NAMES"
[[ ${#YM[@]} -eq ${#YN[@]} ]] || die "YOLO_MODELS 與 YOLO_NAMES 數量不同"
python3 - <<'EOF' || die "缺少套件（見上方），請先 pip install"
import importlib.util, sys
need = {"torch": "torch", "torchvision": "torchvision", "ultralytics": "ultralytics", "scipy": "scipy",
        "cv2": "opencv-python-headless", "segmentation_models_pytorch": "segmentation-models-pytorch",
        "timm": "timm", "numpy": "numpy"}
miss = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
if miss:
    print("  ✗ 缺少：pip install " + " ".join(miss))
sys.exit(1 if miss else 0)
EOF
if [[ "$DEVICE" == cuda* || "$MR_DEVICE" == cuda* ]]; then
  python3 -c "import torch; assert torch.cuda.is_available()" || die "DEVICE=cuda 但偵測不到 GPU"
fi
echo "  ✓ 檢查通過：訓練 $(count_imgs annotations/instances_all.json) 張、holdout $(count_imgs annotations/holdout.json) 張"

mkdir -p "$LOG"
# 第一次執行：把舊的權重與結果改名保存，避免新舊混用
if [[ ! -f "$LOG/started" ]]; then
  for d in checkpoints checkpoints_obb_jit yolo_runs eval; do
    if [[ -e "$d" ]]; then mv "$d" "${d}_bak_$TS"; echo "  舊的 $d → ${d}_bak_$TS"; fi
  done
  echo "$TS" > "$LOG/started"
fi
mkdir -p eval

# 記錄環境（論文 Methods 的硬體與軟體版本）與標註檔指紋（之後可確認用的是哪一版標註）
{
  set +e
  echo "date: $(date)"; echo "host: $(hostname)"
  python3 -c "import sys, torch, torchvision, ultralytics, scipy, cv2; print('python', sys.version.split()[0]); print('torch', torch.__version__, '| cuda', torch.version.cuda, '| cudnn', torch.backends.cudnn.version()); print('torchvision', torchvision.__version__); print('ultralytics', ultralytics.__version__); print('scipy', scipy.__version__); print('opencv', cv2.__version__); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
  python3 -c "import segmentation_models_pytorch as s, timm; print('smp', s.__version__); print('timm', timm.__version__)"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo "settings: DEVICE=$DEVICE MR_DEVICE=$MR_DEVICE YOLO_DEVICE=$YOLO_DEVICE MR_EPOCHS=$MR_EPOCHS HR_EPOCHS=$HR_EPOCHS YOLO_EPOCHS=$YOLO_EPOCHS"
  echo "YOLO_MODELS=$YOLO_MODELS"; echo "YOLO_EVAL=$YOLO_EVAL"
  echo "annotations sha256:"; (cd annotations && sha256sum ./*.json 2>/dev/null || shasum -a 256 ./*.json)
} > "$LOG/env.txt" 2>&1 || true
echo "  環境 → $LOG/env.txt"

# ─────────────────────────── 0b. 在這台機器上重建衍生檔 ───────────────────────────
# crops_obb 與 yolo/ 都由標註產生；yolo/fold*/data.yaml 內含絕對路徑，換機器一定要重建
step "0b. 重建 crops_obb/ 與 yolo/"
if ! is_done prep; then
  python3 scripts/make_crops_obb.py 2>&1 | tee "$LOG/make_crops_obb.log"
  python3 scripts/make_yolo.py 2>&1 | tee "$LOG/make_yolo.log"
  for k in $FOLDS; do
    p="$(sed -n 's/^path: *//p' "yolo/fold$k/data.yaml")"
    [[ -d "$p" ]] || die "yolo/fold$k/data.yaml 的 path 不存在：$p"
  done
  done_mark prep
else
  echo "  已完成，跳過"
fi

# ─────────────────────────── 1. Mask R-CNN ───────────────────────────
step "1. Mask R-CNN（${MR_EPOCHS} epochs × 5 折）"
for k in $FOLDS; do
  is_done "maskrcnn_fold$k" && { echo "  fold $k 已完成，跳過"; continue; }
  python3 scripts/train_maskrcnn.py --fold "$k" --epochs "$MR_EPOCHS" --device "$MR_DEVICE" \
    2>&1 | tee "$LOG/maskrcnn_fold$k.log"
  done_mark "maskrcnn_fold$k"
done

# ─────────────────────────── 2. U-Net-HRNet-w32（框擾動版，最終設定） ───────────────────────────
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
else
  echo "  已完成，跳過"
fi

step "4b. YOLO holdout 評估"
if ! is_done eval_yolo; then
  bash -c "$YOLO_EVAL" 2>&1 | tee "$LOG/eval_yolo.log"
  for n in "${YN[@]}"; do
    for k in $FOLDS; do
      [[ -f "eval/hold5_${n}_fold$k.csv" ]] \
        || die "YOLO 評估沒有產生 eval/hold5_${n}_fold$k.csv，請檢查 YOLO_EVAL / YOLO_NAMES"
    done
  done
  done_mark eval_yolo
else
  echo "  已完成，跳過"
fi

# ─────────────────────────── 5. 論文表格 ───────────────────────────
step "5. 論文表格（數值、p 值、Holm、Friedman）"
python3 scripts/final_table.py --ref FUS_MaskRCNN FUS_fuse "${YN[@]}" \
  --anchor FUS_fuse --compare FUS_MaskRCNN "${YN[@]}" 2>&1 | tee "eval/paper_table.txt"

# ─────────────────────────── 6. 打包（權重、log、評估結果；不含影像） ───────────────────────────
step "6. 打包"
PKG="results_final_$(cat "$LOG/started")"
LIST="$LOG/package_files.txt"
{
  find checkpoints checkpoints_obb_jit -type f \( -name '*.pt' -o -name '*.json' -o -name '*.csv' -o -name '*.log' -o -name '*.txt' \) 2>/dev/null
  for d in yolo_runs runs; do
    [[ -d "$d" ]] && find "$d" -type f \( -name '*.pt' -o -name 'results.csv' -o -name 'args.yaml' \)
  done
  find eval -type f \( -name '*.csv' -o -name '*.txt' \)
  find logs/run_all -type f
  find scripts -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' \)
  ls annotations/fold*_*.json annotations/holdout.json annotations/instances_all.json
  echo run_all.out
} 2>/dev/null | sort -u | while read -r f; do [[ -f "$f" ]] && echo "$f"; done > "$LIST"
tar -czf "$PKG.tar.gz" -T "$LIST"
echo "  → $ROOT/$PKG.tar.gz（$(du -h "$PKG.tar.gz" | cut -f1)，$(wc -l < "$LIST") 個檔案；清單 $LIST）"

echo
echo "全部完成。"
echo "  論文表格　 eval/paper_table.txt"
echo "  環境版本　 $LOG/env.txt"
echo "  全部打包　 $PKG.tar.gz（權重 .pt、訓練 log、評估 CSV、論文表格、腳本、標註；不含影像）"
