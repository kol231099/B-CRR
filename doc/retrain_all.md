# 一次重跑 Table 1：五個模型 × 五折 ＋ holdout 評估 ＋ 論文表格

一支腳本 [`run_all_final.sh`](../koi/setA/scripts/run_all_final.sh) 跑完重標後所需的全部步驟：
訓練五個模型（各五折）→ holdout 評估 → 產生含 p 值與 Friedman 檢定的論文表格。
HRNet 用的是最佳版本（框擾動訓練 ＋ 機率融合），細節見 [`hrnet_improvement.md`](hrnet_improvement.md)。

## 1. 五個模型的設定

與原本 Table 1 相同，五個模型使用同一組切分（`final/annotations/`）。

| 模型 | 訓練腳本 | 設定 |
| --- | --- | --- |
| Mask R-CNN | `train_maskrcnn.py` | 40 epochs、batch 2、lr 1e-4、AdamW、mask head 28×28 |
| **Mask R-CNN + U-Net-HRNet-w32**（本方法） | `final_jit_train.py` | 第二階段 40 epochs、batch 4、lr 1e-4、輸入 512×256；**框擾動**：旋轉 ±5°、平移 ±5%、縮放 ±8%；推論時與 Mask R-CNN 機率以 0.5 : 0.5 **融合** |
| YOLOv8s-seg / YOLO11s-seg / YOLO26s-seg | `train_yolo.py` | 300 epochs、batch 4、imgsz 1024 |

本方法的第一階段直接使用上表 Mask R-CNN 的權重，不另外訓練。HRNet 必須用
`final_jit_train.py` 訓練；用 `train_seg2.py` 訓練出來的是沒有框擾動的舊版本。

## 2. 開跑前

在 `koi/setA/final` 底下：

1. **標註已換成新版**，且已重建衍生檔：
   ```bash
   python3 scripts/make_crops_obb.py
   python3 scripts/make_yolo.py
   ```
2. **下載最新腳本到 `final/scripts/`**（B-CRR 分支 `claude/rcnn-hrnet-w32-performance-mpc373`）：
   `run_all_final.sh`、`final_jit_train.py`、`final_fuse_eval.py`、`final_table.py`
3. **填 `YOLO_EVAL`**：產生 YOLO holdout 結果檔（`eval/hold5_yolov8sseg_fold{k}.csv` 等）
   的指令。請用原本 Table 1 產生 YOLO 結果的**同一支程式**，評估路徑才會一致。
   檔名若不是 `yolov8sseg / yolo11sseg / yolo26sseg`，一併改腳本上方的 `YOLO_NAMES`。

## 3. 執行

```bash
cd koi/setA/final
YOLO_EVAL='python3 scripts/<原本的 YOLO holdout 評估腳本>.py <參數>' \
  nohup bash scripts/run_all_final.sh > run_all.out 2>&1 &
tail -f run_all.out
```

RunPod 等遠端機器建議用 `nohup` 或 `tmux`，斷線也不會中止。裝置預設 `cuda`（YOLO 用 GPU 0），
要改可在前面加 `DEVICE=mps YOLO_DEVICE=mps`。

### 腳本會自動做的事

| 步驟 | 內容 |
| --- | --- |
| 0. 檢查 | 位置是否在 `final/`、腳本與標註是否齊全、`crops_obb/` 與 `yolo/` 是否比標註新、`YOLO_EVAL` 是否已填；任何一項不過就**不開始訓練** |
| 備份 | 第一次執行時把舊的 `checkpoints/`、`checkpoints_obb_jit/`、`yolo_runs/`、`eval/` 改名為 `*_bak_<時間>`，避免新舊權重混用 |
| 環境紀錄 | Python、PyTorch、CUDA、ultralytics、GPU 型號寫到 `logs/run_all/env.txt`（論文 Methods 需要） |
| 1–3. 訓練 | Mask R-CNN → HRNet 框擾動版 → 三個 YOLO，各五折 |
| 4. 評估 | `final_fuse_eval.py --no-tta` 產生 `FUS_MaskRCNN`（單階段）與 `FUS_fuse`（本方法）；再執行 `YOLO_EVAL`。評估前會檢查權重是否比標註新，避免拿到重標前的權重 |
| 5. 表格 | `final_table.py` 輸出論文表格、各模型對本方法的 Wilcoxon p 值（Holm 校正）與 Friedman 檢定 |

### 中斷後續跑

每完成一折會在 `logs/run_all/` 留下 `.done`。中斷後直接再執行同一行指令，已完成的部分會跳過。
想重跑某一折，刪掉對應的 `.done`（例如 `logs/run_all/hrnet_jit_fold2.done`）。

## 4. 產出

| 檔案 | 內容 |
| --- | --- |
| `eval/paper_table.txt` | 論文表格（數值、p 值、Friedman），以及共同命中牙數 n |
| `eval/table_pvalues_FUS_fuse.csv` | 同上，CSV 格式 |
| `logs/run_all/env.txt` | 硬體與軟體版本 |
| `logs/run_all/*.log` | 每一步的完整輸出 |

`eval/paper_table.txt` 與 `logs/run_all/env.txt` 貼回來，即可更新 Results 與 Methods。
重標後共同命中牙數可能不再是 13，論文的數字、p 值與註腳都要跟著更新。
