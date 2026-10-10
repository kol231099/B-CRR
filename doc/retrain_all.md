# RunPod 執行說明：重標後重跑 Table 1 的五個模型

> 給 CRR_PA 端執行者。這份文件自成一體：讀完就知道五個模型的最終配置、要準備什麼、
> 怎麼在 RunPod 上一次跑完，以及跑完要交回什麼。

一支腳本 [`run_all_final.sh`](../koi/setA/scripts/run_all_final.sh) 依序完成：
重建衍生檔 → 訓練五個模型（各五折）→ holdout 評估 → 論文表格（含 p 值、Friedman）→ 打包所有權重與紀錄。

---

## 1. 五個模型的最終配置

五個模型使用**同一組切分**（`final/annotations/` 的 `fold{k}_train.json`、`fold{k}_val.json`、
`holdout.json`），偵測門檻 0.35，所有指標都在原圖座標計算，**不使用 TTA**。

| 論文中的名稱 | 訓練 | 超參數 |
| --- | --- | --- |
| Mask R-CNN | `train_maskrcnn.py` | 40 epochs、batch 2、lr 1e-4、AdamW（wd 1e-4）、cosine、grad clip 10、mask head 28×28 |
| **Mask R-CNN + U-Net-HRNet-w32**（本研究方法） | 第一階段：直接用上一列的 Mask R-CNN 權重<br>第二階段：`final_jit_train.py` | 第二階段 40 epochs、batch 4、lr 1e-4、AdamW（wd 1e-4）、cosine、BCE + Dice、輸入 512×256<br>**框擾動**（見下） |
| YOLOv8-seg | `train_yolo.py --model yolov8s-seg.pt` | 300 epochs、batch 4、imgsz 1024、`optimizer: auto`、seed 0、COCO 預訓練 |
| YOLO11-seg | `train_yolo.py --model yolo11s-seg.pt` | 同上 |
| YOLO26-seg | `train_yolo.py --model yolo26s-seg.pt` | 同上 |

### 本研究方法（Mask R-CNN + U-Net-HRNet-w32）的完整流程

1. Mask R-CNN 偵測每顆牙，輸出遮罩機率圖 P_MaskRCNN。
2. 以遮罩的最小面積外接矩形建立斜框（OBB），四邊各外擴 20%，轉正裁切，縮放到 512×256。
3. U-Net-HRNet-w32（encoder：HRNet-w32，decoder：U-Net）輸出機率圖，轉回原圖座標得到 P_HRNet。
4. **機率融合**：`mask = clean_mask(0.5 · P_HRNet + 0.5 · P_MaskRCNN > 0.5)`
   （`clean_mask`：保留最大連通區域並填洞）。權重固定 0.5，未在任何資料上調整。

**框擾動（第二階段訓練時）**：訓練用的裁切框不是固定的「GT 外接矩形 + 20%」，而是每次取樣時
隨機擾動——旋轉 U(±5°)、沿長短軸各平移 U(±5%) 邊長、長短邊各縮放 U(0.92–1.08)；驗證集用固定
種子的擾動。目的：原本的固定框讓牙齒四個極點永遠落在裁切圖的固定位置，網路學會「照著框邊畫」，
推論時框改由 Mask R-CNN 產生，HRNet 就只會重畫 Mask R-CNN 的結果。擾動後網路改為看影像找邊界。

> ⚠ 第二階段**必須用 `final_jit_train.py`**。用 `train_seg2.py` 直接訓練得到的是沒有框擾動的舊版本，
> 結果會差（holdout HD95：舊版 12.24 px，本方法 9.53 px）。腳本已寫死這一點，不需手動處理。

---

## 2. 資料（重標後）

| | 影像 | 牙齒 |
| --- | --- | --- |
| 訓練（五折） | 96 | 202 |
| 　其中五折輪流當驗證 | 63 | 每折 29–32 |
| 　只當訓練 | 33 | |
| holdout（未重標、未變動） | 18 | 26 |

---

## 3. 在 RunPod 上準備

1. **上傳** `koi/setA/final/` 整個資料夾，至少要有：`annotations/`、`images/`、`holdout/`、`scripts/`。
   `crops_obb/` 與 `yolo/` 不必上傳，腳本會在 RunPod 上重建（`yolo/fold*/data.yaml` 內含絕對路徑，
   換機器一定要重建）。
2. **更新腳本**：從 B-CRR 分支 `claude/rcnn-hrnet-w32-performance-mpc373` 取最新版放進 `final/scripts/`：
   `run_all_final.sh`、`final_jit_train.py`、`final_fuse_eval.py`、`final_table.py`。
3. **套件**（缺什麼腳本會列出來）：
   ```bash
   pip install torch torchvision ultralytics scipy opencv-python-headless segmentation-models-pytorch timm
   ```
4. **填 `YOLO_EVAL`**：產生 YOLO holdout 結果檔 `eval/hold5_yolov8sseg_fold{k}.csv`（及 yolo11sseg、
   yolo26sseg）的指令。**必須是原本 Table 1 產生 YOLO 結果的同一支程式**，評估路徑才與其他模型一致。
   檔名若不同，一併改腳本上方的 `YOLO_NAMES`。

---

## 4. 執行

```bash
tmux new -s train            # 斷線不中止
cd koi/setA/final
YOLO_EVAL='<原本產生 YOLO holdout 結果的指令>' bash scripts/run_all_final.sh 2>&1 | tee run_all.out
# 離開 tmux：Ctrl-b 再按 d；回來：tmux attach -t train
```

看到「✓ 檢查通過」且 Mask R-CNN 開始出現 epoch，就代表正常在跑。

### 腳本自動完成的事

| 步驟 | 內容 |
| --- | --- |
| 0. 檢查 | 位置是否在 `final/`、腳本與標註齊全、每張影像都找得到、套件齊全、GPU 可用、`YOLO_EVAL` 已填；**任何一項不過就不開始訓練** |
| 備份 | 第一次執行時把既有的 `checkpoints/`、`checkpoints_obb_jit/`、`yolo_runs/`、`eval/` 改名為 `*_bak_<時間>` |
| 環境紀錄 | Python、PyTorch、CUDA、cuDNN、ultralytics、smp、timm、GPU 型號、所有設定、標註檔 sha256 → `logs/run_all/env.txt` |
| 0b. 重建 | `make_crops_obb.py`、`make_yolo.py`，並確認 `data.yaml` 路徑存在 |
| 1. Mask R-CNN | 5 折 × 40 epochs |
| 2. HRNet 框擾動版 | 5 折 × 40 epochs（`final_jit_train.py`） |
| 3. YOLO ×3 | 15 次 × 300 epochs |
| 4. 評估 | `final_fuse_eval.py --no-tta` 產生 `FUS_MaskRCNN`（單階段）與 `FUS_fuse`（本方法）；評估前會檢查權重比標註新。接著執行 `YOLO_EVAL` |
| 5. 表格 | `final_table.py`：五列指標、各模型對本方法的配對 Wilcoxon（Holm 校正）、Friedman |
| 6. 打包 | 所有權重與紀錄壓成 `results_final_<時間>.tar.gz`（見下） |

預估時間（RTX 4090 / A100 等級）：Mask R-CNN 約 1 小時、HRNet 約 0.5 小時、YOLO 4–7.5 小時，
合計約 6–10 小時。

### 中斷後續跑

每完成一步會在 `logs/run_all/` 留下 `.done`。中斷後**重跑同一行指令**即可，已完成的部分自動跳過。
要重跑某一步，刪掉對應的 `.done`（例如 `logs/run_all/hrnet_jit_fold2.done`）。

---

## 5. 產出與交回

最後會產生 **`final/results_final_<時間>.tar.gz`**，內容（不含任何影像）：

| 內容 | 路徑 |
| --- | --- |
| Mask R-CNN 權重（5 折） | `checkpoints/**/maskrcnn_fold*.pt` |
| HRNet 框擾動版權重（5 折）與每折訓練設定 | `checkpoints_obb_jit/seg2/unet_tu-hrnet_w32/fold*.pt`、`train_config_fold*.json` |
| YOLO 權重（3 模型 × 5 折） | `yolo_runs/**/weights/*.pt`、`results.csv`、`args.yaml` |
| holdout 逐顆牙結果 | `eval/hold5_*_fold*.csv` |
| 論文表格 | `eval/paper_table.txt`、`eval/table_pvalues_FUS_fuse.csv` |
| 環境版本與標註指紋 | `logs/run_all/env.txt` |
| 每一步的完整 log | `logs/run_all/*.log`、`run_all.out` |
| 本次使用的腳本與切分 | `scripts/*.py`、`scripts/*.sh`、`annotations/*.json` |

**請交回**：

1. `eval/paper_table.txt` 全文（論文 Table 1 與 p 值由此更新）
2. `logs/run_all/env.txt` 全文（Methods 的硬體與軟體版本）
3. `run_all.out` 最後 50 行（確認每一步都完成）
4. `results_final_<時間>.tar.gz` 保存好（權重之後推論與審稿回覆都會用到）

重標後五個模型都偵測到的牙數可能不再是 13，論文的數字、p 值與註腳會跟著更新。
