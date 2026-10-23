# 論文方法（「黃色版」）完整規格

> 給網站／量測系統的維護者：本文件定義論文 Table 1 中
> **Mask R-CNN + U-Net-HRNet-w32** 的完整推論流程。網站的分割結果必須與本規格**逐像素相同**。
> 參考實作是 [`paper_pipeline.py`](../koi/setA/scripts/paper_pipeline.py) 的 `segment()`，
> 它與論文評估用的 [`final_fuse_eval.py`](../koi/setA/scripts/final_fuse_eval.py) 推論路徑相同
> （可用 `paper_pipeline.py --check` 逐像素驗證）。
>
> 「黃色版」是根尖分析圖（`apex_lab.py`）裡黃色輪廓的那一版，即論文定稿的方法。

---

## 1. 流程總覽

```
灰階根尖片
  │
  ├─① Mask R-CNN ──→ 每顆牙一張軟遮罩 P_MaskRCNN（全圖大小、0–1 機率）＋ 分數
  │                   只留分數 ≥ 0.35
  │
  ├─② 斜框：clean_mask(P_MaskRCNN > 0.5) 的最小面積外接矩形，四邊外擴 20%
  │
  ├─③ 轉正裁切 → 縮放 512×256 → U-Net-HRNet-w32 → sigmoid
  │   → 放大回裁切尺寸 → 反仿射貼回原圖 = P_HRNet（全圖大小、0–1 機率）
  │
  └─④ 融合：mask = clean_mask( 0.5·P_HRNet + 0.5·P_MaskRCNN > 0.5 )
```

`clean_mask`：保留最大的 8-連通區域，再填滿內部孔洞（[`postprocess.py`](../koi/setA/scripts/postprocess.py)）。

**不做任何 TTA**（不翻轉、不多尺度）。

---

## 2. 逐步規格

### ① Mask R-CNN

| 項目 | 規格 |
| --- | --- |
| 架構 | `torchvision.models.detection.maskrcnn_resnet50_fpn_v2`，2 類（背景＋牙齒），遮罩頭 28×28（`train_maskrcnn.build_model(False, 28)`） |
| 輸入 | 灰階影像 `uint8` → `float / 255` → 複製成 3 通道；**不做其他前處理**（縮放由 torchvision 內建 transform 處理） |
| 輸出取用 | `out["scores"] >= 0.35` 的偵測；`out["masks"][keep, 0]` 是**軟遮罩（機率）**，必須保留到第④步 |
| 權重 | 見第 3 節 |

### ② 斜框（OBB）

| 項目 | 規格 |
| --- | --- |
| 遮罩 | `cm = clean_mask(P_MaskRCNN > 0.5)`；空的就跳過這顆 |
| 斜框 | `obb_of(cm)`：最大輪廓的 `cv2.minAreaRect`，長邊轉為直立（[`make_crops_obb.py`](../koi/setA/scripts/make_crops_obb.py)） |
| 外擴 | `warp_of(..., pad=0.2)`：裁切尺寸 = 邊長 × (1 + 2×0.2) |
| 裁切 | `cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR)` |

### ③ U-Net-HRNet-w32

| 項目 | 規格 |
| --- | --- |
| 架構 | `segmentation_models_pytorch` U-Net，encoder `tu-hrnet_w32`（`train_seg2.build_seg2("unet", "tu-hrnet_w32")`） |
| 輸入 | `cv2.resize(crop, (256, 512), interpolation=cv2.INTER_AREA)`（寬 256、高 512）→ `float / 255` → 3 通道 |
| 輸出 | `sigmoid(logits)`；多個權重（集成）時**機率取平均** |
| 貼回 | `cv2.resize(prob, (cw, ch), INTER_LINEAR)` → `cv2.warpAffine(prob, invertAffineTransform(M), (W, H), INTER_LINEAR)`；**此時不二值化** |
| 權重 | **框擾動訓練版** `checkpoints_obb_jit/seg2/unet_tu-hrnet_w32/fold{k}.pt`（見第 3 節）。**不是** `checkpoints_obb/`（那是舊版，會照著框畫） |

### ④ 融合

```python
mask = clean_mask(0.5 * P_HRNet + 0.5 * P_MaskRCNN > 0.5)
```

- `P_MaskRCNN` 是同一顆牙在第①步的軟遮罩（**不是**二值化後的遮罩）
- 權重固定 0.5，未在任何資料上調整
- 這張 `mask` 就是後續 CRR / B-CRR 量測（`scripts/measure.py`）的輸入

---

## 3. 權重與訓練資料

### 論文評估（Table 1）用的權重

重標後的資料（訓練 96 張／202 顆，五折；holdout 18 張／26 顆），於 RunPod（RTX 4090）以
[`run_all_final.sh`](../koi/setA/scripts/run_all_final.sh) 訓練：

| 模型 | 權重 | 訓練 |
| --- | --- | --- |
| Mask R-CNN | `checkpoints/` 底下的 `maskrcnn_fold{k}.pt`（`train_maskrcnn.py` 存放的位置），k = 0–4 | 40 epochs、batch 2、lr 1e-4、AdamW |
| U-Net-HRNet-w32 | `checkpoints_obb_jit/seg2/unet_tu-hrnet_w32/fold{k}.pt`，k = 0–4 | 40 epochs、batch 4、lr 1e-4、AdamW、BCE+Dice；**框擾動**：旋轉 U(±5°)、沿長短軸各平移 U(±5%)、長短邊各縮放 U(0.92–1.08) |

權重與訓練紀錄在 `results_final_<時間>.tar.gz`（`logs/run_all/env.txt` 內有標註檔 sha256）。
論文的數字是每一折的 Mask R-CNN 搭配**同一折**的 HRNet，分別評估後取平均。

### 網站部署建議

網站處理的是新影像，不屬於任何一折，因此：

| 項目 | 建議 | 說明 |
| --- | --- | --- |
| HRNet | 五折**集成**：五個 `checkpoints_obb_jit` 權重的機率取平均 | 方法相同，集成只會更穩 |
| Mask R-CNN | **待決定**：(a) 以相同設定用全部 96 張重訓 `maskrcnn_final.pt`（重標後資料）；或 (b) 暫用某一折權重 | 現有的 `maskrcnn_final.pt` 是**重標前**的資料訓練的，不可再用 |
| 融合 | 用集成後的 P_HRNet 與 Mask R-CNN 的軟遮罩，0.5 : 0.5 | 與第④步相同 |

---

## 4. 目前網站（`server/app.py`）與本規格的差異

以 commit 時的 `server/app.py` 檢查，**有四處與論文方法不同**：

| # | 位置 | 目前網站 | 論文方法 |
| --- | --- | --- | --- |
| 1 | `load_segmenters()` 第 164 行 | 讀 `checkpoints_obb/`（**舊版，無框擾動**） | 讀 `checkpoints_obb_jit/`（框擾動、重標後資料） |
| 2 | `segment_in_obb()` 第 214 行 | `HRNet 機率 > 0.5`，**只用 HRNet** | `0.5·P_HRNet + 0.5·P_MaskRCNN > 0.5`（融合） |
| 3 | `detect_obb()` 第 182 行 | Mask R-CNN 遮罩**立即二值化**，軟遮罩被丟掉 | 軟遮罩須保留並傳到融合步驟 |
| 4 | `load_detector()` 第 144 行 | `maskrcnn_final.pt`（**重標前**資料） | 重標後資料的 Mask R-CNN（見第 3 節） |

以下幾點與規格**等價、不需要改**：

- `roi_heads.score_thresh = 0.35`、`detections_per_img = 12`（第 151–152 行）：提前在模型內過濾，
  保留下來的偵測與「先輸出再以 0.35 過濾」相同（除非一張片子超過 12 顆牙）
- 長邊超過 `MAX_SIDE`（2400）先縮小：資料集影像最大 1384×1747，不會觸發
- 偵測結果依框的 x 座標排序：只影響輸出順序

---

## 5. 驗證方式（網站改完後）

1. **與論文評估逐像素一致**（在 `koi/setA/final` 底下，需有 holdout 影像與重標後權重）：
   ```bash
   python3 scripts/paper_pipeline.py --check
   ```
   應印出「✅ 完全一致」。這確認參考實作 = 論文評估。
2. **網站與參考實作一致**：同一張影像、同一組權重，分別跑網站與
   ```bash
   python3 scripts/paper_pipeline.py --image X.jpg --maskrcnn <Mask R-CNN 權重> \
       --hrnet checkpoints_obb_jit/seg2/unet_tu-hrnet_w32/fold{0,1,2,3,4}.pt --out ref.png
   ```
   比對每顆牙的面積與外接框（腳本會印出）。CPU 上應完全相同；GPU 可能有個位數像素的浮點差異。

---

## 6. 與舊版的差別（為什麼要換）

舊版（`checkpoints_obb/` + 只用 HRNet）的第二階段是用「GT 遮罩的外接矩形 + 固定 20% 外擴」
訓練的：牙齒四個極點永遠落在裁切圖的固定位置，網路學會**照著框邊畫**。推論時框來自
Mask R-CNN，HRNet 就只是重畫 Mask R-CNN 的結果（兩者 Dice 一致度 0.981）。框擾動讓網路改看
影像找邊界，融合再讓兩個模型互補。完整診斷見 [`hrnet_improvement.md`](hrnet_improvement.md)。
