# 兩階段 Mask R-CNN + U-Net-HRNet-w32：從輸給單階段到贏過單階段

這份記錄兩階段方法改了什麼、為什麼這樣改、怎麼驗證。評估條件與
[`segmentation_benchmark.md`](segmentation_benchmark.md) 的 Table 1 完全相同
（`koi/setA/final/`：93 張訓練、18 張 / 26 顆 holdout、n = 13 共同命中、
逐折中位數再平均五折、門檻 0.35、pad 0.2）。

## 1. 結果

### 不加 TTA（與原 Table 1 同條件）

| Pipeline | DICE ↑ | IOU ↑ | HD95 (px) ↓ | ASSD (px) ↓ | Area ICC ↑ |
| --- | --- | --- | --- | --- | --- |
| Mask R-CNN | 0.9742 | 0.9496 | 9.97 | 3.57 | 0.9997 |
| 兩階段（改前） | 0.9727 | 0.9469 | 11.76 | 3.54 | 0.9997 |
| **兩階段（改後）** | **0.9770** | **0.9549** | **9.53** | **3.48** | **0.9998** |

### 加 TTA（四向翻轉；YOLO-seg 在 ultralytics 不支援 TTA，故不列）

| Pipeline | DICE ↑ | IOU ↑ | HD95 (px) ↓ | ASSD (px) ↓ | Area ICC ↑ |
| --- | --- | --- | --- | --- | --- |
| Mask R-CNN + TTA | 0.9735 | 0.9484 | 9.41 | 3.59 | 0.9997 |
| **兩階段（改後）+ TTA** | **0.9764** | **0.9540** | **8.73** | **3.37** | **0.9998** |

### 配對 Wilcoxon（每顆牙先取五折平均）

| 比較 | 牙數 | DICE | HD95 |
| --- | --- | --- | --- |
| 改後 vs Mask R-CNN | 13（共同命中） | 10/13 勝，p = 0.003 | 9/13 勝，**p = 0.110（未達顯著）** |
| 改後 vs Mask R-CNN | 24（Mask R-CNN 五折全命中） | 21/24 勝，p < 0.001 | 20/24 勝，p = 0.001 |
| 改後 vs 改前（同設定重訓） | 13 | 13/13 勝，p < 0.001 | 10/13 勝，p = 0.048 |
| 改後 + TTA vs Mask R-CNN + TTA | 13 | 10/13 勝，p = 0.008 | 11/13 勝，p = 0.021 |

## 2. 改了什麼：只有兩處

**Mask R-CNN 完全沒動**——同一組權重、同一個門檻、同一套斜框導出。第二階段的
架構（U-Net × HRNet-w32）與訓練超參（40 epochs、AdamW lr 1e-4、wd 1e-4、
cosine、BCE + Dice、batch 4、每 5 epoch 驗證存最佳、seed 0）也完全沒動。

### 改動一：第二階段訓練時對斜框加隨機擾動

| | 改前 | 改後 |
| --- | --- | --- |
| 訓練 crop | `crops_obb/` 預裁好的圖：GT 遮罩的 `minAreaRect` 加固定 pad 0.2 | 每次取樣從原圖現場裁切，GT 斜框先加隨機擾動 |
| 擾動 | 無 | 旋轉 U(−5°, 5°)、沿兩軸各平移 U(−5%, 5%) 邊長、兩邊各自縮放 U(0.92, 1.08) |
| 驗證集 | 預裁圖 | 同樣擾動，但每個樣本固定種子（每次驗證看到同一組擾動） |

其餘增強（翻轉、gamma）與縮放到 512×256 仍走原本 `CropDataset` 的程式碼。

程式：[`koi/setA/scripts/final_jit_train.py`](../koi/setA/scripts/final_jit_train.py)
——直接呼叫 `final/scripts/train_seg2.py` 的 `main()`，只把資料集換成
`JitterCropDataset`。

### 改動二：最終遮罩改為兩階段機率圖的等權融合

```
改前：mask = clean_mask(P_HRNet > 0.5)
改後：mask = clean_mask(0.5 · P_HRNet + 0.5 · P_MaskRCNN > 0.5)
```

`P_MaskRCNN` 是 Mask R-CNN 對這顆牙本來就算好的機率圖（貼回原圖、未二值化），
`P_HRNet` 是第二階段輸出轉回原圖的機率圖。權重 0.5 事先固定，未在任何資料上調整。
融合幾乎不增加推論時間。

程式：[`koi/setA/scripts/final_fuse_eval.py`](../koi/setA/scripts/final_fuse_eval.py)
的 `run_variants()`。

## 3. 為什麼這樣改：診斷過程

### 3.1 舊版第二階段在「照著框畫」

訓練框是 GT 遮罩的外接矩形加固定 pad，所以每張訓練 crop 裡牙齒的四個極點
都落在固定位置（約 14.3% 與 85.7%）。網路學到的捷徑是「沿著框往內縮一段」。

[`diag_box_leak.py`](../koi/setA/scripts/diag_box_leak.py) 的量測（舊切分 OOF，155 顆）：

| 指標 | 改前 | 加擾動後 |
| --- | --- | --- |
| 邊界跟隨斜率（框邊推 1 px，預測邊界跟著動多少 px） | 0.23–0.54 | 0.01–0.05 |
| 框偏 4% 時的 Dice（Oracle 框加雜訊） | 0.959 | 0.970（不受影響） |
| 第二階段輸出 vs Mask R-CNN 遮罩的 Dice | 0.981 | — |

換框歸因：把 GT 框只換掉角度或中心成 Mask R-CNN 的值，改前分別掉到 0.966 與
0.964，加擾動後兩者都是 0.970，不再受影響。

**結論**：框是從 Mask R-CNN 的遮罩算出來的，第二階段又照著框畫，等於在重畫
Mask R-CNN 的遮罩（兩者 Dice 0.981），所以永遠贏不了它。先前十二個介入
（換偵測器、調 padding、OOF 訓練、穩健框估計……）都沒處理到這一點，因此都無效。

### 3.2 為什麼單有擾動還不夠，要加融合

擾動讓第二階段不再抄框、改為看影像找邊界；這時它的錯誤與 Mask R-CNN 不再相同，
兩者的機率圖平均才有互補效果。2×2 消融（HD95，n = 13，四個第二階段模型皆以
同一支腳本、同一台機器、相同超參訓練）：

| | 不融合 | 融合 |
| --- | --- | --- |
| 不擾動 | 12.24 | 12.03 |
| 擾動 | 10.48 | **9.53** |

- 只融合不擾動：幾乎沒用——兩個模型錯在同樣的地方，平均起來沒有新資訊
- 只擾動：有改善，但仍輸 Mask R-CNN
- 兩者一起：才贏過 Mask R-CNN

### 3.3 試過但無效、已放棄的方向

在舊切分的 OOF 上比較了 15 種只改推論的做法
（[`hd95_lab.py`](../koi/setA/scripts/hd95_lab.py)）：調二值化門檻（0.3–0.7）、
高斯平滑機率圖（σ 1–3）都無顯著效果；HD95 的來源多圈、少圈約各半，
沒有系統性偏誤可以靠門檻修正。

### 3.4 改後的進一步嘗試（都無效，不必再試）

全部在 `final/` 的 OOF（63 張、152 顆牙；每張由沒看過它的那一折推論，未碰 holdout）
上用 [`final_lab.py`](../koi/setA/scripts/final_lab.py) 評估。OOF 上改後本身相對 Mask R-CNN：
HD95 中位 10.69 → 10.00、平均 13.42 → 12.23，64% 的牙較好，p < 0.0001。

| 嘗試 | 與現行融合相比 | 結論 |
| --- | --- | --- |
| 融合權重 0.3 / 0.4 / 0.6 / 0.7 | HD95 中位全為 10.00 | 0.5 即最佳，無須調 |
| 依信心加權（每像素聽較有把握的模型） | 結果與 0.5 完全相同 | 無效 |
| 依區域加權（冠／側／根各學一個權重，交叉擬合） | +0.01 px，p = 0.94 | 學出的權重都在 0.4–0.7，無位置規律 |
| 邏輯迴歸融合（機率、亮度、邊緣、軸向位置，交叉擬合） | +0.10 px，p = 0.03（變差） | 無效 |
| 保留與 Mask R-CNN 重疊的所有連通塊（取代只留最大塊） | 完全相同 | 無效：沒有斷塊被丟的情形 |
| 框擾動 + 邊界損失（α_max 0.5，40 epochs） | −0.03 px，p = 0.55 | 無效 |

大錯（HD95 ≥ 20 px，16 顆）的三個假說——斷塊被 `clean_mask` 丟掉、GT 超出 crop、
補綴物——命中數全為 0；這些牙在 Mask R-CNN 上同樣很差（例如 89.jpg 76 → 85 px）。

**誤差拆解**：所有錯誤像素中 52% 是兩個模型錯在同一處，任何組合都救不回；在冠、側、
根三區的比例都約一半，最差 10% 的牙只貢獻其中 25%（不集中）。剩下可救的部分無法
由推論時可得的資訊預測（見上表的學習式融合），所以組合類方法已到頂。

**解析度天花板**（[`final_res_ceiling.py`](../koi/setA/scripts/final_res_ceiling.py)：把 198 顆 GT
遮罩當作完美輸出，只走縮放與貼回）：

| 流程 | HD95 中位 | HD95 P95 |
| --- | --- | --- |
| 第二階段 512×256 | 1.00 | 1.00 |
| 第二階段 768×384 | 1.00 | 1.00 |
| Mask R-CNN 遮罩頭 28×28 | 5.39 | 8.27 |
| Mask R-CNN 遮罩頭 56×56 | 2.00 | 4.00 |
| 融合（512 + 28） | 1.41 | 2.00 |

第二階段提高解析度沒有空間；實際約 10 px 的誤差中，幾何上限只佔 1–1.5 px，其餘來自
預測本身與標註的不確定性。另一方面，Mask R-CNN 的 28×28 遮罩頭本身就有 5.4 px 的
HD95 下限（約為其實際誤差的一半），這是兩階段架構的直接動機：第一階段定位，第二階段
以 512×256 的解析度描邊界。

**`exp/` 舊實驗的注意事項**：`exp/eval_fold0.py` 預設以 GT 導出的 oracle OBB 評估，
因此 `eval_A_BLOSS`、`eval_A_RES768`、`eval_A_ROT5/18`、`eval_A_CLAHE` 都是在 oracle 框下
與 oracle 基準（HD95 7.00）比較。Oracle 框會獎勵照抄框的模型，凡是減少抄框的改動在
oracle 下都會變差，所以那批「失敗」不能直接當成在實際串接中無效的證據；本節的邊界損失
與解析度天花板是在實際串接下重新驗證的結果。

**下一步的建議**：模型側已近極限。最有價值的是請第二位標註者重標 20–30 顆牙，
量測人與人之間的 Dice / HD95，以確認天花板。

## 4. 驗證：確定比較是公平的

1. **推論路徑重現**：`final_fuse_eval.py --check` 用同一段程式碼、原本的權重重算
   Table 1 的 Mask R-CNN 與兩階段（改前），與既有 CSV 逐顆比對——128 筆 TP，
   TP 集合與 Dice / HD95 / ASSD 全部一致。
2. **出表規則重現**：[`final_table.py`](../koi/setA/scripts/final_table.py) 重算的
   原五列與 Table 1 每個數字都相同，n = 13。
3. **訓練設定確認**：原兩階段的訓練紀錄已不存在（RunPod）。以 `--no-jitter`、
   40 epochs 重訓後，各折最佳 val Dice 為 0.9752 / 0.9745 / 0.9707 / 0.9730 / 0.9700，
   原權重存的是 0.9752 / 0.9729 / 0.9707 / 0.9727 / 0.9732——fold 0 與 fold 2 到小數
   第四位完全相同，確認原設定為 40 epochs；其餘差異來自 GPU 與 MPS 的數值差。
4. **跑與跑之間的變異**：同設定重訓的兩階段（改前）HD95 為 12.24，原 Table 1 為
   11.76，差約 0.5 px。改後相對 Mask R-CNN 的 HD95 改善（0.44 px）與此同量級，
   所以不加 TTA、n = 13 時只宣稱 DICE / IOU 顯著，HD95 宣稱需引用 n = 24 或 TTA 的結果。

## 5. 重現

在 `koi/setA/final/` 底下（腳本放進 `final/scripts/`）：

```bash
# 0. 確認推論路徑與 Table 1 相同（應顯示「完全吻合」）
python3 scripts/final_fuse_eval.py --check MaskRCNN MaskRCNN_OBB_HRNet

# 1. 訓練第二階段：改後（擾動）；--no-jitter 則為改前的同設定重訓（消融用）
for k in 0 1 2 3 4; do
  python3 scripts/final_jit_train.py --arch unet --encoder tu-hrnet_w32 --fold $k --epochs 40 --device mps
done

# 2. holdout 評估（輸出 eval/hold5_FUS_*_fold{0..4}.csv）
python3 scripts/final_fuse_eval.py

# 3. 出表（上半部為原 Table 1，下半部為新方法與消融）
python3 scripts/final_table.py --ref MaskRCNN MaskRCNN_OBB_HRNet yolov8sseg yolo11sseg yolo26sseg
```

權重寫到 `final/checkpoints_obb_jit/`（擾動）與 `final/checkpoints_obb_base/`（不擾動），
不覆蓋原本的 `checkpoints_obb/`；每折另存 `train_config_fold{k}.json` 記錄完整參數。

## 6. 論文方法章節需寫明

- 第二階段訓練時對 GT 斜框施加隨機擾動（旋轉 ±5°、平移 ±5%、縮放 ±8%）
- 最終遮罩為兩階段機率圖等權平均後以 0.5 二值化
- 統計：每顆牙五折平均後做配對 Wilcoxon 符號等級檢定

另外 `segmentation_benchmark.md` 第 6 節把 Oracle OBB（HD95 8.08）寫成第二階段
能力的上限，這個推論不成立：Oracle 的框由 GT 遮罩導出，而改前的第二階段會照著框
邊界畫（3.1 節），所以 Oracle 的分數含有 GT 洩漏，只能當作「框完美時」的參考值。

## 7. 部署成本

融合只是兩張已算好的機率圖相加，推論時間與改前相同。一張有 N 顆牙的影像：

| 設定 | 推論次數 |
| --- | --- |
| 單模型 | Mask R-CNN 1 次 + HRNet N 次 |
| + TTA | 各 ×4 |
| 五折集成 + TTA | 各 ×20（Mask R-CNN 的五折集成需另做跨模型實例配對，目前未實作） |
