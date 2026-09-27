# Li et al. 2024 論文中的疑點與錯誤

針對 `doc/12903_2024_Article_5039.pdf`（Li JM, Wang XE, et al. *BMC Oral Health* 2024;24:1232）
在複現其測量方法過程中發現的問題。

依嚴重程度分四類。每一項都附上原文出處與判斷依據，以便日後引用或在論文中
說明差異時查證。

> 撰寫論文時的建議措辭：這些多屬編輯或報告層面的疏漏，不必以「錯誤」直指，
> 可寫成「原文對此的描述不一致，本研究採用……」。真正需要主動揭露的是
> A 類——那些會影響方法能否複現的部分。

---

## A. 影響方法複現的問題

### A1. 長軸的定義無法客觀複現

原文對長軸的唯一描述（Materials and methods 步驟 2，Fig. 1 說明重複同一段）：

> "The tooth axis was determined by **considering** the crown area below the
> mesiodistal marginal ridge and the coronal two-thirds of the root area,
> ensuring that the axis evenly divides the tooth into mesial and distal halves."

這是描述人工目視的判斷過程，**沒有給出任何幾何構造**：

- 未定義切割線的方向（水平？垂直長軸？沿 CD？）
- 「邊緣嵴以下」是指哪個高度——A、B 兩點不等高，取高者、低者還是中點？
- 「牙根冠側 2/3」的起點未定義

**對本研究的影響**：本研究改用可操作的定義（切割線垂直於初始主軸、上界取 J），
必須在方法章節聲明這是自訂的操作型定義，而非照論文複現。

**但影響有限**：本研究已證明 CRR、ABLR、Max BLR、B-CRR 皆為沿長軸的長度比值，
而沿固定方向投影到不同方向的軸上時，長度差的比值不變（推導見
`scripts/measure.py` 開頭）。因此**長軸方向不影響這些指標的數值**，此處的
定義模糊不會傷害結果的可比性。

### A2. C、D 被描述成兩個不同的解剖構造

- 步驟 1：`"mesial and distal **enamel-dentin junctions** (C, D)"`（釉牙本質界，DEJ）
- Fig. 1 說明與步驟 3：`"mesial and distal points of the central **cemento-enamel junction** (CEJ)"`

DEJ 與 CEJ 是**不同位置的構造**。依上下文（CEJ 用於區分牙冠與牙根）應以 CEJ
為準，步驟 1 的 DEJ 疑為誤植或翻譯錯誤。

### A3. PRW 的數值相差十倍，定義無法確認

| 出處 | 數值 |
| --- | --- |
| Table 2 | PRW = 5.69 ± 5.49 mm |
| Results 內文（短／中／長根三組） | 0.56±0.23、0.57±0.63、0.58±0.55 mm |
| 分類閾值（引 Liu et al. 2022） | 0.37 |

Table 2 與內文相差約十倍，而閾值 0.37 只與後者的量級相符。無從判斷何者為準，
也就無法確認複現是否正確。

**本研究的處置**：不實作 PRW。理由除上述定義不明外，論文自己的分析中
PRW 與牙根寬度類型**在所有檢定中皆不顯著**（見 B4），且未進入任何迴歸模型。

---

## B. 數據與表格的錯誤

### B1. Table 3 的 Beta 欄遺漏負號（系統性）

Table 3 共 24 列迴歸係數，**全部印成正值**，但對照同列的 OR 可知其中約半數
應為負：邏輯迴歸中 OR = exp(Beta)，OR < 1 必然對應 Beta < 0。

抽樣驗證：

| 模型 | 變項 | 表列 Beta | 表列 OR | exp(Beta) | 判定 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 Non/I | APD | 0.426 | 0.653 | 1.531 | **應為 −0.426** |
| 1 Non/I | Root length | 0.300 | 1.350 | 1.350 | 正確 |
| 1 I/II&III | ABI | 0.193 | 0.824 | 1.213 | **應為 −0.193** |
| 3 Non/I | B-CRR | 4.056 | 0.017 | 57.7 | **應為 −4.056** |
| 3 I/II&III | B-CRR | 0.816 | 2.261 | 2.261 | 正確 |

（exp(−0.426)=0.653、exp(−0.193)=0.824、exp(−4.056)=0.017，皆與表列 OR 相符。）

**影響**：只看 Beta 欄會把效果的方向解讀反。OR 欄本身是正確的，引用時應以
OR 為準。

### B2. Discussion 引用了錯誤的 OR，且比較組別描述有誤

Discussion 原文：

> "Significant differences were also found in the APD (P<0.001, OR=1.581) and
> **B-CRR (P<0.05, OR=0.824)** between teeth with mobility **degrees II and III**."

兩處問題：

1. **OR 數值錯誤**：Table 3 模型 3 的 I/II&III 中，B-CRR 的 OR 是 **2.261**（P=0.000），
   不是 0.824。0.824 實際上是**模型 1** 中 **ABI** 的 OR。
2. **比較組別錯誤**：該模型比較的是「I 對 II&III 合併組」（論文自述因三度動搖
   樣本過少而合併），不是「II 與 III 之間」。

### B3. 牙位標示重複

Table 2 的 tooth position 列：

| 組別 | 表列 | 應為 |
| --- | --- | --- |
| 3 | 34 & 44 | 34 & 44 |
| 4 | **44 & 45** | **35 & 45** |

44 在兩組重複出現。依前文「236 mandibular first premolars、256 mandibular
second premolars」可知第 4 組應為下顎第二前臼齒 35 與 45。

### B4. PRW 的標準差大於平均值

Table 2：PRW 總體 5.69 ± **5.49**，動度 0 組 6.38 ± **11.32**。

標準差接近或大於平均值，代表分布極度偏斜或存在離群值。配合 A3 的十倍差異，
此欄數據品質存疑。

（附帶一提，這也解釋了為何 PRW 在所有檢定中都不顯著：F=0.528, P=0.663；
牙根寬度類型 χ²=2.317, P=0.509；三組間差異 P=0.830。）

---

## C. 標示與排版的不一致

| 項目 | 說明 |
| --- | --- |
| **C1** | Fig. 1 說明 (d) 寫 **PDW**，其餘各處為 **PRW** |
| **C2** | 倫理審查編號兩處不同：Methods 為 `PKUSSIIT02305`，Declarations 為 `PKUSSIRB-202495004` |
| **C3** | 動度標示不一致：Table 1 用 `Mobility = 0/1/2/3`，Table 2 用 `Non/I/II/III` |
| **C4** | ABLR 公式並列兩式，其一帶 `× 100%` 另一不帶，量綱不一致（百分比 vs 比值） |

---

## D. 方法學上的限制（非錯誤，但複現時須知）

### D1. ICC 只反映單一觀察者的重現性

原文：

> "Measurements were taken twice with an interval of over 3 months, with an
> ICC of 0.99, indicating extremely high reproducibility."

這是**同一位研究者前後兩次測量**的一致性（intra-observer），證明該研究者能
穩定地重複自己的判準；**但不足以證明這些特徵點在影像上是客觀明確的**——一個
人可以穩定地判在同一個「錯」的位置。論文未報告觀察者間一致性（inter-observer）。

**與本研究的關聯**：邊緣嵴（A、B）在根尖片上缺乏可辨識的形狀特徵，本研究
實際嘗試後亦確認輪廓上無轉折可循。ICC 0.99 不能作為「該特徵點可自動化定位」
的依據。

### D2. B-CRR 的動度門檻無法從任何表格推導

Discussion 稱：

> "the B-CRR thresholds for mobility grades I, II, and III were 1, 1.3, and 1.9"

但 Table 1 給的是 B-CRR 的 95% 信賴區間：動度 0 為 (0.68, 0.74)、I 為 (1.07, 1.16)、
II 為 (1.43, 1.59)、III 為 (2.13, 2.89)。1 / 1.3 / 1.9 這三個數字未出現於任何表格，
也未說明推導方式（看似為各區間下界的無條件捨去，但未明言）。

引用這三個門檻時應註明其來源僅為 Discussion 的敘述。

---

## 未發現問題的部分

以下經核算無誤，可安心引用：

- 樣本數加總：286+286+236+256 = 1064 ✓；動度分組 213+458+300+93 = 1064 ✓
- 各牙位百分比：26.88 + 26.88 + 22.18 + 24.06 = 100.00 ✓
- **B-CRR 的兩個公式代數等價**：內文的 (CRR+ABLR)/(1−ABLR) 與 Fig. 1(f) 的
  JS/KS 可互相推導。本研究已在 `scripts/measure.py` 中同時計算兩者作為自我
  檢查，五顆測試牙齒的差異皆小於 1e-9。
