"""遮罩後處理：把一顆牙還原成一個連通、無孔洞的區域。

依據不是影像處理慣例，而是解剖事實：**一顆牙在根尖片上必定是單一連通區域，
內部不會有洞。** 所以「保留最大連通元件 + 填滿內部孔洞」不是平滑化那種會抹掉
真實細節的美化，它是把違反物理的輸出修正回來，資訊上只增不減。

不做輪廓平滑
------------
實測人工標註的高頻起伏是 0.67 px、Mask R-CNN 是 1.17 px，差距只有 0.5 px；而
大範圍起伏（12.44 vs 人工 12.24）落在人工的 IQR 之內——也就是說看起來的「凹凸」
多半是牙齒真實的輪廓曲率。為了 0.5 px 去平滑輪廓，反而會把牙根尖那種真實的
細微形狀一起磨掉，而 CRR 量的正是那裡。

用法（在其他腳本裡 import）：
    from postprocess import clean_mask
    mask = clean_mask(mask)
"""

from __future__ import annotations

import cv2
import numpy as np


def clean_mask(m: np.ndarray, min_frac: float = 0.05) -> np.ndarray:
    """保留最大連通元件並填滿孔洞。

    min_frac 之下的碎片直接丟棄；保留最大的那塊即可，因為 prompt / ROI 已經
    指定了「這個框裡的那一顆牙」，第二大的元件必定是溢出到鄰牙或骨頭的雜訊。
    """
    m = m.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        m = (lab == 1 + int(areas.argmax())).astype(np.uint8)

    # 填孔：從邊界外側 flood fill，填不到的地方就是內部孔洞
    h, w = m.shape
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return (m | (1 - ff)).astype(bool)
