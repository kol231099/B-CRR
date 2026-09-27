"""根尖片的影像增強方法集合，目標是讓牙齒邊界更明顯。

為什麼需要
----------
實測人工標註輪廓上的影像對比：根尖端只有牙冠端的 46%（13.1 vs 28.3）。這是物理
上的必然——牙冠琺瑯質對比軟組織與空氣，密度差很大；牙根牙骨質對比齒槽骨，密度
非常接近。模型在根尖段常常沒有邊界可循，只能猜。

每個方法的用意都不同，不是同一件事的變體：

    contrast_stretch   只把 2~98 百分位拉滿。最保守，不改變局部關係，純粹用掉
                       被浪費的動態範圍。
    hist_eq            全域直方圖等化。會把整張圖的分布拉平，但根尖片的骨頭佔
                       面積最大，等化後往往被骨頭主導，牙齒反而被壓縮。
    clahe / clahe_str  分格做等化並限制對比放大倍率。這是牙科 X 光的標準做法，
                       因為它針對**局部**——根尖那片均勻灰霧正是局部對比不足。
                       strong 版格子更大、放大更兇，代價是雜訊也被放大。
    gamma_dark         γ<1 提亮暗部。根尖片的牙根常落在中低灰階，提亮可以把那
                       段的差異拉開。
    unsharp            銳化。直接強化既有邊緣，但**無中生有不了**——邊界本來就
                       看不見的地方，銳化只會放大雜訊。
    clahe_unsharp      先拉局部對比再銳化。兩者互補：前者讓邊界浮現，後者讓它
                       變利。
    bilateral_clahe    先做保邊去雜訊再 CLAHE。根尖片顆粒雜訊明顯，直接 CLAHE
                       會連雜訊一起放大；先壓雜訊再增強，訊噪比較好。
    homomorphic        除以大尺度模糊背景，移除低頻的照明不均。121.jpg 那種
                       「均勻灰霧蓋住牙根」正是低頻成分，這個方法針對它。
    sobel              純梯度圖。不是要拿來訓練，是用來**看**邊界到底在不在。

用法（在其他腳本裡 import）：
    from enhance import METHODS
    out = METHODS["clahe"](gray)
"""

from __future__ import annotations

import cv2
import numpy as np


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def original(g: np.ndarray) -> np.ndarray:
    return g


def contrast_stretch(g: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    a, b = np.percentile(g, [lo, hi])
    return _u8((g.astype(np.float32) - a) * 255 / max(b - a, 1e-6))


def hist_eq(g: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(g)


def clahe(g: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def clahe_strong(g: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16)).apply(g)


def gamma_dark(g: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    return _u8(((g.astype(np.float32) / 255) ** gamma) * 255)


def unsharp(g: np.ndarray, sigma: float = 3.0, amount: float = 1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(g, (0, 0), sigma)
    return _u8(g.astype(np.float32) * (1 + amount) - blur.astype(np.float32) * amount)


def clahe_unsharp(g: np.ndarray) -> np.ndarray:
    return unsharp(clahe(g))


def bilateral_clahe(g: np.ndarray) -> np.ndarray:
    return clahe(cv2.bilateralFilter(g, 9, 50, 50))


def homomorphic(g: np.ndarray, sigma: float = 60.0) -> np.ndarray:
    """除以大尺度背景，移除低頻照明不均。均勻灰霧就是低頻成分。"""
    f = g.astype(np.float32) + 1.0
    bg = cv2.GaussianBlur(f, (0, 0), sigma)
    return _u8(f / bg * float(np.mean(bg)))


def sobel(g: np.ndarray) -> np.ndarray:
    m = np.sqrt(cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=5) ** 2
                + cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5) ** 2)
    return _u8(m / max(np.percentile(m, 99), 1e-6) * 255)


METHODS = {
    "original": original,
    "contrast_stretch": contrast_stretch,
    "hist_eq": hist_eq,
    "clahe": clahe,
    "clahe_strong": clahe_strong,
    "gamma_0.6": gamma_dark,
    "unsharp": unsharp,
    "clahe+unsharp": clahe_unsharp,
    "bilateral+clahe": bilateral_clahe,
    "homomorphic": homomorphic,
    "sobel": sobel,
}
