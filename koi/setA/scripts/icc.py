"""組內相關係數（ICC），Shrout & Fleiss (1979) / McGraw & Wong (1996)。

兩種形式都回報，因為問的問題不同：

    ICC(3,1) consistency      兩次測量若只差一個固定偏移，仍算一致。
                              問「排序是否穩定」。
    ICC(2,1) absolute         固定偏移算不一致。問「數值是否可互換」。

冠根比是絕對數值，不是排序，因此 absolute agreement 才是該看的那個；
consistency 一併列出是為了顯示有無系統性偏移。

95% 信賴區間依 McGraw & Wong 的 F 分布法。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import f as fdist


def icc(data: np.ndarray, form: str = "absolute", alpha: float = 0.05) -> dict:
    """data: (n 個受試者, k 次測量)。回傳 ICC 與 95% CI、SEM。"""
    n, k = data.shape
    gm = data.mean()
    MSR = k * ((data.mean(axis=1) - gm) ** 2).sum() / (n - 1)          # 受試者間
    MSC = n * ((data.mean(axis=0) - gm) ** 2).sum() / (k - 1)          # 測量次間
    SSE = ((data - data.mean(axis=1, keepdims=True)
            - data.mean(axis=0, keepdims=True) + gm) ** 2).sum()
    MSE = SSE / ((n - 1) * (k - 1))

    if form == "consistency":
        val = (MSR - MSE) / (MSR + (k - 1) * MSE)
        F = MSR / MSE
        df2 = (n - 1) * (k - 1)
    else:                                                               # absolute
        val = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
        F = MSR / MSE
        df2 = (n - 1) * (k - 1)

    fl = F / fdist.ppf(1 - alpha / 2, n - 1, df2)
    fu = F * fdist.ppf(1 - alpha / 2, df2, n - 1)
    lo = (fl - 1) / (fl + k - 1)
    hi = (fu - 1) / (fu + k - 1)
    return {"icc": float(val), "lo": float(lo), "hi": float(hi),
            "sem": float(np.sqrt(MSE)), "MSE": float(MSE), "n": n}


def label(v: float) -> str:
    """Koo & Li (2016) 的慣用分級。"""
    return ("差" if v < 0.5 else "中等" if v < 0.75 else "良好" if v < 0.9 else "優異")
