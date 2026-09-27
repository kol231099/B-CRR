"""測試時增強（TTA）：把同一張影像的多個翻轉版本推論後平均。

動機
----
模型在邊界模糊處的判斷帶有隨機性——實測根尖段的邊界對比只有牙冠段的 46%，
CNR 低於 1，模型只能猜。對同一顆牙做多次「不同視角」的推論再平均，可以把這種
隨機猜測抵銷掉一部分。這不需要重新訓練，也不改變模型。

用的是訓練時就用過的翻轉（水平、垂直、兩者），所以推論看到的分布與訓練一致。

實例分割的 TTA 比語意分割麻煩
------------------------------
語意分割只要把機率圖翻回來平均即可。實例分割每次推論得到的是一組**實例**，
順序不保證一致，數量也可能不同，所以必須先把各次推論的實例配對起來。這裡以
未增強的那次為基準，其餘各次用遮罩 IoU 配到基準上，再平均**機率圖**（不是
二值遮罩）最後才二值化——先二值化再平均會丟掉模型的不確定性資訊。

只在基準中出現的實例維持原樣；只在翻轉版出現的實例不予採納，因為無從判斷它是
真的被漏掉還是翻轉造成的假陽性。

用法：
    from tta import predict_tta
    masks, boxes, scores = predict_tta(model, gray, conf=0.35)
"""

from __future__ import annotations

import numpy as np
import torch

# (水平翻轉, 垂直翻轉)
VIEWS = [(False, False), (True, False), (False, True), (True, True)]


def _flip(a: np.ndarray, h: bool, v: bool) -> np.ndarray:
    if h:
        a = a[..., ::-1]
    if v:
        a = a[..., ::-1, :]
    return np.ascontiguousarray(a)


@torch.no_grad()
def predict_tta(model, gray: np.ndarray, conf: float = 0.35):
    """回傳 (機率遮罩 [N,H,W] float, 框 [N,4], 分數 [N])，皆為 TTA 平均後的結果。"""
    ref_masks = ref_scores = ref_boxes = None
    acc: list[list[np.ndarray]] = []
    acc_score: list[list[float]] = []

    for h, v in VIEWS:
        img = _flip(gray, h, v)
        t = torch.from_numpy(img).float().div(255).unsqueeze(0).repeat(3, 1, 1)
        out = model([t])[0]
        keep = out["scores"].numpy() >= conf
        m = out["masks"].numpy()[keep, 0]            # 機率圖，尚未二值化
        s = out["scores"].numpy()[keep]
        b = out["boxes"].numpy()[keep]
        m = _flip(m, h, v)                            # 翻回原始方向
        if h:
            b = b[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1]) + np.array([gray.shape[1], 0, gray.shape[1], 0])
        if v:
            b = b[:, [0, 3, 2, 1]] * np.array([1, -1, 1, -1]) + np.array([0, gray.shape[0], 0, gray.shape[0]])

        if ref_masks is None:
            ref_masks, ref_scores, ref_boxes = m, s, b
            acc = [[x] for x in m]
            acc_score = [[float(y)] for y in s]
            continue

        # 以基準的實例為錨，用遮罩 IoU 配對
        rb = ref_masks > 0.5
        mb = m > 0.5
        used: set[int] = set()
        for i in range(len(rb)):
            best, best_iou = -1, 0.0
            for j in range(len(mb)):
                if j in used:
                    continue
                inter = np.logical_and(rb[i], mb[j]).sum()
                union = np.logical_or(rb[i], mb[j]).sum()
                iou = inter / union if union else 0.0
                if iou > best_iou:
                    best, best_iou = j, iou
            if best_iou >= 0.5:
                used.add(best)
                acc[i].append(m[best])
                acc_score[i].append(float(s[best]))

    masks = np.stack([np.mean(v, axis=0) for v in acc]) if acc else np.zeros((0, *gray.shape))
    scores = np.array([float(np.mean(v)) for v in acc_score]) if acc_score else np.zeros(0)
    return masks, ref_boxes, scores
