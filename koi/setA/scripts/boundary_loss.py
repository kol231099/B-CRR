"""邊界感知損失：讓遮罩頭的訓練目標對齊 HD95 這個評估指標。

問題
----
torchvision 的遮罩損失是逐像素 BCE——每個像素的權重相同，答錯輪廓轉折處的一個
像素，跟答錯牙齒正中央的一個像素，代價完全一樣。但本專案的主指標是 HD95，量的
是輪廓之間的**距離**。也就是在優化 A、卻用 B 評分。

實測的分層分析支持這個診斷：誤差集中在牙齒**兩端曲率大的轉折**（根尖 7.9 px、
牙冠 7.0 px，中段只有 3.3–4.0 px），而 BCE 完全不會特別在意那些位置。提高遮罩
解析度（28×28 → 56×56）對 HD95 沒有任何改善，也指向限制不在「能畫多細」，
而在「知不知道該注意哪裡」。

做法（Kervadec et al., 2019 的 boundary loss）
---------------------------------------------
對每個真實遮罩計算帶號距離場 φ：物體外為正、物體內為負，數值是到輪廓的距離。
損失取 φ 與預測前景機率的乘積平均：

    L_B = mean(φ ⊙ sigmoid(logits))

離輪廓越遠的地方，答錯的代價越大；正確的前景區域 φ 為負，會拉低損失。這個項
本身是距離的線性函數，因此直接懲罰「差多遠」而不只是「對或錯」。

單獨使用會不穩定（初期預測近乎隨機，梯度會把遮罩推向極端），所以與 BCE 混合，
並讓權重 α 隨 epoch 從 0 線性升到 alpha_max：

    L = (1 − α)·BCE + α·L_B

用法：由 train_maskrcnn.py 的 --mask-loss boundary 啟用，不需直接呼叫。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


class AlphaSchedule:
    """在訓練迴圈裡逐 epoch 更新，供損失函式讀取當前的 α。"""

    def __init__(self, alpha_max: float, epochs: int, warmup_frac: float = 0.25):
        self.alpha_max = alpha_max
        self.epochs = max(epochs, 1)
        self.warmup = warmup_frac
        self.value = 0.0

    def step(self, epoch: int) -> float:
        """epoch 由 1 起算。前 warmup_frac 的訓練期間 α = 0，之後線性升到 alpha_max。"""
        t = (epoch - 1) / self.epochs
        self.value = 0.0 if t < self.warmup else self.alpha_max * (t - self.warmup) / (1 - self.warmup)
        return self.value


def signed_distance(targets: torch.Tensor) -> torch.Tensor:
    """[N, M, M] 的 0/1 遮罩 → 帶號距離場，外正內負，除以 M 正規化。

    正規化是為了讓 α 的意義不隨遮罩解析度改變——28×28 與 56×56 的距離場尺度
    相差一倍，不除掉的話同一個 α 在兩者上的效果不同。
    """
    out = np.empty(targets.shape, np.float32)
    t = targets.detach().cpu().numpy()
    m = targets.shape[-1]
    for i in range(t.shape[0]):
        fg = t[i] > 0.5
        if not fg.any() or fg.all():
            out[i] = 0.0            # 退化情況不提供梯度，交給 BCE
            continue
        out[i] = (distance_transform_edt(~fg) - distance_transform_edt(fg)) / m
    return torch.from_numpy(out).to(targets.device)


def make_loss(schedule: AlphaSchedule):
    """回傳可取代 torchvision.models.detection.roi_heads.maskrcnn_loss 的函式。"""
    from torchvision.models.detection.roi_heads import project_masks_on_boxes

    def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
        size = mask_logits.shape[-1]
        labels = torch.cat([g[i] for g, i in zip(gt_labels, mask_matched_idxs)], dim=0)
        targets = torch.cat(
            [project_masks_on_boxes(m, p, i, size)
             for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)], dim=0
        )
        if targets.numel() == 0:
            return mask_logits.sum() * 0

        sel = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels]
        bce = F.binary_cross_entropy_with_logits(sel, targets)

        alpha = schedule.value
        if alpha <= 0:
            return bce
        with torch.no_grad():
            phi = signed_distance(targets)
        boundary = (phi * torch.sigmoid(sel)).mean()
        return (1 - alpha) * bce + alpha * boundary

    return maskrcnn_loss


def patch(schedule: AlphaSchedule) -> None:
    """就地替換 torchvision 的遮罩損失。

    RoIHeads.forward 以模組層級的名稱呼叫 maskrcnn_loss，因此替換該模組屬性即可
    生效，不需要繼承或改寫模型類別。
    """
    import torchvision.models.detection.roi_heads as rh

    rh.maskrcnn_loss = make_loss(schedule)
