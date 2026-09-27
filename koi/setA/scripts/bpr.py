#!/usr/bin/env python3
"""BPR（Boundary Patch Refinement, Tang et al. CVPR 2021）的 patch 與精修網路。

想法：Dice 已經 0.97，錯的只有邊界那幾個像素。與其整顆牙重切，不如沿著粗糙
遮罩的邊界切出一連串小 patch，把每個 patch 放大後重新分割，再貼回去平均。
放大是關鍵——在 64px 的 patch 上做 128px 的推論，等於用兩倍解析度看邊界，
這正是次像素精度的來源。

精修網路的骨幹用 HRNetV2-W18，與原論文相同。
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

PATCH = 64      # 原尺寸下的 patch 邊長
STRIDE = 32     # 沿邊界取樣的間隔（50% 重疊）
NET = 128       # 送進網路前放大到這個尺寸


def boundary_points(mask: np.ndarray, stride: int = STRIDE) -> list[tuple[int, int]]:
    """沿遮罩邊界等間隔取 patch 中心。"""
    import cv2
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = []
    for c in cnts:
        c = c.reshape(-1, 2)
        if len(c) < stride:
            continue
        # 依弧長累積取樣，避免在轉角處擠成一團
        d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
        for t in np.arange(0, d[-1], stride):
            i = int(np.searchsorted(d, t))
            pts.append((int(c[i, 0]), int(c[i, 1])))
    return pts


def crop_patch(a: np.ndarray, cx: int, cy: int, size: int = PATCH) -> np.ndarray:
    """以 (cx, cy) 為中心取 size×size，超出邊界的地方補 0。"""
    h, w = a.shape
    half = size // 2
    out = np.zeros((size, size), a.dtype)
    x0, y0 = cx - half, cy - half
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + size), min(h, y0 + size)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
    return out


class PatchSet(Dataset):
    """從 bpr_dump 產生的 npz 取邊界 patch。

    輸入兩通道：灰階影像 + 粗糙遮罩。輸出是該 patch 的 GT。
    """

    def __init__(self, files: list, train: bool = True):
        import cv2
        self.items = []
        for f in files:
            d = np.load(f)
            img, coarse, gt = d["img"], d["coarse"].astype(bool), d["gt"].astype(bool)
            if img.shape != coarse.shape:
                img = cv2.resize(img, coarse.shape[::-1], interpolation=cv2.INTER_AREA)
            if not coarse.any():
                continue
            for cx, cy in boundary_points(coarse):
                self.items.append((img, coarse, gt, cx, cy))
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        import cv2
        img, coarse, gt, cx, cy = self.items[i]
        if self.train:   # 中心抖動，避免網路只學到「邊界必在正中央」
            cx += np.random.randint(-8, 9)
            cy += np.random.randint(-8, 9)
        pi = crop_patch(img, cx, cy)
        pc = crop_patch(coarse.astype(np.uint8), cx, cy)
        pg = crop_patch(gt.astype(np.uint8), cx, cy)
        if self.train and np.random.rand() < 0.5:
            pi, pc, pg = pi[:, ::-1], pc[:, ::-1], pg[:, ::-1]
        r = lambda a, m: cv2.resize(np.ascontiguousarray(a), (NET, NET), interpolation=m)
        x = np.stack([r(pi, cv2.INTER_LINEAR).astype(np.float32) / 255.0,
                      r(pc, cv2.INTER_NEAREST).astype(np.float32)])
        y = r(pg, cv2.INTER_NEAREST).astype(np.float32)[None]
        return torch.from_numpy(x), torch.from_numpy(y)


def build_refiner():
    """精修網路。

    原論文用 HRNetV2-W18，但實測在 CPU 上是 1.85s/iter，五折要 7.7 小時；
    resnet18 是 0.75s/iter，3.1 小時。BPR 的貢獻在於「沿邊界切 patch 放大重切」
    這個機制，骨幹是自由選擇，因此改用 resnet18。這個取捨要寫進論文的限制。

    附帶一提，efficientnet-b0（3.54s）與 mobilenet_v2（3.02s）雖然參數更少卻
    更慢——深度可分離卷積在 CPU 上是記憶體受限的。
    """
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name="resnet18", encoder_weights="imagenet",
                    in_channels=2, classes=1)


@torch.no_grad()
def refine(net, img: np.ndarray, coarse: np.ndarray, batch: int = 32) -> np.ndarray:
    """把粗糙遮罩沿邊界逐 patch 精修，重疊處取平均。回傳精修後的二值遮罩。"""
    import cv2
    pts = boundary_points(coarse)
    if not pts:
        return coarse
    acc = np.zeros(coarse.shape, np.float32)
    cnt = np.zeros(coarse.shape, np.float32)
    for s in range(0, len(pts), batch):
        chunk = pts[s:s + batch]
        xs = []
        for cx, cy in chunk:
            pi = crop_patch(img, cx, cy)
            pc = crop_patch(coarse.astype(np.uint8), cx, cy)
            xs.append(np.stack([
                cv2.resize(pi, (NET, NET), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0,
                cv2.resize(pc, (NET, NET), interpolation=cv2.INTER_NEAREST).astype(np.float32)]))
        o = torch.sigmoid(net(torch.from_numpy(np.stack(xs))))[:, 0].numpy()
        for (cx, cy), p in zip(chunk, o):
            p = cv2.resize(p, (PATCH, PATCH), interpolation=cv2.INTER_LINEAR)
            h, w = coarse.shape
            half = PATCH // 2
            x0, y0 = cx - half, cy - half
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(w, x0 + PATCH), min(h, y0 + PATCH)
            if sx1 > sx0 and sy1 > sy0:
                acc[sy0:sy1, sx0:sx1] += p[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
                cnt[sy0:sy1, sx0:sx1] += 1
    out = coarse.copy()
    hit = cnt > 0
    out[hit] = (acc[hit] / cnt[hit]) > 0.5      # 只改邊界帶，內部維持原判
    return out
