"""在 final/ 的切分上，用框擾動訓練 OBB 第二階段。訓練流程與 Table 1 的 ④ 逐行相同。

放在 koi/setA/final/scripts/ 執行。**不自己寫訓練迴圈**：直接呼叫 final 版
train_seg2.main()，只換掉資料集——所以模型、優化器、學習率、epoch、增強、
驗證與存檔規則全部是 final 原本那一套，跟 ④ 的第二階段完全一樣。
與 ④ 唯一的差別是送進去的 crop 來源：

    ④　　　　 讀 crops_obb/ 裡預先裁好的圖（GT 遮罩的 minAreaRect + pad 0.2）
    本腳本　　 每次取樣都從原圖現場裁切，GT 斜框加隨機擾動（旋轉、平移、縮放）

做法：繼承 final 的 CropDataset，__getitem__ 先算出擾動後的 crop，再把它交給
父類別原本的 __getitem__——父類別讀圖時拿到的是這張擾動 crop，之後的縮放、
翻轉、gamma 增強都走 final 的原程式。

為什麼要擾動：原本的訓練框是 GT 遮罩的外接矩形加固定 pad，牙齒四個極點永遠落在
crop 的固定位置，網路因此學會照著框邊畫（實測邊界跟隨斜率 0.2–0.5）。擾動後
斜率降到 0.01–0.05，網路改為看影像找邊界。

權重寫到 final/checkpoints_obb_jit/，不覆蓋 checkpoints_obb/。

用法（在 koi/setA/final 底下；--epochs 等參數必須與訓練 ④ 時相同）：
    python3 scripts/final_jit_train.py --arch unet --encoder tu-hrnet_w32 --fold 0 --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_seg2  # noqa: E402  final 版
from make_crops_obb import warp_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ANN, IMAGES, CROPS = ROOT / "annotations", ROOT / "images", ROOT / "crops_obb"
PAD = 0.2
JIT = {"ang": 5.0, "shift": 0.05, "scale": 0.08}
_ORIG_DS = train_seg2.CropDataset


class _FeedImread:
    """假的 cv2：前兩次 imread 依序回傳指定的影像與遮罩，其餘屬性全部轉給真的 cv2。"""

    def __init__(self, real, img, msk):
        self._real, self._queue = real, [img, msk]

    def __getattr__(self, k):
        return getattr(self._real, k)

    def imread(self, *a, **k):
        if not self._queue:
            raise RuntimeError("CropDataset 讀了超過兩張圖，與預期不符，請回報")
        return self._queue.pop(0)


class JitterCropDataset(_ORIG_DS):
    def __init__(self, ids, train):
        super().__init__(ids, train)
        self._rows = {r["crop_id"]: r for r in
                      csv.DictReader((CROPS / "manifest.csv").open(encoding="utf-8"))}
        self._segs = {a["id"]: a["segmentation"] for a in
                      json.loads((ANN / "instances_all.json").read_text(encoding="utf-8"))["annotations"]}
        self._imgs: dict[str, np.ndarray] = {}

    def _gray(self, name):
        if name not in self._imgs:
            g = cv2.imread(str(IMAGES / name), cv2.IMREAD_GRAYSCALE)
            if g is None:
                raise FileNotFoundError(IMAGES / name)
            self._imgs[name] = g
        return self._imgs[name]

    def _jitter_crop(self, i):
        r = self._rows[self.ids[i]]
        gray = self._gray(r["image"])
        full = np.zeros(gray.shape, np.uint8)
        for poly in self._segs[int(r["ann_id"])]:
            cv2.fillPoly(full, [np.array(poly, np.int32).reshape(-1, 2)], 255)
        # 驗證集固定種子：每次驗證看到同一組擾動，val Dice 才可比
        rng = np.random if self.train else np.random.RandomState(i)
        a, s, z = JIT["ang"], JIT["shift"], JIT["scale"]
        cx, cy, rw, rh, ang = (float(r[k]) for k in ("cx", "cy", "rw", "rh", "ang"))
        M0, _, _ = warp_of(cx, cy, rw, rh, ang, PAD)
        ux, uy = M0[0, :2], M0[1, :2]
        c = np.array([cx, cy]) + ux * rng.uniform(-s, s) * rw + uy * rng.uniform(-s, s) * rh
        box = (c[0], c[1], rw * rng.uniform(1 - z, 1 + z), rh * rng.uniform(1 - z, 1 + z),
               ang + rng.uniform(-a, a))
        M, cw, ch = warp_of(*box, PAD)
        return (cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR),
                cv2.warpAffine(full, M, (cw, ch), flags=cv2.INTER_NEAREST))

    def __getitem__(self, i):
        img, msk = self._jitter_crop(i)
        real = train_seg2.cv2
        train_seg2.cv2 = _FeedImread(real, img, msk)
        try:
            return super().__getitem__(i)     # final 原本的縮放與增強
        finally:
            train_seg2.cv2 = real


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--jit-ang", type=float, default=JIT["ang"])
    ap.add_argument("--jit-shift", type=float, default=JIT["shift"])
    ap.add_argument("--jit-scale", type=float, default=JIT["scale"])
    own, rest = ap.parse_known_args()
    JIT.update(ang=own.jit_ang, shift=own.jit_shift, scale=own.jit_scale)
    train_seg2.CROPS = CROPS
    train_seg2.CKPT = ROOT / "checkpoints_obb_jit"
    train_seg2.CropDataset = JitterCropDataset
    print(f"框擾動：旋轉±{JIT['ang']}° 平移±{JIT['shift']:.0%} 縮放±{JIT['scale']:.0%}"
          f"　crop 來源 {CROPS}　權重 → {train_seg2.CKPT}", flush=True)
    sys.argv = [sys.argv[0]] + rest
    train_seg2.main()
