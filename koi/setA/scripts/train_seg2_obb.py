"""用 OBB crop 訓練第二階段，其餘與 train_seg2.py 完全相同。

只改兩個模組變數：資料改讀 crops_obb/，權重改寫到 checkpoints_obb/，
避免覆蓋既有的 HBB 版權重。訓練超參、增強、fold 切分一律沿用，
這樣 OBB vs HBB 的差異才只來自裁切方式。

--jitter：框擾動訓練
--------------------
diag_box_leak.py 的結論：crops_obb 的框是 GT 遮罩的 minAreaRect，牙齒必定碰到框的
四條邊（內縮 pad），網路因此學會「照著框邊畫」（邊界跟隨斜率 0.2–0.5）。換成
Mask R-CNN 的框時，它就在重畫 Mask R-CNN 的遮罩（兩者 Dice 0.981）。

開啟 --jitter 後改為從原圖現場裁切，每次取樣都對 GT 框做隨機擾動（旋轉、沿兩軸
平移、兩邊各自縮放），讓牙的邊緣不再落在 crop 裡固定的位置，網路只能看影像找
邊界。驗證集也擾動，但每個樣本固定種子，存檔挑的是「對框不敏感」的權重。
推論時的 pad 仍是 0.2，與 eval_e2e_obb.py / diag_box_leak.py 一致。
權重寫到 checkpoints_obb_jit/，不覆蓋原本的 checkpoints_obb/。

用法：
    py scripts/train_seg2_obb.py --arch unet --encoder tu-hrnet_w32 --fold 0
    py scripts/train_seg2_obb.py --arch unet --encoder tu-hrnet_w32 --fold 0 \\
        --epochs 20 --device mps --jitter
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
import train_seg2  # noqa: E402
from make_crops_obb import ANN, IMAGES, warp_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
train_seg2.CROPS = ROOT / "crops_obb"
train_seg2.CKPT = ROOT / "checkpoints_obb"
PAD = 0.2


class JitterCropDataset(train_seg2.CropDataset):
    """從原圖現場裁切，框依 JIT 隨機擾動。其餘增強（翻轉、gamma）沿用父類別。"""

    JIT = {"ang": 5.0, "shift": 0.05, "scale": 0.08}
    _imgs: dict[str, np.ndarray] = {}

    def __init__(self, ids: list[str], train: bool):
        super().__init__(ids, train)
        rows = {r["crop_id"]: r for r in
                csv.DictReader((train_seg2.CROPS / "manifest.csv").open(encoding="utf-8"))}
        polys = {a["id"]: a["segmentation"][0] for a in
                 json.loads((ANN / "instances_all.json").read_text(encoding="utf-8"))["annotations"]}
        self.items = [(rows[c], polys[int(rows[c]["ann_id"])]) for c in ids]

    def gray(self, name: str) -> np.ndarray:
        if name not in self._imgs:
            g = cv2.imread(str(IMAGES / name), cv2.IMREAD_GRAYSCALE)
            if g is None:
                raise FileNotFoundError(IMAGES / name)
            self._imgs[name] = g
        return self._imgs[name]

    def load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        r, poly = self.items[i]
        gray = self.gray(r["image"])
        full = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(full, [np.array(poly, np.int32).reshape(-1, 2)], 255)

        # 驗證集固定種子：每個 epoch 看到同一組擾動，val Dice 才可比
        rng = np.random if self.train else np.random.RandomState(i)
        j = self.JIT
        cx, cy, rw, rh, ang = (float(r[k]) for k in ("cx", "cy", "rw", "rh", "ang"))
        M0, _, _ = warp_of(cx, cy, rw, rh, ang, PAD)
        ux, uy = M0[0, :2], M0[1, :2]           # crop 的 x、y 軸在原圖的方向
        c = (np.array([cx, cy]) + ux * rng.uniform(-j["shift"], j["shift"]) * rw
             + uy * rng.uniform(-j["shift"], j["shift"]) * rh)
        box = (c[0], c[1], rw * rng.uniform(1 - j["scale"], 1 + j["scale"]),
               rh * rng.uniform(1 - j["scale"], 1 + j["scale"]),
               ang + rng.uniform(-j["ang"], j["ang"]))

        M, cw, ch = warp_of(*box, PAD)
        img = cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR)
        # 遮罩用 NEAREST：插值會在邊界產生灰階值，二值化後邊界會漂移
        msk = cv2.warpAffine(full, M, (cw, ch), flags=cv2.INTER_NEAREST)
        return img, msk


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--jitter", action="store_true")
    ap.add_argument("--jit-ang", type=float, default=JitterCropDataset.JIT["ang"],
                    help="旋轉擾動上限（度），均勻分布")
    ap.add_argument("--jit-shift", type=float, default=JitterCropDataset.JIT["shift"],
                    help="沿兩軸平移上限（佔該邊長比例）")
    ap.add_argument("--jit-scale", type=float, default=JitterCropDataset.JIT["scale"],
                    help="兩邊各自縮放上限（比例）")
    own, rest = ap.parse_known_args()
    if own.jitter:
        JitterCropDataset.JIT = {"ang": own.jit_ang, "shift": own.jit_shift,
                                 "scale": own.jit_scale}
        train_seg2.CropDataset = JitterCropDataset
        train_seg2.CKPT = ROOT / "checkpoints_obb_jit"
        print(f"框擾動訓練：{JitterCropDataset.JIT}　→ {train_seg2.CKPT}", flush=True)
    sys.argv = [sys.argv[0]] + rest
    train_seg2.main()
