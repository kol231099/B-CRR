"""在 final/ 的切分上，用框擾動訓練 OBB 第二階段（U-Net × HRNet-w32）。

放在 koi/setA/final/scripts/ 執行，所有路徑都以 final/ 為根，讀 final 自己的
annotations、images、crops_obb。原本的 train_seg2.py 不做任何修改，只借用它的
模型建構、fold 切分與增強設定；訓練超參與 Table 1 的第二階段完全相同
（AdamW lr 1e-4、wd 1e-4、cosine、BCE + Dice、batch 4、每 5 epoch 驗證存最佳）。

唯一的差別：每次取樣都從原圖現場裁切，並對 GT 斜框做隨機擾動（旋轉、沿兩軸平移、
兩邊各自縮放），讓牙齒邊緣不再固定落在 crop 的同一條線上。原本的訓練框是 GT 遮罩的
minAreaRect 加固定 pad，牙齒四個極點永遠在固定位置，網路因此學會照著框邊畫
（實測邊界跟隨斜率 0.2–0.5）；擾動後斜率降到 0.01–0.05。

驗證集也擾動，但每個樣本固定種子，存檔挑的是對框不敏感的權重。
權重寫到 final/checkpoints_obb_jit/，不覆蓋 checkpoints_obb/。

用法（在 koi/setA/final 底下）：
    python3 scripts/final_jit_train.py --fold 0 --device mps
    for k in 0 1 2 3 4; do python3 scripts/final_jit_train.py --fold $k --device mps; done
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_seg2  # noqa: E402
from make_crops_obb import warp_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ANN, IMAGES = ROOT / "annotations", ROOT / "images"
CROPS = ROOT / "crops_obb"
train_seg2.CROPS = CROPS          # fold_ids() 讀這裡的 manifest
PAD = 0.2


class JitterCropDataset(Dataset):
    """從原圖現場裁切，GT 斜框隨機擾動；翻轉與 gamma 增強與 train_seg2.CropDataset 相同。"""

    def __init__(self, ids, train, jit):
        self.train, self.jit = train, jit
        rows = {r["crop_id"]: r for r in csv.DictReader((CROPS / "manifest.csv").open(encoding="utf-8"))}
        polys = {a["id"]: a["segmentation"] for a in
                 json.loads((ANN / "instances_all.json").read_text(encoding="utf-8"))["annotations"]}
        self.items = [(rows[c], polys[int(rows[c]["ann_id"])]) for c in ids]
        self._imgs: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.items)

    def gray(self, name):
        if name not in self._imgs:
            g = cv2.imread(str(IMAGES / name), cv2.IMREAD_GRAYSCALE)
            if g is None:
                raise FileNotFoundError(IMAGES / name)
            self._imgs[name] = g
        return self._imgs[name]

    def __getitem__(self, i):
        r, segm = self.items[i]
        gray = self.gray(r["image"])
        full = np.zeros(gray.shape, np.uint8)
        for poly in segm:
            cv2.fillPoly(full, [np.array(poly, np.int32).reshape(-1, 2)], 255)

        rng = np.random if self.train else np.random.RandomState(i)
        a, s, z = self.jit
        cx, cy, rw, rh, ang = (float(r[k]) for k in ("cx", "cy", "rw", "rh", "ang"))
        M0, _, _ = warp_of(cx, cy, rw, rh, ang, PAD)
        ux, uy = M0[0, :2], M0[1, :2]
        c = np.array([cx, cy]) + ux * rng.uniform(-s, s) * rw + uy * rng.uniform(-s, s) * rh
        box = (c[0], c[1], rw * rng.uniform(1 - z, 1 + z), rh * rng.uniform(1 - z, 1 + z),
               ang + rng.uniform(-a, a))
        M, cw, ch = warp_of(*box, PAD)
        img = cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR)
        msk = cv2.warpAffine(full, M, (cw, ch), flags=cv2.INTER_NEAREST)

        size = train_seg2.SIZE
        img = cv2.resize(img, size[::-1], interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, size[::-1], interpolation=cv2.INTER_NEAREST)
        if self.train:
            if np.random.rand() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if np.random.rand() < 0.5:
                img, msk = img[::-1], msk[::-1]
            if np.random.rand() < 0.8:
                gamma, gain = np.random.uniform(0.7, 1.4), np.random.uniform(0.85, 1.15)
                img = np.clip(((img / 255.0) ** gamma) * gain * 255, 0, 255).astype(np.uint8)
        x = torch.from_numpy(np.ascontiguousarray(img)).float().div(255).unsqueeze(0).repeat(3, 1, 1)
        y = torch.from_numpy(np.ascontiguousarray(msk)).float().div(255).unsqueeze(0)
        return x, y


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--encoder", default="tu-hrnet_w32")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu", help="cpu / cuda / mps")
    ap.add_argument("--jit-ang", type=float, default=5.0)
    ap.add_argument("--jit-shift", type=float, default=0.05)
    ap.add_argument("--jit-scale", type=float, default=0.08)
    args = ap.parse_args()

    import segmentation_models_pytorch as smp

    torch.manual_seed(0)
    np.random.seed(0)
    tag = f"{args.arch}_{args.encoder}"
    jit = (args.jit_ang, args.jit_shift, args.jit_scale)
    tr_ids, va_ids = train_seg2.fold_ids(args.fold)
    dl_tr = DataLoader(JitterCropDataset(tr_ids, True, jit), batch_size=args.batch, shuffle=True,
                       num_workers=0, drop_last=len(tr_ids) % args.batch == 1)
    dl_va = DataLoader(JitterCropDataset(va_ids, False, jit), batch_size=1, num_workers=0)

    model = train_seg2.build_seg2(args.arch, args.encoder, pretrained=True).to(args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    dice_loss = smp.losses.DiceLoss(mode="binary")

    out_dir = ROOT / "checkpoints_obb_jit" / "seg2" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"fold{args.fold}.pt"
    print(f"{tag}　fold {args.fold}　train {len(tr_ids)} / val {len(va_ids)}　"
          f"擾動 旋轉±{jit[0]}° 平移±{jit[1]:.0%} 縮放±{jit[2]:.0%}　→ {ckpt}", flush=True)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in dl_tr:
            x, y = x.to(args.device), y.to(args.device)
            logit = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y) + dice_loss(logit, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss)
        sched.step()
        line = f"ep {ep:3d}/{args.epochs}　loss {total / len(dl_tr):.4f}　{time.time() - t0:.0f}s"
        if ep % 5 == 0 or ep == args.epochs:
            model.eval()
            ds = []
            with torch.no_grad():
                for x, y in dl_va:
                    x, y = x.to(args.device), y.to(args.device)
                    p, g = torch.sigmoid(model(x)) > 0.5, y > 0.5
                    ds.append(float(2 * (p & g).sum() / (p.sum() + g.sum() + 1e-9)))
            m = float(np.mean(ds))
            line += f"　| val Dice {m:.4f}"
            if m > best:
                best = m
                torch.save({"model": model.state_dict(), "arch": args.arch, "encoder": args.encoder,
                            "fold": args.fold, "val_dice": m, "jitter": jit}, ckpt)
                line += "　← 存檔"
        print(line, flush=True)
    print(f"\n最佳 val Dice {best:.4f}　→ {ckpt}")


if __name__ == "__main__":
    main()
