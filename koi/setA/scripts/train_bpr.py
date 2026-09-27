#!/usr/bin/env python3
"""訓練 BPR 精修網路。每個 fold 一個，只用該 fold 的訓練影像。

用法
    py scripts/train_bpr.py --fold 0 --epochs 12
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpr import PatchSet, build_refiner  # noqa: E402
from train_maskrcnn import CKPT, ROOT  # noqa: E402

BPR = ROOT / "bpr"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="unetpp_resnet34_tta",
                    help="粗糙遮罩的來源模型；一個精修網路要套用到全部模型，故固定一個來源")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    import segmentation_models_pytorch as smp

    src = BPR / args.source / "train"
    files = [f for f in sorted(src.glob("*.npz")) if int(np.load(f)["fold"]) == args.fold]
    if not files:
        sys.exit(f"找不到 fold {args.fold} 的資料，請先跑 bpr_dump.py --split train")

    ds = PatchSet(files, train=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    drop_last=len(ds) % args.batch == 1)
    print(f"BPR fold {args.fold}　{len(files)} 顆牙 → {len(ds)} 個邊界 patch", flush=True)

    net = build_refiner()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    bce = torch.nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary")

    out = CKPT / "bpr"
    out.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        net.train()
        tot, t0 = 0.0, time.time()
        for x, y in dl:
            opt.zero_grad()
            o = net(x)
            loss = bce(o, y) + dice(o, y)
            loss.backward()
            opt.step()
            tot += float(loss)
        sched.step()
        print(f"  ep {ep:2d}/{args.epochs}　loss {tot / max(1, len(dl)):.4f}　"
              f"{time.time() - t0:.0f}s", flush=True)
    torch.save({"model": net.state_dict()}, out / f"fold{args.fold}.pt")
    print(f"→ {out / f'fold{args.fold}.pt'}", flush=True)


if __name__ == "__main__":
    main()
