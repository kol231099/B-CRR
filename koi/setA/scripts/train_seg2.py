"""第二階段語意分割：在單顆牙的裁切上比較各種編碼器／解碼器組合。

為什麼第二階段可以用語意分割
----------------------------
整張根尖片上不行——實測 54% 的多牙影像中，相鄰牙齒的遮罩會沾黏成同一個連通
元件，無法逐顆量測。但裁切之後一張影像只有一顆主體牙，「哪些像素屬於這顆牙」
有唯一答案，語意分割就適用了。裁切這個動作把實例問題轉成了語意問題。

難點在於 crop 帶有 20% padding，94% 的 crop 裡仍看得到鄰牙（鄰牙面積中位是主體
的 0.33 倍）。所以模型學的不是「找出牙齒像素」，而是「找出中央那一顆」。

候選模型的依據
--------------
依牙齒分割文獻挑選，不是依實作方便：

    U-Net × resnet34         所有相關論文的共同基準
    DeepLabv3+ × resnet101   Leite 等人於全景片報告 IoU 0.936、F1 0.966
    U-Net++ × resnet34       牙齒分割實測 IoU 0.8619、Dice 0.9258
    U-Net × efficientnet-b0  骨幹比較研究中精度／計算成本的最佳點
    U-Net × mit_b0           SegFormer 編碼器，代表 Transformer 路線

fold 切分與第一階段完全相同（以影像為單位），因此同一顆牙在兩個階段都落在
相同的驗證折，兩階段的結果可以逐顆配對比較。

用法：
    py scripts/train_seg2.py --arch unet --encoder resnet34 --fold 0
    py scripts/train_seg2.py --arch unet --encoder resnet34 --fold -1   # 全量
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
ANN, CROPS, CKPT = ROOT / "annotations", ROOT / "crops", ROOT / "checkpoints"
SIZE = (512, 256)  # (高, 寬)——牙齒又高又窄，維持 2:1 比原尺寸正方形化更省算力

ARCHS = {"unet": "Unet", "unetpp": "UnetPlusPlus",
         "deeplabv3p": "DeepLabV3Plus", "fpn": "FPN",
         "unetscratch": "Unet",    # ResNet 編碼器但隨機初始化，分離預訓練的貢獻
         "unetvanilla": "Unet",    # 原始 2015 U-Net，見 VanillaUNet
         "deeplabv3phr": "DeepLabV3Plus"}   # DeepLabv3+ 接 HRNet，見 build_deeplab_hrnet


class VanillaUNet(torch.nn.Module):
    """Ronneberger et al. 2015 的原始 U-Net。

    編碼器就是幾層普通卷積堆疊，沒有 ImageNet 預訓練權重可用。放進來是為了做
    消融：它與 U-Net×ResNet34（隨機初始化）的差距是 ResNet 架構的貢獻，後者
    與 ImageNet 版本的差距則是預訓練的貢獻。只跑其中一條，兩個效果會混在一起。
    """

    def __init__(self, in_channels: int = 3, classes: int = 1, base: int = 64):
        super().__init__()
        import torch.nn as nn

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        ch = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList()
        c = in_channels
        for o in ch[:-1]:
            self.enc.append(block(c, o))
            c = o
        self.bottom = block(ch[-2], ch[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2))
            self.dec.append(block(ch[i - 1] * 2, ch[i - 1]))
        self.head = nn.Conv2d(base, classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            x = u(x)
            x = d(torch.cat([s, x], dim=1))
        return self.head(x)


class DeepLabHRNet(torch.nn.Module):
    """DeepLabv3+ 的 decoder 接 HRNet encoder。

    smp 拒絕這個組合：DeepLabv3+ 只接受 encoder output stride 8 或 16，而 timm 的
    HRNet 寫死 `assert output_stride == 32`（不支援空洞卷積）。但 DeepLabv3+ 的核心
    是「ASPP + 低層特徵跳接」，encoder 的空洞卷積只是原論文用來把解析度保持在
    stride 16 的手段；HRNet 本來就以維持高解析度為設計原則，不需要那個手段。

    因此這裡手動組裝：encoder 照常在 stride 32 輸出，decoder 的上採樣倍率由 4 改為 8，
    才能接上 stride 4 的跳接。

    要注意的偏差：ASPP 在 stride 32 而非 16 的特徵圖上運作，感受野相對輸入放大一倍。
    這一點必須在論文寫明，此配置嚴格說是 DeepLabv3+ 的變體而非原版。
    """

    def __init__(self, encoder_name: str = "tu-hrnet_w32", pretrained: bool = True):
        super().__init__()
        import torch.nn as nn
        from segmentation_models_pytorch.base import SegmentationHead
        from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
        from segmentation_models_pytorch.encoders import get_encoder

        self.encoder = get_encoder(encoder_name, in_channels=3, depth=5,
                                   weights="imagenet" if pretrained else None)
        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels, encoder_depth=5,
            out_channels=256, atrous_rates=(12, 24, 36),   # 與其餘 DeepLabv3+ 一致
            output_stride=16, aspp_separable=True, aspp_dropout=0.5)
        self.decoder.up = nn.UpsamplingBilinear2d(scale_factor=8)
        self.segmentation_head = SegmentationHead(256, 1, kernel_size=1, upsampling=4)

    def forward(self, x):
        return self.segmentation_head(self.decoder(self.encoder(x)))


def build_seg2(arch: str, encoder: str, pretrained: bool = True):
    """所有第二階段模型的唯一建構入口，訓練與評估共用以免兩邊不一致。"""
    import segmentation_models_pytorch as smp
    if arch == "unetvanilla":
        return VanillaUNet(in_channels=3, classes=1)
    if arch == "deeplabv3phr":
        return DeepLabHRNet(encoder, pretrained)
    return getattr(smp, ARCHS[arch])(encoder_name=encoder,
                                     encoder_weights="imagenet" if pretrained else None,
                                     in_channels=3, classes=1)


def split_tag(tag: str) -> tuple[str, str]:
    """把 'unetscratch_resnet34' 拆成 ('unetscratch', 'resnet34')。

    先比長的，否則 'unetpp_resnet34' 會被 'unet_' 搶先匹配。
    """
    for a in sorted(ARCHS, key=len, reverse=True):
        if tag.startswith(a + "_"):
            return a, tag[len(a) + 1:]
    raise ValueError(tag)


class CropDataset(Dataset):
    """讀 crops/ 的單顆牙影像與遮罩。fold 依所屬**原始影像**切分，與第一階段一致。"""

    def __init__(self, ids: list[str], train: bool):
        self.ids, self.train = ids, train

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        cid = self.ids[i]
        img = cv2.imread(str(CROPS / "images" / f"{cid}.png"), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(CROPS / "masks" / f"{cid}.png"), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, SIZE[::-1], interpolation=cv2.INTER_AREA)
        # 遮罩用 NEAREST：插值會在邊界產生灰階值，二值化後邊界會漂移
        msk = cv2.resize(msk, SIZE[::-1], interpolation=cv2.INTER_NEAREST)

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


def fold_ids(fold: int) -> tuple[list[str], list[str]]:
    """把 crop 依其來源影像分到 train / val，沿用第一階段的 fold 切分。"""
    rows = list(csv.DictReader((CROPS / "manifest.csv").open(encoding="utf-8")))
    if fold < 0:
        return [r["crop_id"] for r in rows], []
    val_imgs = {i["file_name"] for i in
                json.loads((ANN / f"fold{fold}_val.json").read_text(encoding="utf-8"))["images"]}
    tr = [r["crop_id"] for r in rows if r["image"] not in val_imgs]
    va = [r["crop_id"] for r in rows if r["image"] in val_imgs]
    return tr, va


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="unet", choices=list(ARCHS))
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    import segmentation_models_pytorch as smp

    torch.manual_seed(0)
    np.random.seed(0)
    tag = f"{args.arch}_{args.encoder}"
    final = args.fold < 0

    tr_ids, va_ids = fold_ids(args.fold)
    # drop_last：最後一批若只剩 1 個樣本，DeepLabv3+ 的 ASPP 全域池化會產生
    # 1×1 的空間尺寸，BatchNorm 在訓練模式下無法計算統計量而拋出例外
    # （fold4 的訓練集為 125 張，125 % 4 == 1）。丟掉不完整的批次是標準做法，
    # 125 張中少用 1 張對結果無實質影響。
    dl_tr = DataLoader(CropDataset(tr_ids, True), batch_size=args.batch, shuffle=True,
                       num_workers=0, drop_last=len(tr_ids) % args.batch == 1)
    dl_va = None if final else DataLoader(CropDataset(va_ids, False), batch_size=1, num_workers=0)

    model = build_seg2(args.arch, args.encoder,
                       pretrained=args.arch not in ("unetscratch", "unetvanilla"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # Dice + BCE 是分割的標準組合：BCE 提供穩定的逐像素梯度，Dice 處理前景背景
    # 的面積不平衡（牙齒只佔 crop 的三成）
    dice_loss = smp.losses.DiceLoss(mode="binary")

    out_dir = CKPT / "seg2" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / ("final.pt" if final else f"fold{args.fold}.pt")
    print(f"{tag}　{'全量' if final else f'fold {args.fold}'}　"
          f"train {len(tr_ids)} / val {len(va_ids)}　"
          f"參數 {sum(p.numel() for p in params) / 1e6:.1f} M", flush=True)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in dl_tr:
            logit = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y) + dice_loss(logit, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss)
        sched.step()
        line = f"ep {ep:3d}/{args.epochs}　loss {total / len(dl_tr):.4f}　{time.time() - t0:.0f}s"

        if final:
            if ep == args.epochs:
                torch.save({"model": model.state_dict(), "arch": args.arch,
                            "encoder": args.encoder, "fold": -1}, ckpt)
                line += "　← 存檔"
        elif ep % 5 == 0 or ep == args.epochs:
            model.eval()
            ds = []
            with torch.no_grad():
                for x, y in dl_va:
                    p = torch.sigmoid(model(x)) > 0.5
                    g = y > 0.5
                    ds.append(float(2 * (p & g).sum() / (p.sum() + g.sum() + 1e-9)))
            m = float(np.mean(ds))
            line += f"　| val Dice {m:.4f}"
            if m > best:
                best = m
                torch.save({"model": model.state_dict(), "arch": args.arch,
                            "encoder": args.encoder, "fold": args.fold, "val_dice": m}, ckpt)
                line += "　← 存檔"
        print(line, flush=True)

    print(f"\n{tag} {'完成' if final else f'最佳 val Dice {best:.4f}'}　→ {ckpt}")


if __name__ == "__main__":
    main()
