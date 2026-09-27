#!/usr/bin/env python3
"""產生論文用的三張表。

表 1  主要比較：Mask R-CNN + 格子內各組合，10 指標，含 TTA，保留測試集
      消融用的 unetvanilla / unetscratch 不列入——它們沒有預訓練，與其餘
      模型條件不同，混在一起比大小會誤導。
表 2  消融階梯：每列只動一個因素，相鄰兩列的差就是該因素的貢獻

用法
    py scripts/make_tables.py
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "eval"
# 原始 U-Net 列入主表當基準；unetscratch 只在消融表出現（它不是一個會被推薦
# 使用的配置，只是拆解預訓練貢獻的中間步驟）
ABL = ("unetscratch",)
KEYS = [("Dice", "dice", "%.4f"), ("IoU", "iou", "%.4f"), ("Sens", "sens", "%.4f"),
        ("Prec", "prec", "%.4f"), ("B-IoU", "biou", "%.4f"), ("HD95", "hd95", "%.1f"),
        ("ASSD", "assd", "%.2f"), ("NSD@3", "nsd3", "%.4f"), ("NSD@5", "nsd5", "%.4f"),
        ("RVD", "rvd", "%+.4f")]
NICE = {"unet": "U-Net", "unetpp": "U-Net++", "deeplabv3p": "DeepLabv3+", "fpn": "FPN",
        "deeplabv3phr": "DeepLabv3+†",   # † ASPP 在 stride 32 運作，見頁尾說明
        "unetvanilla": "原始 U-Net", "unetscratch": "U-Net"}
ENC = {"tu-hrnet_w32": "HRNet-w32", "tu-hrnet_w18": "HRNet-w18", "resnet34": "resnet34",
       "resnet50": "resnet50", "resnet101": "resnet101",
       "efficientnet-b0": "efficientnet-b0", "mit_b0": "mit_b0", "none": "普通卷積"}


def load(p):
    return [x for x in csv.DictReader(open(p)) if x["kind"] == "TP"]


def split(tag: str) -> tuple[str, str]:
    for a in sorted(NICE, key=len, reverse=True):
        if tag.startswith(a + "_"):
            return a, tag[len(a) + 1:]
    return tag, ""


def rows_holdout() -> list[tuple[str, str, str, list]]:
    out = []
    for p in sorted(glob.glob(str(E / "holdout_seg2_*_tta.csv"))):
        tag = os.path.basename(p)[len("holdout_seg2_"):-len("_tta.csv")]
        a, e = split(tag)
        out.append((tag, a, e, load(p)))
    return out


def table1() -> None:
    base = load(E / "holdout_tta.csv")
    rs = [("Mask R-CNN（單階段）", "", base)]
    for tag, a, e, r in rows_holdout():
        if a in ABL:
            continue
        rs.append(("原始 U-Net（2015）" if a == "unetvanilla"
                   else f"{NICE[a]} × {ENC.get(e, e)}", tag, r))
    rs.sort(key=lambda x: -np.median([float(v["biou"]) for v in x[2]]))

    print(f"\n{'=' * 126}")
    print(f"表 1　保留測試集（18 張 / 26 顆牙）　含 TTA　指標皆於原圖座標　依 Boundary IoU 排序")
    print("=" * 126)
    print(f"{'方法':<30}{'n':>4}" + "".join(f"{k:>9}" for k, _, _ in KEYS))
    print("-" * 126)
    vals = {k: [np.median([float(x[k]) for x in r]) for _, _, r in rs] for _, k, _ in KEYS}
    LOWER = {"hd95", "assd"}          # 越小越好
    best = {}
    for _, k, _ in KEYS:
        v = vals[k]
        if k == "rvd":                # RVD 越接近 0 越好
            best[k] = int(np.argmin(np.abs(v)))
        elif k in LOWER:
            best[k] = int(np.argmin(v))
        else:
            best[k] = int(np.argmax(v))
    for i, (name, _, r) in enumerate(rs):
        cells = ""
        for _, k, f in KEYS:
            s = f % vals[k][i]
            cells += f"{'*' + s if best[k] == i else s:>9}"
        print(f"{name:<30}{len(r):>4}{cells}")
    print("\n  * = 該指標最佳（HD95/ASSD 取最小，RVD 取絕對值最小）")

    bm = {(x["image"], x["gt_idx"]): x for x in base}
    print(f"\n  各方法 vs Mask R-CNN　配對 Wilcoxon（Holm 校正，family = {len(rs) - 1}）")
    for key, lab in (("biou", "Boundary IoU"), ("hd95", "HD95")):
        res = []
        for name, tag, r in rs:
            if not tag:
                continue
            pm = {(x["image"], x["gt_idx"]): x for x in r}
            c = sorted(set(bm) & set(pm))
            a = np.array([float(pm[i][key]) for i in c])
            b = np.array([float(bm[i][key]) for i in c])
            if len(c) < 5 or np.allclose(a, b):
                continue
            res.append((name, len(c), float(np.median(a - b)), float(wilcoxon(a, b).pvalue)))
        res.sort(key=lambda x: x[3])
        m, prev = len(res), 0.0
        print(f"    【{lab}】")
        for i, (n, cn, md, p) in enumerate(res):
            hp = max(prev, min(1.0, p * (m - i)))
            prev = hp
            print(f"      {n:<30} n={cn:3d}　中位差 {md:+8.4f}　p={p:.4f}　"
                  f"Holm={hp:.4f}　{'顯著' if hp < 0.05 else '不顯著'}")


def table2() -> None:
    d = {t: r for t, a, e, r in rows_holdout()}
    ladder = [("原始 U-Net（2015）", "unetvanilla_none", "普通卷積", "✗", ""),
              ("U-Net × resnet34", "unetscratch_resnet34", "resnet34", "✗", "換 ResNet 殘差區塊"),
              ("U-Net × resnet34", "unet_resnet34", "resnet34", "✓", "載入 ImageNet 權重"),
              ("U-Net × HRNet-w32", "unet_tu-hrnet_w32", "HRNet-w32", "✓", "換不降採樣骨幹")]
    print(f"\n{'=' * 126}\n表 2　消融階梯　每列只動一個因素，相鄰兩列的差即該因素的貢獻\n{'=' * 126}")
    print(f"{'條件':<24}{'編碼器':<16}{'預訓練':<8}{'Dice':>9}{'B-IoU':>9}{'HD95':>8}"
          f"{'ΔB-IoU':>10}   本列改動")
    print("-" * 126)
    prev = None
    for name, tag, enc, pre, change in ladder:
        r = d.get(tag)
        if r is None:
            print(f"{name:<24}{enc:<16}{pre:<8}{'（尚未完成）':>28}   {change}")
            prev = None
            continue
        dice = np.median([float(x["dice"]) for x in r])
        b = np.median([float(x["biou"]) for x in r])
        h = np.median([float(x["hd95"]) for x in r])
        delta = f"{b - prev:+.4f}" if prev is not None else "—"
        print(f"{name:<24}{enc:<16}{pre:<8}{dice:>9.4f}{b:>9.4f}{h:>8.1f}{delta:>10}   {change}")
        prev = b


if __name__ == "__main__":
    if not (E / "holdout_tta.csv").exists():
        sys.exit("找不到 holdout_tta.csv")
    table1()
    table2()
