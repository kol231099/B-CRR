"""論文方法（「黃色版」）的參考實作：一張根尖片 → 每顆牙的分割遮罩。

規格說明見 doc/paper_method_spec.md。網站／量測系統應直接呼叫 segment()，或逐行與它對照；
兩邊的遮罩必須逐像素相同。

    Mask R-CNN（門檻 0.35）→ 每顆牙的軟遮罩 P_MaskRCNN
      → clean_mask(P_MaskRCNN > 0.5) → 最小外接斜框（OBB），四邊外擴 20%，轉正裁切
      → 縮放 512×256 → U-Net-HRNet-w32（框擾動訓練的權重）→ sigmoid → 轉回原圖得 P_HRNet
      → mask = clean_mask(0.5·P_HRNet + 0.5·P_MaskRCNN > 0.5)

用法：
    # 單張影像（網站對照用）：輸出遮罩疊圖與每顆牙的面積，供兩邊比對
    python3 scripts/paper_pipeline.py --image X.jpg --maskrcnn M.pt --hrnet H0.pt [H1.pt ...] --out out.png

    # 一致性檢查（在 koi/setA/final 底下）：與論文評估用的 final_fuse_eval.py 逐像素比對
    python3 scripts/paper_pipeline.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_crops_obb import obb_of, warp_of  # noqa: E402
from postprocess import clean_mask  # noqa: E402

DET_THR = 0.35          # Mask R-CNN 偵測分數門檻
PAD = 0.2               # 斜框四邊外擴比例
SIZE = (512, 256)       # HRNet 輸入（高, 寬）
FUSE_W = 0.5            # 融合權重：0.5·P_HRNet + 0.5·P_MaskRCNN
MASK_THR = 0.5          # 所有機率圖一律以 0.5 二值化
TAG = "unet_tu-hrnet_w32"


# ───────────────────────── 載入 ─────────────────────────
def load_maskrcnn(path, device="cpu"):
    from train_maskrcnn import build_model
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(False, ck.get("mask_res", 28))
    m.load_state_dict(ck["model"])
    return m.eval().to(device)


def load_hrnets(paths, device="cpu"):
    """一個或多個 HRNet 權重；多個時機率取平均（集成）。"""
    from train_seg2 import build_seg2, split_tag
    arch, enc = split_tag(TAG)
    out = []
    for p in paths:
        m = build_seg2(arch, enc, pretrained=False)
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"])
        out.append(m.eval().to(device))
    return out


# ───────────────────────── 推論 ─────────────────────────
@torch.no_grad()
def maskrcnn_probs(model, gray, device="cpu"):
    """回傳 (軟遮罩 [N,H,W] float32，分數 [N])；只留分數 ≥ DET_THR 者。不做 TTA。"""
    t = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1).to(device)
    out = model([t])[0]
    sc = out["scores"].cpu().numpy()
    keep = sc >= DET_THR
    prob = out["masks"].cpu().numpy()[keep, 0]
    return np.asarray(prob, np.float32).reshape(-1, *gray.shape), sc[keep]


@torch.no_grad()
def hrnet_prob(models, gray, box, device="cpu"):
    """斜框裁切 → 512×256 → HRNet（多個模型取平均）→ 轉回原圖座標的機率圖（未二值化）。不做 TTA。"""
    h, w = gray.shape
    M, cw, ch = warp_of(*box, PAD)
    crop = cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR)
    base = cv2.resize(crop, SIZE[::-1], interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(np.ascontiguousarray(base)).float().div(255)
    t = t.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
    prob = np.mean([torch.sigmoid(m(t))[0, 0].cpu().numpy() for m in models], axis=0)
    small = cv2.resize(prob, (cw, ch), interpolation=cv2.INTER_LINEAR)
    return cv2.warpAffine(small, cv2.invertAffineTransform(M), (w, h), flags=cv2.INTER_LINEAR)


def segment(gray, maskrcnn, hrnets, device="cpu"):
    """論文方法：回傳 [{mask, score, box}]，依 Mask R-CNN 輸出順序（分數由高到低）。"""
    prob, sc = maskrcnn_probs(maskrcnn, gray, device)
    teeth = []
    for p, s in zip(prob, sc):
        cm = np.asarray(clean_mask(p > MASK_THR), bool)
        if not cm.any():
            continue
        box = obb_of(cm.astype(np.uint8))
        pe = hrnet_prob(hrnets, gray, box, device)
        mask = np.asarray(clean_mask(FUSE_W * pe + (1 - FUSE_W) * p > MASK_THR), bool)
        teeth.append({"mask": mask, "score": float(s), "box": box})
    return teeth


# ───────────────────────── 命令列 ─────────────────────────
def run_image(args):
    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        sys.exit(f"讀不到 {args.image}")
    teeth = segment(gray, load_maskrcnn(args.maskrcnn), load_hrnets(args.hrnet))
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    print(f"{args.image}：{len(teeth)} 顆牙（{gray.shape[1]}×{gray.shape[0]}）")
    for i, t in enumerate(teeth):
        ys, xs = np.nonzero(t["mask"])
        print(f"  #{i}  分數 {t['score']:.4f}  面積 {t['mask'].sum()} px  "
              f"外接框 x {xs.min()}–{xs.max()}  y {ys.min()}–{ys.max()}")
        cs = cv2.findContours(t["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cv2.drawContours(img, cs, -1, (0, 255, 255), 2)
    if args.out:
        cv2.imwrite(args.out, img)
        print(f"→ {args.out}")


def run_check():
    """與論文評估腳本 final_fuse_eval.py 的「融合」逐像素比對（holdout、五折、CPU）。"""
    import json
    import final_fuse_eval as F
    from eval_seg2_holdout import HOLD
    from train_maskrcnn import ANN

    coco = json.loads((ANN / "holdout.json").read_text(encoding="utf-8"))
    total = diff_px = n_teeth = 0
    for fold in range(5):
        seg = F.load_jit("checkpoints_obb_jit", fold)
        mr = F.E.maskrcnn(fold)
        for im in coco["images"]:
            gray = cv2.imread(str(HOLD / im["file_name"]), cv2.IMREAD_GRAYSCALE)
            mine = segment(gray, mr, [seg])
            ref, _ = F.run_variants(fold, seg, gray, DET_THR, False, {"F": "F"})["F"]
            if len(mine) != len(ref):
                print(f"  ✗ fold {fold} {im['file_name']}：牙數不同 {len(mine)} vs {len(ref)}")
                diff_px += 1
                continue
            for a, b in zip(mine, ref):
                diff_px += int((a["mask"] != b).sum())
                total += int(b.sum())
                n_teeth += 1
        print(f"  fold {fold} 完成", flush=True)
    print(f"\n比對 {n_teeth} 顆牙：不同的像素 {diff_px}（總牙齒像素 {total}）")
    print("✅ 完全一致" if diff_px == 0 else "❌ 不一致，請把上面的輸出貼回來")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="與 final_fuse_eval.py 逐像素比對（須在 final/ 底下）")
    ap.add_argument("--image")
    ap.add_argument("--maskrcnn")
    ap.add_argument("--hrnet", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.check:
        run_check()
    elif a.image and a.maskrcnn and a.hrnet:
        run_image(a)
    else:
        ap.print_help()
