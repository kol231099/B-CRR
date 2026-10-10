"""在 final/ 的 holdout 上評估融合法，輸出與 Table 1 同格式的逐顆 CSV。

放在 koi/setA/final/scripts/ 執行。Mask R-CNN 直接沿用 final 版 eval_holdout_all.py
的 maskrcnn() 載入器，holdout 的讀法、偵測門檻 0.35、pad 0.2、clean_mask、
metrics.match 也全部一致，所以與 Table 1 只差在新增的方法本身。

輸出四組（每組五折，各一個 CSV，檔名都以 hold5_FUS_ 開頭，不會覆蓋既有結果）：

    FUS_OBBjit        Mask R-CNN → OBB → 框擾動版 HRNet（單獨，無融合）
    FUS_fuse          clean_mask(0.5·P_HRNet + 0.5·P_MaskRCNN > 0.5)，無 TTA
    FUS_MaskRCNN_tta  Mask R-CNN + 翻轉 TTA（給 TTA 版融合當公平對照）
    FUS_fuse_tta      融合，兩個模型都做翻轉 TTA

融合權重固定 0.5，未在任何資料上調整。holdout 只做確認，不要在上面挑參數。

用法（在 koi/setA/final 底下）：
    python3 scripts/final_fuse_eval.py --device mps
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_holdout_all as E  # noqa: E402  final 版：沿用它的 Mask R-CNN 載入器
from eval_seg2_holdout import HOLD, gt_mask  # noqa: E402
from make_crops_obb import obb_of, warp_of  # noqa: E402
from metrics import FIELDS, match  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, ROOT  # noqa: E402
from train_seg2 import SIZE, build_seg2, split_tag  # noqa: E402
from tta import predict_tta  # noqa: E402

PAD, TAG = 0.2, "unet_tu-hrnet_w32"
VIEWS = [(False, False), (True, False), (False, True), (True, True)]
NAMES = ("FUS_OBBjit", "FUS_fuse", "FUS_MaskRCNN_tta", "FUS_fuse_tta")


def load_seg(ckdir, fold, device):
    arch, enc = split_tag(TAG)
    m = build_seg2(arch, enc, pretrained=False)
    m.load_state_dict(torch.load(ROOT / ckdir / "seg2" / TAG / f"fold{fold}.pt",
                                 map_location="cpu", weights_only=False)["model"])
    return m.eval().to(device)


@torch.no_grad()
def mr_probs(fold, gray, thr, use_tta):
    """Mask R-CNN 的機率圖（未二值化）與分數；模型由 final 的 eval_holdout_all 載入。"""
    model = E.maskrcnn(fold)
    if use_tta:
        prob, _, sc = predict_tta(model, gray, thr)
    else:
        t = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1)
        out = model([t])[0]
        keep = out["scores"].numpy() >= thr
        prob, sc = out["masks"].numpy()[keep, 0], out["scores"].numpy()[keep]
    return np.asarray(prob, np.float32).reshape(-1, *gray.shape), np.asarray(sc)


@torch.no_grad()
def hr_prob(seg, gray, box, use_tta, device):
    """斜框裁切 → HRNet（可選翻轉 TTA）→ 轉回原圖的機率圖。前處理同 eval_seg2_holdout.predict。"""
    h, w = gray.shape
    M, cw, ch = warp_of(*box, PAD)
    crop = cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR)
    base = cv2.resize(crop, SIZE[::-1], interpolation=cv2.INTER_AREA)
    views = VIEWS if use_tta else VIEWS[:1]
    xs = []
    for fh, fv in views:
        v = base[:, ::-1] if fh else base
        xs.append(np.ascontiguousarray(v[::-1] if fv else v))
    x = torch.from_numpy(np.stack(xs)).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
    out = torch.sigmoid(seg(x.to(device)))[:, 0].cpu().numpy()
    acc = []
    for o, (fh, fv) in zip(out, views):
        o = o[::-1] if fv else o
        acc.append(np.ascontiguousarray(o[:, ::-1] if fh else o))
    small = cv2.resize(np.mean(acc, 0), (cw, ch), interpolation=cv2.INTER_LINEAR)
    return cv2.warpAffine(small, cv2.invertAffineTransform(M), (w, h), flags=cv2.INTER_LINEAR)


def predict_all(fold, seg, gray, thr, device):
    """一次算完四組，回傳 {名稱: (遮罩 [N,H,W], 分數)}。"""
    hw = gray.shape
    out = {}
    for use_tta in (False, True):
        prob, sc = mr_probs(fold, gray, thr, use_tta)
        m_bin, e_bin, f_bin, keep = [], [], [], []
        for i, p in enumerate(prob):
            cm = np.asarray(clean_mask(p > 0.5), bool)
            m_bin.append(cm)
            if not cm.any():
                continue
            pe = hr_prob(seg, gray, obb_of(cm.astype(np.uint8)), use_tta, device)
            e_bin.append(np.asarray(clean_mask(pe > 0.5), bool))
            f_bin.append(np.asarray(clean_mask(0.5 * pe + 0.5 * p > 0.5), bool))
            keep.append(i)

        def arr(x):
            return np.array(x, bool).reshape(-1, *hw)
        if use_tta:
            out["FUS_MaskRCNN_tta"] = (arr(m_bin), sc)
            out["FUS_fuse_tta"] = (arr(f_bin), sc[keep])
        else:
            out["FUS_OBBjit"] = (arr(e_bin), sc[keep])
            out["FUS_fuse"] = (arr(f_bin), sc[keep])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", default="checkpoints_obb_jit")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--device", default="cpu", help="HRNet 用；Mask R-CNN 固定 CPU")
    args = ap.parse_args()

    coco = json.loads((ANN / "holdout.json").read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    per: dict[int, list] = {}
    for a in coco["annotations"]:
        if not a.get("iscrowd"):
            per.setdefault(a["image_id"], []).append(a)
    print(f"holdout（{ANN / 'holdout.json'}）：{len(per)} 張、{sum(map(len, per.values()))} 顆標註牙"
          f"　HRNet 在 {args.device}", flush=True)

    EVAL = ROOT / "eval"
    EVAL.mkdir(parents=True, exist_ok=True)
    t0, done, total = time.time(), 0, 5 * len(per)
    for fold in range(5):
        seg = load_seg(args.ckpt_dir, fold, args.device)
        rows = {n: [] for n in NAMES}
        for iid, anns in sorted(per.items()):
            im = imgs[iid]
            gray = cv2.imread(str(HOLD / im["file_name"]), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"  ⚠ 讀不到 {HOLD / im['file_name']}")
                continue
            h, w = gray.shape
            gt = np.stack([gt_mask(a, h, w) for a in anns])
            for name, (pred, sc) in predict_all(fold, seg, gray, args.thr, args.device).items():
                r = match(pred, sc, gt, im["file_name"])[0]
                for x in r:
                    x["gt_idx"] = f"{x['gt_idx']}"
                rows[name] += r
            done += 1
            el = time.time() - t0
            print(f"  fold {fold}　{done}/{total}　{im['file_name']}　已花 {el:.0f}s　"
                  f"預估剩 {el / done * (total - done):.0f}s", flush=True)
        for name, rs in rows.items():
            with (EVAL / f"hold5_{name}_fold{fold}.csv").open("w", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(fh, fieldnames=FIELDS)
                wr.writeheader()
                wr.writerows(rs)
    print(f"\n完成 → {EVAL}/hold5_FUS_*_fold{{0..4}}.csv")
    print("下一步：python3 scripts/final_table.py --ref <Table 1 的五個名稱>")


if __name__ == "__main__":
    main()
