"""兩階段 pipeline 的五折 OOF 端到端評估，座標系為原圖。

eval_e2e.py 只跑 holdout（n=26）且用 maskrcnn_final + 五模型集成 + TTA，
那是部署設定；要跟單階段 YOLO 比就必須換成同一批 fold、同樣單模型無 TTA，
否則比的是「集成」而不是「兩階段」。

每折：maskrcnn_fold{f}.pt 偵測 → 預測遮罩外接框 +20% 裁切 → seg2/fold{f}.pt
圈輪廓 → 貼回原圖。漏檢記 FN，多檢記 FP，與 eval_maskrcnn / eval_yolo 共用
metrics.match，因此三者可直接配對檢定。

用法：
    py scripts/eval_e2e_oof.py --fold 0
    py scripts/eval_e2e_oof.py --all
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_seg2 import build_seg2, split_tag  # noqa: E402
from eval_seg2_holdout import predict  # noqa: E402
from make_crops import crop_box  # noqa: E402
from metrics import FIELDS, match, summarize  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, CKPT, ROOT, ToothDataset, build_model, collate  # noqa: E402

EVAL = ROOT / "eval"


def load_fold_seg2(tag: str, fold: int):
    arch, enc = split_tag(tag)
    m = build_seg2(arch, enc, pretrained=False)
    ck = torch.load(CKPT / "seg2" / tag / f"fold{fold}.pt", map_location="cpu",
                    weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def run(fold: int, tag: str, score_thr: float) -> list[dict]:
    ck = torch.load(CKPT / "original" / f"maskrcnn_fold{fold}.pt", map_location="cpu",
                    weights_only=False)
    det = build_model(False, ck.get("mask_res", 28))
    det.load_state_dict(ck["model"])
    det.eval()
    seg = load_fold_seg2(tag, fold)

    ds = ToothDataset(ANN / f"fold{fold}_val.json", train=False, enhance="original")
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    rows = []
    for imgs, targets in loader:
        tgt = targets[0]
        name = tgt["_name"]
        gt = tgt["masks"].numpy().astype(bool)
        h, w = gt.shape[1:]
        gray = (imgs[0][0].numpy() * 255).astype(np.uint8)

        out = det([imgs[0]])[0]
        keep = out["scores"].numpy() >= score_thr
        coarse = np.array([clean_mask(m) for m in out["masks"].numpy()[keep, 0] > 0.5],
                          bool).reshape(-1, h, w)
        scores = out["scores"].numpy()[keep]

        # 每個偵測框各自進第二階段，再以精修後的遮罩與 GT 配對
        fine = []
        for cm in coarse:
            ys, xs = np.nonzero(cm)
            if len(ys) == 0:
                fine.append(cm)
                continue
            box = [float(xs.min()), float(ys.min()),
                   float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
            x0, y0, x1, y1 = crop_box(box, 0.2, w, h)
            prob = predict([seg], gray[y0:y1, x0:x1], False)
            small = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0.5
            m = np.zeros((h, w), bool)
            m[y0:y1, x0:x1] = small
            fine.append(clean_mask(m))
        pred = np.array(fine, bool).reshape(-1, h, w)

        rows += match(pred, scores, gt, name)[0]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="unet_tu-hrnet_w32")
    ap.add_argument("--score-thr", type=float, default=0.35,
                    help="與 eval_maskrcnn.py 預設一致")
    args = ap.parse_args()

    folds = range(5) if args.all else [args.fold]
    EVAL.mkdir(parents=True, exist_ok=True)
    allrows = []
    for f in folds:
        rows = run(f, args.model, args.score_thr)
        allrows += rows
        with (EVAL / f"e2e_oof_{args.model}_fold{f}.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(rows)
        summarize(rows, f"e2e OOF {args.model} fold {f}")
    if len(list(folds)) > 1:
        summarize(allrows, f"e2e OOF {args.model} 全部五折")


if __name__ == "__main__":
    main()
