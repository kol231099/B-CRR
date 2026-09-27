"""評估某個 fold 的 Mask R-CNN，輸出逐顆牙的指標與錯誤案例圖。

所有指標都在**原圖座標**上計算。在 crop 內算 Dice 會系統性虛高，因為裁切已把
大部分背景移掉、分母變小；三條 pipeline 要能互比，就必須共用同一個座標系。

指標
----
Dice / IoU   標準分割指標。牙齒是大而完整的物件，Dice 很快就會停在 0.95 上下，
             對邊界的細微差異不敏感。
HD95         預測輪廓與真實輪廓之間距離的 95 百分位（像素）。Dice 幾乎不動的
             情況下 HD95 仍會反映邊界偏移，對「圍出範圍」這個任務更貼切。
             取 95% 而非最大值，是為了不讓單一個離群像素主導。
TP/FP/FN     以 IoU >= 0.5 貪婪配對。漏一顆牙比圈歪一顆嚴重得多，但 Dice 完全
             反映不出來——漏掉的牙根本不會進 Dice 的平均，所以必須另外報。

用法：
    py koi/scripts/eval_maskrcnn.py --fold 0
    py koi/scripts/eval_maskrcnn.py --fold 0 --no-figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import FIELDS, match, summarize  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from tta import predict_tta  # noqa: E402
from train_maskrcnn import ANN, CKPT, IMAGES, ROOT, ToothDataset, build_model, collate  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

EVAL = ROOT / "eval"


@torch.no_grad()
def run(fold: int, score_thr: float, figures: bool, enhance: str = "original", tag: str = "",
        use_tta: bool = False) -> list[dict]:
    ckpt = torch.load(CKPT / (tag or enhance) / f"maskrcnn_fold{fold}.pt",
                      map_location="cpu", weights_only=False)
    model = build_model(False, ckpt.get("mask_res", 28))
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = ToothDataset(ANN / f"fold{fold}_val.json", train=False, enhance=enhance)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    out_dir = EVAL / f"maskrcnn_{enhance}_fold{fold}"
    if figures:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for imgs, targets in loader:
        tgt = targets[0]
        name = tgt["_name"]
        gt = tgt["masks"].numpy().astype(bool)
        if use_tta:
            gray = (imgs[0][0].numpy() * 255).astype(np.uint8)
            prob, _, scores = predict_tta(model, gray, score_thr)
            pred = np.array([clean_mask(m) for m in prob > 0.5], bool).reshape(-1, *gt.shape[1:])
            r, matched = match(pred, scores, gt, name)
            rows += r
            continue
        out = model([imgs[0]])[0]
        keep = out["scores"].numpy() >= score_thr
        pred = np.array([clean_mask(m) for m in out["masks"].numpy()[keep, 0] > 0.5],
                        bool).reshape(-1, *gt.shape[1:])
        scores = out["scores"].numpy()[keep]

        r, matched = match(pred, scores, gt, name)
        rows += r

        if figures:
            vis = cv2.cvtColor(cv2.imread(str(IMAGES / name), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
            for g in gt:  # 真實輪廓：白色
                cv2.drawContours(vis, cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_NONE)[0], -1, (255, 255, 255), 3)
            for pi, p in enumerate(pred):  # TP 綠、FP 紅
                col = (0, 255, 80) if pi in matched else (60, 60, 255)
                cv2.drawContours(vis, cv2.findContours(p.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_NONE)[0], -1, col, 3)
                ys, xs = np.nonzero(p)
                if len(xs):
                    cv2.putText(vis, f"{scores[pi]:.2f}", (int(xs.min()), max(30, int(ys.min()) - 8)),
                                0, 1.0, col, 3)
            cv2.imwrite(str(out_dir / name.replace(".jpg", ".png")), vis)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--enhance", default="original")
    ap.add_argument("--tag", default="", help="權重子資料夾，預設同 --enhance")
    ap.add_argument("--score-thr", type=float, default=0.35,
                    help="由 sweep_conf.py 掃出：0.25~0.45 表現完全相同，取中點最耐用")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--tta", action="store_true", help="測試時增強：四種翻轉推論後平均")
    args = ap.parse_args()

    rows = run(args.fold, args.score_thr, not args.no_figures, args.enhance, args.tag, args.tta)
    EVAL.mkdir(parents=True, exist_ok=True)
    with (EVAL / f"maskrcnn_{args.tag or args.enhance}{'_tta' if args.tta else ''}_fold{args.fold}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    summarize(rows, f"maskrcnn [{args.tag or args.enhance}{' +TTA' if args.tta else ''}] fold {args.fold}")
    print(f"  → {EVAL}/maskrcnn_{args.tag or args.enhance}"
          f"{'_tta' if args.tta else ''}_fold{args.fold}.csv")


if __name__ == "__main__":
    main()
