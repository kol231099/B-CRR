#!/usr/bin/env python3
"""端到端評估：Mask R-CNN 偵測出框 → 用該框裁切 → 第二階段圈輪廓。

這是實際部署時會發生的流程。表 1 的數字用的是標註框（oracle bbox），
假設偵測完美；此處改用 Mask R-CNN 真實預測的框，因此結果一定不高於 oracle，
差距即為偵測誤差的代價。

漏檢的牙齒沒有第二階段輸出，計為 FN——這是 oracle 設定下不會出現的失敗模式。

同時報告裁切本身的品質：預測框與標註框的 IoU、中心偏移，以及最關鍵的
「標註遮罩有多少比例落在裁切範圍內」——被裁掉的部分第二階段永遠救不回來。

用法
    py scripts/eval_e2e.py --all
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
from bpr_holdout import load_seg2  # noqa: E402
from eval_seg2_holdout import HOLD, gt_mask, predict  # noqa: E402
from make_crops import crop_box  # noqa: E402
from metrics import FIELDS, match  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, CKPT, ROOT, build_model  # noqa: E402
from train_seg2 import ARCHS  # noqa: E402
from tta import predict_tta  # noqa: E402

EVAL = ROOT / "eval"


def detect() -> tuple[dict, list[dict]]:
    """跑 Mask R-CNN，回傳每張影像配對成功的 (GT 序號 → 預測遮罩) 與裁切品質統計。"""
    coco = json.loads((ANN / "holdout.json").read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    per: dict[int, list] = {}
    for a in coco["annotations"]:
        per.setdefault(a["image_id"], []).append(a)

    ck = torch.load(CKPT / "original" / "maskrcnn_final.pt", map_location="cpu",
                    weights_only=False)
    model = build_model(False, ck.get("mask_res", 28))
    model.load_state_dict(ck["model"])
    model.eval()

    det: dict = {}
    box_rows: list[dict] = []
    for iid, anns in sorted(per.items()):
        im = imgs[iid]
        gray = cv2.imread(str(HOLD / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h, w = gray.shape
        keep = [a for a in anns if not a.get("iscrowd", 0)]
        gts = np.stack([gt_mask(a, h, w) for a in keep])

        prob, _, scores = predict_tta(model, gray, 0.35)
        pred = np.array([clean_mask(m) for m in prob > 0.5], bool)
        matched = {}
        if len(pred):
            _, matched = match(pred, scores, gts, im["file_name"])

        gt_of_pred = {gi: pi for pi, gi in matched.items()}
        for gi, a in enumerate(keep):
            entry = {"image": im["file_name"], "gt_idx": gi, "gray": gray,
                     "gt": gts[gi], "hw": (h, w)}
            if gi not in gt_of_pred:
                entry["box"] = None            # 漏檢
            else:
                pm = pred[gt_of_pred[gi]]
                ys, xs = np.nonzero(pm)
                # 預測框取預測遮罩的外接矩形，與偵測輸出等價且與 TTA 平均一致
                pbox = [float(xs.min()), float(ys.min()),
                        float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
                entry["box"] = pbox

                gx0, gy0, gx1, gy1 = crop_box(a["bbox"], 0.2, w, h)
                px0, py0, px1, py1 = crop_box(pbox, 0.2, w, h)
                ix0, iy0 = max(gx0, px0), max(gy0, py0)
                ix1, iy1 = min(gx1, px1), min(gy1, py1)
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                union = (gx1 - gx0) * (gy1 - gy0) + (px1 - px0) * (py1 - py0) - inter
                covered = gts[gi][py0:py1, px0:px1].sum() / max(gts[gi].sum(), 1)
                box_rows.append({
                    "image": im["file_name"], "gt_idx": gi,
                    "box_iou": round(inter / union if union else 0.0, 4),
                    "dx": round(((px0 + px1) - (gx0 + gx1)) / 2, 1),
                    "dy": round(((py0 + py1) - (gy0 + gy1)) / 2, 1),
                    "gt_covered": round(float(covered), 4),
                })
            det.setdefault(im["file_name"], []).append(entry)
    return det, box_rows


def run_one(tag: str, det: dict) -> list[dict]:
    ms = load_seg2(tag)
    if not ms:
        return []
    rows = []
    for name, entries in det.items():
        for e in entries:
            if e["box"] is None:                      # 漏檢：沒有第二階段輸出
                rows.append({f: "" for f in FIELDS} | {
                    "image": name, "gt_idx": e["gt_idx"], "kind": "FN", "score": 0.0})
                continue
            h, w = e["hw"]
            x0, y0, x1, y1 = crop_box(e["box"], 0.2, w, h)
            prob = predict(ms, e["gray"][y0:y1, x0:x1], True)
            small = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0.5
            m = np.zeros((h, w), bool)
            m[y0:y1, x0:x1] = small
            m = clean_mask(m)
            r = match(m[None], np.array([1.0]), e["gt"][None], name)[0]
            for x in r:
                x["gt_idx"] = e["gt_idx"]
            rows += r
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="只跑 U-Net decoder 的")
    ap.add_argument("--all-seg2", action="store_true", help="跑全部第二階段模型")
    ap.add_argument("--model", default="unet_tu-hrnet_w32")
    args = ap.parse_args()

    print("跑 Mask R-CNN 偵測…", flush=True)
    det, box_rows = detect()
    n_gt = sum(len(v) for v in det.values())
    n_hit = sum(1 for v in det.values() for e in v if e["box"] is not None)
    EVAL.mkdir(exist_ok=True)
    with (EVAL / "e2e_boxes.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(box_rows[0]))
        wr.writeheader()
        wr.writerows(box_rows)

    b = np.array([r["box_iou"] for r in box_rows])
    c = np.array([r["gt_covered"] for r in box_rows])
    d = np.hypot([r["dx"] for r in box_rows], [r["dy"] for r in box_rows])
    print(f"\n=== Mask R-CNN 裁切品質（{n_hit}/{n_gt} 顆偵測到）===")
    print(f"  裁切框 IoU        中位 {np.median(b):.4f}　最低 {b.min():.4f}")
    print(f"  中心偏移          中位 {np.median(d):.1f} px　最大 {d.max():.1f} px")
    print(f"  標註落在框內比例   中位 {np.median(c):.4f}　最低 {c.min():.4f}")
    print(f"  完整包住的顆數     {int((c >= 0.9999).sum())}/{len(c)}")

    if args.all_seg2:
        tags = [d.name for d in sorted((CKPT / "seg2").iterdir())
                if d.is_dir() and list(d.glob("fold*.pt"))]
    elif args.all:
        tags = [d.name for d in sorted((CKPT / "seg2").iterdir())
                if d.is_dir() and d.name.startswith("unet_") and list(d.glob("fold*.pt"))]
    else:
        tags = [args.model]
    print(f"\n=== 端到端指標（Mask R-CNN 框 → 第二階段）===")
    print(f"  {'encoder':<22}{'TP':>4}{'FN':>4}{'Dice':>9}{'B-IoU':>9}{'HD95':>7}{'ASSD':>7}")
    print("  " + "-" * 62)
    for tag in tags:
        rows = run_one(tag, det)
        if not rows:
            continue
        with (EVAL / f"e2e_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(rows)
        tp = [r for r in rows if r["kind"] == "TP"]
        fn = [r for r in rows if r["kind"] == "FN"]
        g = lambda k: np.median([float(r[k]) for r in tp])
        enc = tag[len("unet_"):] if tag.startswith("unet_") else tag
        print(f"  {enc:<22}{len(tp):>4}{len(fn):>4}{g('dice'):>9.4f}"
              f"{g('biou'):>9.4f}{g('hd95'):>7.1f}{g('assd'):>7.2f}", flush=True)


if __name__ == "__main__":
    main()
