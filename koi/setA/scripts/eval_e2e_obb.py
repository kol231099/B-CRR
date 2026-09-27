"""OBB 版兩階段的五折 OOF 端到端評估，座標系為原圖。

支援兩種第一階段，對應論文的兩個變體：

    --det maskrcnn   Mask R-CNN 出遮罩 → minAreaRect 推出斜框。這是**對照組**，
                     用來把「換偵測器」與「換框型態」兩個變因拆開，不要當成
                     OBB→seg 來寫，因為斜框是從分割結果推導的，不是偵測出來的。
    --det yoloobb    YOLO11-OBB 直接迴歸斜框。這才是 OBB→seg。

第二階段共用 checkpoints_obb 的 HRNet-w32（兩者都吃轉正 crop）。

一律單模型：五折 OOF 不能用五模型集成，fold f 的驗證影像是其餘四折的訓練資料。
--tta 只做測試時翻轉平均，不引入額外資訊，在 OOF 上合法。

用法：
    py scripts/eval_e2e_obb.py --det maskrcnn --all
    py scripts/eval_e2e_obb.py --det yoloobb --all --tta
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
from eval_seg2_holdout import predict  # noqa: E402
from make_crops_obb import obb_of, warp_of  # noqa: E402
from metrics import FIELDS, match, summarize  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, CKPT, IMAGES, ROOT, ToothDataset, build_model, collate  # noqa: E402
from train_seg2 import build_seg2, split_tag  # noqa: E402
from tta import predict_tta  # noqa: E402

EVAL, PAD = ROOT / "eval", 0.2


def load_seg2_obb(tag: str, fold: int):
    arch, enc = split_tag(tag)
    m = build_seg2(arch, enc, pretrained=False)
    m.load_state_dict(torch.load(ROOT / "checkpoints_obb" / "seg2" / tag / f"fold{fold}.pt",
                                 map_location="cpu", weights_only=False)["model"])
    m.eval()
    return m


def obbs_maskrcnn(fold: int, gray, img_t, score_thr: float, use_tta: bool, hw):
    """Mask R-CNN 的遮罩 → minAreaRect。回傳 [(cx,cy,rw,rh,ang)], scores。"""
    global _MR
    if _MR[0] != fold:
        ck = torch.load(CKPT / "original" / f"maskrcnn_fold{fold}.pt", map_location="cpu",
                        weights_only=False)
        m = build_model(False, ck.get("mask_res", 28))
        m.load_state_dict(ck["model"])
        m.eval()
        _MR[0], _MR[1] = fold, m
    model = _MR[1]
    if use_tta:
        prob, _, scores = predict_tta(model, gray, score_thr)
        masks = np.array([clean_mask(x) for x in prob > 0.5], bool).reshape(-1, *hw)
    else:
        out = model([img_t])[0]
        keep = out["scores"].numpy() >= score_thr
        masks = np.array([clean_mask(x) for x in out["masks"].numpy()[keep, 0] > 0.5],
                         bool).reshape(-1, *hw)
        scores = out["scores"].numpy()[keep]
    boxes = []
    for m8 in masks:
        if not m8.any():
            boxes.append(None)
            continue
        boxes.append(obb_of(m8.astype(np.uint8)))
    ok = [i for i, b in enumerate(boxes) if b is not None]
    return [boxes[i] for i in ok], np.asarray(scores)[ok]


def obbs_yolo(fold: int, path: Path, conf: float, use_tta: bool, hw):
    global _YL
    if _YL[0] != (fold, _ARCH[0]):
        from ultralytics import YOLO
        _YL[0] = (fold, _ARCH[0])
        _YL[1] = YOLO(ROOT / "yolo_obb_runs" / _ARCH[0] / f"fold{fold}" / "weights" / "best.pt")
    res = _YL[1].predict(str(path), conf=conf, imgsz=1024, augment=use_tta, verbose=False)[0]
    if res.obb is None or len(res.obb) == 0:
        return [], np.zeros(0)
    boxes = []
    for quad in res.obb.xyxyxyxy.cpu().numpy():
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(quad.reshape(-1, 2).astype(np.float32))
        if rw > rh:
            rw, rh, ang = rh, rw, ang + 90
        boxes.append((cx, cy, rw, rh, ang))
    return boxes, res.obb.conf.cpu().numpy()


_MR: list = [-1, None]
_YL: list = [None, None]
_ARCH: list = ["yolo11s"]   # 由 --arch 設定，決定讀哪個 yolo_obb_runs 子目錄


@torch.no_grad()
def run(fold: int, det: str, tag: str, thr: float, use_tta: bool) -> list[dict]:
    seg = load_seg2_obb(tag, fold)
    ds = ToothDataset(ANN / f"fold{fold}_val.json", train=False, enhance="original")
    rows = []
    for imgs, targets in DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate):
        t = targets[0]
        name = t["_name"]
        gt = t["masks"].numpy().astype(bool)
        h, w = gt.shape[1:]
        gray = (imgs[0][0].numpy() * 255).astype(np.uint8)

        if det == "maskrcnn":
            boxes, scores = obbs_maskrcnn(fold, gray, imgs[0], thr, use_tta, (h, w))
        else:
            boxes, scores = obbs_yolo(fold, IMAGES / name, thr, use_tta, (h, w))

        fine = []
        for (cx, cy, rw, rh, ang) in boxes:
            M, cw, ch = warp_of(cx, cy, rw, rh, ang, PAD)
            crop = cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR)
            prob = predict([seg], crop, use_tta)
            small = cv2.resize(prob, (cw, ch), interpolation=cv2.INTER_LINEAR)
            back = cv2.warpAffine(small, cv2.invertAffineTransform(M), (w, h),
                                  flags=cv2.INTER_LINEAR) > 0.5
            fine.append(clean_mask(back))
        pred = np.array(fine, bool).reshape(-1, h, w)
        rows += match(pred, np.asarray(scores), gt, name)[0]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", choices=["maskrcnn", "yoloobb"], required=True)
    ap.add_argument("--arch", default="yolo11s",
                    choices=["yolo11s", "yolov8s", "yolo12s", "yolo26s"],
                    help="--det yoloobb 時，要讀 yolo_obb_runs 下的哪個架構")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="unet_tu-hrnet_w32")
    ap.add_argument("--thr", type=float, default=0.35,
                    help="偵測門檻，與 eval_maskrcnn.py / eval_yolo.py 對齊")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    _ARCH[0] = args.arch
    sfx = "_tta" if args.tta else ""
    det_tag = args.arch if args.det == "yoloobb" else "maskrcnn"
    stem = f"obb_{det_tag}_{args.model}{sfx}"
    EVAL.mkdir(parents=True, exist_ok=True)
    allrows = []
    folds = list(range(5)) if args.all else [args.fold]
    for f in folds:
        rows = run(f, args.det, args.model, args.thr, args.tta)
        allrows += rows
        with (EVAL / f"{stem}_fold{f}.csv").open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(rows)
        summarize(rows, f"{stem} fold {f}")
    if len(folds) > 1:
        summarize(allrows, f"{stem} 全部五折")


if __name__ == "__main__":
    main()
