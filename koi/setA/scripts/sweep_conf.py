"""掃描偵測信心門檻，看 Mask R-CNN 與 SAM 2 在哪個 conf 表現最好。

一個 conf 同時管兩個模型：Mask R-CNN 的偵測結果被門檻篩過之後，留下來的 box
才會餵給 SAM 2。所以這是在調整整條 pipeline 的入口，不是只調一個模型。

效率
----
SAM 2 只跑一次，用 conf >= 0.05 的所有 box；之後每個門檻只是對快取的遮罩取
子集。若每個門檻都重跑一次 SAM 2，時間會是 19 倍。

指標怎麼看
----------
    Dice(TP)      只算配對成功的牙。**單看這個會被騙**——門檻拉高會濾掉難的牙，
                  剩下容易的，Dice 反而上升，但實際上漏了更多。
    全牙 Dice     對每一顆真實牙齒計分，漏掉的算 0，再除以總牙數。這才是
                  「整批圖平均圈得多好」，也是挑門檻該看的主指標。
    F1            偵測層級的調和平均。**目前這個數字不可信**：13 個 FP 全都是
                  未標註的真牙，用 F1 挑門檻等於為了掩蓋標註缺失而犧牲召回。
    F1(忽略FP)    把 FP 全部當成未標註真牙來算（等於 precision = 1）。真實值
                  落在 F1 與這個之間，補完 unclear 標註後才會確定。

用法：
    py koi/scripts/sweep_conf.py
    py koi/scripts/sweep_conf.py --min-conf 0.05 --step 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weights import weight  # noqa: E402
from metrics import match  # noqa: E402
from train_maskrcnn import ANN, CKPT, IMAGES, build_model  # noqa: E402


def collect(min_conf: float) -> list[dict]:
    """對每個 fold 的 val 影像跑一次推論，快取遮罩與分數。"""
    from ultralytics import SAM

    sam = SAM(weight("sam2.1_b.pt"))
    cache = []
    for fold in range(5):
        model = build_model(False)
        model.load_state_dict(torch.load(CKPT / f"maskrcnn_fold{fold}.pt",
                                         map_location="cpu", weights_only=False)["model"])
        model.eval()
        coco = json.loads((ANN / f"fold{fold}_val.json").read_text(encoding="utf-8"))
        by: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            by.setdefault(a["image_id"], []).append(a)

        for im in coco["images"]:
            h, w = im["height"], im["width"]
            path = IMAGES / im["file_name"]
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            gt = np.stack([
                cv2.fillPoly(np.zeros((h, w), np.uint8),
                             [np.array(a["segmentation"][0], np.int32).reshape(-1, 2)], 1).astype(bool)
                for a in by.get(im["id"], []) if not a["iscrowd"]
            ])

            t = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1)
            with torch.no_grad():
                o = model([t])[0]
            k = o["scores"].numpy() >= min_conf
            scores = o["scores"].numpy()[k]
            rcnn = o["masks"].numpy()[k, 0] > 0.5
            boxes = o["boxes"].numpy()[k]

            if len(boxes):
                r = sam.predict(str(path), bboxes=boxes.tolist(), verbose=False)[0]
                m = r.masks.data.cpu().numpy() > 0.5
                sam_m = (np.stack([cv2.resize(x.astype(np.uint8), (w, h),
                                              interpolation=cv2.INTER_NEAREST).astype(bool) for x in m])
                         if m.shape[1:] != (h, w) else m)
            else:
                sam_m = np.zeros((0, h, w), bool)

            cache.append({"name": im["file_name"], "gt": gt, "scores": scores,
                          "rcnn": rcnn, "sam": sam_m[: len(rcnn)]})
        print(f"  fold{fold} 完成", flush=True)
    return cache


def score_at(cache: list[dict], key: str, conf: float) -> dict:
    rows, n_gt = [], 0
    for c in cache:
        k = c["scores"] >= conf
        rows += match(c[key][k], c["scores"][k], c["gt"], c["name"])[0]
        n_gt += len(c["gt"])
    tp = [r for r in rows if r["kind"] == "TP"]
    n_fp = sum(r["kind"] == "FP" for r in rows)
    n_fn = sum(r["kind"] == "FN" for r in rows)
    d = np.array([float(r["dice"]) for r in tp]) if tp else np.array([0.0])
    recall = len(tp) / n_gt
    prec = len(tp) / (len(tp) + n_fp) if len(tp) + n_fp else 0.0
    return {
        "conf": conf, "tp": len(tp), "fp": n_fp, "fn": n_fn,
        "dice_tp": float(np.median(d)),
        "dice_all": float(d.sum() / n_gt),          # 漏掉的算 0
        "recall": recall, "precision": prec,
        "f1": 2 * prec * recall / (prec + recall) if prec + recall else 0.0,
        "f1_ignore_fp": 2 * recall / (1 + recall),  # 把 precision 當作 1
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-conf", type=float, default=0.05)
    ap.add_argument("--max-conf", type=float, default=0.95)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    print("推論中（SAM 2 只跑一次，之後各門檻取子集）…")
    cache = collect(args.min_conf)
    confs = np.arange(args.min_conf, args.max_conf + 1e-9, args.step)

    for key, label in (("rcnn", "Mask R-CNN"), ("sam", "SAM 2（用 Mask R-CNN 的 box）")):
        print(f"\n{label}")
        print(f"{'conf':>6}{'TP':>5}{'FP':>5}{'FN':>5}{'Dice(TP)':>11}{'全牙Dice':>11}"
              f"{'recall':>9}{'F1':>8}{'F1(忽略FP)':>12}")
        print("-" * 76)
        best = None
        for c in confs:
            s = score_at(cache, key, float(c))
            star = ""
            if best is None or s["dice_all"] > best["dice_all"]:
                best = s
            print(f"{s['conf']:>6.2f}{s['tp']:>5}{s['fp']:>5}{s['fn']:>5}{s['dice_tp']:>11.4f}"
                  f"{s['dice_all']:>11.4f}{s['recall']:>9.3f}{s['f1']:>8.3f}{s['f1_ignore_fp']:>12.3f}{star}")
        print(f"  → 全牙 Dice 最高：conf = {best['conf']:.2f}（{best['dice_all']:.4f}，"
              f"漏 {best['fn']} 顆）")


if __name__ == "__main__":
    main()
