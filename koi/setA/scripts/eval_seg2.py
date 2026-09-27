"""評估第二階段的語意分割模型，指標與第一階段共用 metrics.py。

關鍵：貼回原圖座標
------------------
第二階段是在裁切的小圖上預測的，但**指標必須在原圖座標上計算**——在 crop 內算
Dice 會系統性虛高，因為裁切已經把大部分背景移掉、分母變小。所以流程是：

    crop 上預測 → 縮放回 crop 原尺寸 → 依 manifest 的 (x0,y0) 貼回全圖 → 再算指標

這樣算出來的數字才能跟 Mask R-CNN 的並排比較。

測試時增強
----------
--tta 會對每個 crop 做四種翻轉推論後平均**機率圖**再二值化。語意分割的 TTA 比
實例分割單純得多：不需要配對實例，直接把機率圖翻回原方向平均即可。

第一階段的 Mask R-CNN 也用同一組翻轉，兩邊條件一致，比較才成立。

oracle 條件
-----------
這裡用的是**真實標註的 bbox** 裁切出來的 crop，等於假設 ROI 完美。它回答的是
「給定正確的 ROI，這個解碼器能把邊界畫多準」，不是端到端的表現。端到端還要
把偵測誤差算進去，那是另一個實驗（用 Mask R-CNN 預測的框重新裁切）。

用法：
    py scripts/eval_seg2.py --arch unet --encoder resnet34
    py scripts/eval_seg2.py --all
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
from train_seg2 import ARCHS, CROPS, SIZE, build_seg2, fold_ids  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ANN, CKPT, EVAL = ROOT / "annotations", ROOT / "checkpoints", ROOT / "eval"


@torch.no_grad()
def run_one(arch: str, encoder: str, use_tta: bool = False) -> list[dict]:
    import segmentation_models_pytorch as smp

    tag = f"{arch}_{encoder}"
    manifest = {r["crop_id"]: r for r in csv.DictReader((CROPS / "manifest.csv").open(encoding="utf-8"))}
    coco = json.loads((ANN / "instances_all.json").read_text(encoding="utf-8"))
    gt_of = {(a["image_id"], a["id"]): a for a in coco["annotations"]}
    # 每顆牙在其所屬影像中的序號。第二階段一次只評一顆，match() 回傳的 gt_idx
    # 永遠是 0，無法與第一階段配對——這裡補回真正的序號。
    order: dict[tuple[int, int], int] = {}
    per: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        per.setdefault(a["image_id"], []).append(a)
    for iid, lst in per.items():
        for i, a in enumerate([x for x in lst if not x["iscrowd"]]):
            order[(iid, a["id"])] = i
    imgs = {i["id"]: i for i in coco["images"]}
    id_of_name = {v["file_name"]: k for k, v in imgs.items()}

    rows = []
    for fold in range(5):
        ck = CKPT / "seg2" / tag / f"fold{fold}.pt"
        if not ck.exists():
            continue
        model = build_seg2(arch, encoder, pretrained=False)
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
        model.eval()

        _, va_ids = fold_ids(fold)
        for cid in va_ids:
            r = manifest[cid]
            h, w = int(r["img_h"]), int(r["img_w"])
            x0, y0, x1, y1 = (int(r[k]) for k in ("x0", "y0", "x1", "y1"))

            img = cv2.imread(str(CROPS / "images" / f"{cid}.png"), cv2.IMREAD_GRAYSCALE)
            base = cv2.resize(img, SIZE[::-1], interpolation=cv2.INTER_AREA)
            views = [(False, False), (True, False), (False, True), (True, True)] if use_tta \
                else [(False, False)]
            acc = []
            for fh, fv in views:
                v = base[:, ::-1] if fh else base
                v = v[::-1] if fv else v
                t = torch.from_numpy(np.ascontiguousarray(v))
                t = t.float().div(255).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
                o = torch.sigmoid(model(t))[0, 0].numpy()
                if fv:
                    o = o[::-1]
                if fh:
                    o = o[:, ::-1]
                acc.append(np.ascontiguousarray(o))
            prob = np.mean(acc, axis=0)

            # 縮放回 crop 原尺寸，再貼回全圖座標
            small = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0.5
            pred = np.zeros((h, w), bool)
            pred[y0:y1, x0:x1] = small
            pred = clean_mask(pred)

            a = gt_of[(id_of_name[r["image"]], int(r["ann_id"]))]
            gt = cv2.fillPoly(np.zeros((h, w), np.uint8),
                              [np.array(a["segmentation"][0], np.int32).reshape(-1, 2)], 1).astype(bool)
            rr = match(pred[None], np.array([1.0]), gt[None], r["image"])[0]
            for x in rr:
                x["gt_idx"] = order[(id_of_name[r["image"]], int(r["ann_id"]))]
            rows += rr
        print(f"  {tag} fold{fold} 完成", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--all", action="store_true", help="評估 checkpoints/seg2 下所有已完成的模型")
    ap.add_argument("--tta", action="store_true", help="四種翻轉推論後平均")
    args = ap.parse_args()

    targets = []
    if args.all:
        for d in sorted((CKPT / "seg2").glob("*")):
            if list(d.glob("fold*.pt")):
                arch, enc = d.name.split("_", 1)
                targets.append((arch, enc))
    else:
        targets = [(args.arch, args.encoder)]

    EVAL.mkdir(parents=True, exist_ok=True)
    for arch, enc in targets:
        rows = run_one(arch, enc, args.tta)
        if not rows:
            continue
        tag = f"{arch}_{enc}" + ("_tta" if args.tta else "")
        with (EVAL / f"seg2_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        summarize(rows, f"seg2 {tag}")
        print()


if __name__ == "__main__":
    main()
