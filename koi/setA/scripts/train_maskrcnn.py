"""在整張根尖片上訓練 Mask R-CNN，一次一個 fold。

Mask R-CNN 吃整張原圖，不吃預先裁好的小圖——ROI 是 RPN 提出、ROIAlign 取出
的，所以訓練資料就是 fold{k}_train.json 加上同資料夾裡的原圖。

三個對這批資料做的調整
----------------------
anchor 長寬比    torchvision 預設 aspect_ratios=(0.5, 1.0, 2.0)，那是高/寬比。
                 本資料集牙齒的高/寬中位數 2.98、範圍 1.48~4.60，89% 落在預設
                 的上限之外，RPN 會提不出貼合的候選框。改成 (1.5, 3.0, 4.5)。
                 每個位置仍是 3 個 anchor，所以 RPN head 不用動。

bbox 由遮罩導出  即使沒有增強也一律從遮罩重算 bbox。翻轉後若各自變換影像與
                 座標，很容易出現差一像素或左右顛倒的錯誤；從遮罩重算則不可
                 能不一致。

ignore 區塗黑    torchvision 的偵測模型沒有 ignore region 的概念，未標註的牙
                 會被當成背景，等於教模型不要偵測那種牙。iscrowd=1 的區域在
                 此直接塗黑，讓它既不是前景也不提供可學的紋理。目前標註檔裡
                 還沒有 unclear，這段不會觸發。

最終模型（--fold -1）
--------------------
交叉驗證的用途是取得可信的效能估計；正式使用的模型應以**全部資料**訓練。
--fold -1 會用 instances_all.json 的所有影像訓練，不保留驗證集。

沒有驗證集就無法挑「最佳 epoch」，只能訓練固定輪數後存下最後一個。這是可行的：
超參數已由交叉驗證確認，且五個 fold 都未觀察到過擬合（val Dice 一路緩升至第 40
個 epoch）。效能請引用交叉驗證的數字，不要宣稱這個模型更準——學習曲線顯示
資料量在 n=10 之後即已飽和。

遮罩損失
--------
--mask-loss boundary 會把 torchvision 的逐像素 BCE 換成 BCE 與 boundary loss 的
混合，權重隨 epoch 由 0 升到 --boundary-alpha。動機是主指標 HD95 量的是輪廓距離，
而 BCE 對每個像素一視同仁——實測誤差集中在牙齒兩端曲率大的轉折，正是 BCE 最弱
的地方。詳見 boundary_loss.py。

遮罩解析度
----------
--mask-res 控制每個 ROI 的遮罩輸出邊長。torchvision 預設 roi_pool 14x14、經反卷積
加倍成 28x28，再放大回 bbox。牙齒 bbox 中位 261x743 px，等於每個遮罩格子涵蓋
9.3 x 26.5 px——這跟實測 HD95 中位 11 px 是同一個量級，也就是邊界精度的天花板。
設 56 會把 roi_pool 改成 28x28、輸出 56x56，量化降到 4.7 x 13.3 px。

代價是遮罩頭的參數量與計算量上升，且該層無法沿用 COCO 預訓練權重（本來就會
因為類別數不同而重新初始化，所以實際上不多付什麼）。

為什麼預設用 CPU 而不是 MPS
---------------------------
實測 maskrcnn_resnet50_fpn_v2 在 MPS 上會卡死：單一個 forward + backward 超過
180 秒仍未完成，而同一批資料在 CPU 上只要 4 秒。原因是模型裡有算子沒有 MPS
kernel，靠 PYTORCH_ENABLE_MPS_FALLBACK 回退到 CPU 時每步都要來回搬張量。
M4 Pro 的 CPU 跑 20 張圖的 epoch 約 40 秒，完全可接受，所以預設就用 CPU。
若日後 torchvision 補上 kernel，加 --device mps 即可。

影像增強
--------
--enhance 指定 enhance.py 裡的方法，訓練與推論都會套用同一種。務必一致：用未
增強的影像訓練、卻餵增強過的影像推論，測到的是分布不一致，不是增強的效果。
權重存到 koi/checkpoints/<方法名>/maskrcnn_fold<k>.pt，每種方法一個資料夾。

用法：
    py koi/scripts/train_maskrcnn.py --fold 0
    py koi/scripts/train_maskrcnn.py --fold 0 --enhance clahe+unsharp
    py koi/scripts/train_maskrcnn.py --fold 0 --epochs 60 --batch 2
    py koi/scripts/train_maskrcnn.py --fold 0 --epochs 2   # 冒煙測試
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "images"
ANN = ROOT / "annotations"
CKPT = ROOT / "checkpoints"
NUM_CLASSES = 2  # 背景 + tooth


class ToothDataset(Dataset):
    def __init__(self, coco_json: Path, train: bool, enhance: str = "original",
                 subset: int = 0, seed: int = 0):
        import random

        from enhance import METHODS

        data = json.loads(coco_json.read_text(encoding="utf-8"))
        self.train = train
        self.enhance = METHODS[enhance]
        by: dict[int, list[dict]] = {}
        for a in data["annotations"]:
            by.setdefault(a["image_id"], []).append(a)
        images = data["images"]
        if subset:
            # 固定種子後取前 N 張，使 n=5 ⊂ n=10 ⊂ n=15 ⊂ n=20。學習曲線的各點
            # 若各自獨立抽樣，點與點的差異會混入抽樣變異，看不出資料量的效果。
            images = sorted(images, key=lambda im: im["file_name"])
            random.Random(seed).shuffle(images)
            images = images[:subset]
        self.items = [(im, by.get(im["id"], [])) for im in images]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        im, anns = self.items[i]
        gray = self.enhance(cv2.imread(str(IMAGES / im["file_name"]), cv2.IMREAD_GRAYSCALE))
        h, w = gray.shape

        masks = []
        for a in anns:
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.array(a["segmentation"][0], np.int32).reshape(-1, 2)], 1)
            if a["iscrowd"]:
                gray = gray * (1 - m)  # ignore 區塗黑，不進 target
            else:
                masks.append(m)
        masks = np.stack(masks) if masks else np.zeros((0, h, w), np.uint8)

        if self.train:
            gray, masks = augment(gray, masks)

        boxes = np.array([bbox_of(m) for m in masks], np.float32).reshape(-1, 4)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, masks = boxes[keep], masks[keep]

        img = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1)
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "masks": torch.from_numpy(masks),
            "image_id": torch.tensor(im["id"]),
            "_name": im["file_name"],
        }
        return img, target


def bbox_of(m: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def augment(gray: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """左右與上下翻轉都是解剖上合理的：資料裡本來就同時有左右側、上下顎的片子。"""
    rng = np.random
    if rng.rand() < 0.5:
        gray, masks = gray[:, ::-1], masks[:, :, ::-1]
    if rng.rand() < 0.5:
        gray, masks = gray[::-1], masks[:, ::-1]
    if rng.rand() < 0.8:  # 曝光差異是根尖片最大的變異來源
        gamma = rng.uniform(0.7, 1.4)
        gain = rng.uniform(0.85, 1.15)
        gray = np.clip(((gray / 255.0) ** gamma) * gain * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(gray), np.ascontiguousarray(masks)


def collate(batch):
    return tuple(zip(*batch))


def build_model(pretrained: bool, mask_res: int = 28) -> torch.nn.Module:
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)

    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, NUM_CLASSES)

    if mask_res != 28:
        # 反卷積會把 roi_pool 的輸出加倍，所以 roi_pool 設一半
        model.roi_heads.mask_roi_pool.output_size = (mask_res // 2, mask_res // 2)

    sizes = model.rpn.anchor_generator.sizes
    model.rpn.anchor_generator = AnchorGenerator(sizes, ((1.5, 3.0, 4.5),) * len(sizes))
    return model


@torch.no_grad()
def evaluate(model, loader, device, score_thr: float = 0.5) -> dict:
    """在原圖座標上算 per-tooth Dice 與偵測層級的 TP/FP/FN。

    Dice 一律在全圖座標算——在 crop 內算會虛高，因為背景已被裁掉、分母變小。
    """
    model.eval()
    dices, tp, fp, fn = [], 0, 0, 0
    for imgs, targets in loader:
        outs = model([i.to(device) for i in imgs])
        for out, tgt in zip(outs, targets):
            gt = tgt["masks"].numpy().astype(bool)
            keep = out["scores"].cpu().numpy() >= score_thr
            pred = (out["masks"].cpu().numpy()[keep, 0] > 0.5)

            used = set()
            for p in pred:
                best, best_iou = -1, 0.0
                for j, g in enumerate(gt):
                    if j in used:
                        continue
                    inter = np.logical_and(p, g).sum()
                    union = np.logical_or(p, g).sum()
                    iou = inter / union if union else 0.0
                    if iou > best_iou:
                        best, best_iou = j, iou
                if best_iou >= 0.5:
                    used.add(best)
                    tp += 1
                    dices.append(2 * np.logical_and(p, gt[best]).sum() / (p.sum() + gt[best].sum()))
                else:
                    fp += 1
            fn += len(gt) - len(used)

    return {
        "dice": float(np.mean(dices)) if dices else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--enhance", default="original", help="enhance.py 裡的方法名")
    ap.add_argument("--subset", type=int, default=0, help="只用前 N 張訓練影像，0 = 全部")
    ap.add_argument("--tag", default="", help="權重子資料夾名稱，預設同 --enhance")
    ap.add_argument("--mask-res", type=int, default=28, choices=[28, 56], help="每個 ROI 的遮罩邊長")
    ap.add_argument("--mask-loss", default="bce", choices=["bce", "boundary"], help="遮罩損失")
    ap.add_argument("--boundary-alpha", type=float, default=0.5, help="boundary 項的最終權重")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--device", default="cpu", help="預設 cpu；MPS 上此模型會卡死，見檔頭說明")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)

    final = args.fold < 0
    tr = ToothDataset(ANN / ("instances_all.json" if final else f"fold{args.fold}_train.json"),
                      train=True, enhance=args.enhance, subset=args.subset)
    va = None if final else ToothDataset(ANN / f"fold{args.fold}_val.json",
                                        train=False, enhance=args.enhance)
    # num_workers=0：MPS 上多行程 worker 常在 fork 時卡住，而且這裡每 epoch
    # 只有 20 張圖，讀檔不是瓶頸。
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=0)
    dl_va = None if final else DataLoader(va, batch_size=1, shuffle=False,
                                         collate_fn=collate, num_workers=0)

    schedule = None
    if args.mask_loss == "boundary":
        from boundary_loss import AlphaSchedule, patch

        schedule = AlphaSchedule(args.boundary_alpha, args.epochs)
        patch(schedule)

    model = build_model(not args.no_pretrained, args.mask_res).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"{'最終模型（全量）' if final else f'fold {args.fold}'}　enhance={args.enhance}"
          f"　mask={args.mask_res}x{args.mask_res}　loss={args.mask_loss}"
          f"　train {len(tr)} 張 / val {0 if final else len(va)} 張　device={device}")
    out_dir = CKPT / (args.tag or args.enhance)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, ckpt = -1.0, out_dir / ("maskrcnn_final.pt" if final else f"maskrcnn_fold{args.fold}.pt")

    for ep in range(1, args.epochs + 1):
        if schedule is not None:
            schedule.step(ep)
        model.train()
        t0, total = time.time(), 0.0
        for imgs, targets in dl_tr:
            imgs = [i.to(device) for i in imgs]
            tg = [{k: v.to(device) for k, v in t.items() if k != "_name"} for t in targets]
            losses = model(imgs, tg)
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            total += float(loss.detach())
        sched.step()

        line = f"ep {ep:3d}/{args.epochs}　loss {total / len(dl_tr):.4f}　{time.time() - t0:.0f}s"
        if schedule is not None:
            line += f"　α {schedule.value:.2f}"
        if final:
            # 沒有驗證集可挑最佳 epoch，固定訓練完 args.epochs 後存最後一個
            if ep == args.epochs:
                torch.save({"model": model.state_dict(), "fold": -1, "enhance": args.enhance,
                            "mask_res": args.mask_res, "mask_loss": args.mask_loss,
                            "n_train": len(tr), "epochs": args.epochs}, ckpt)
                line += "　← 存檔（最終模型）"
            print(line, flush=True)
            continue
        if ep % args.eval_every == 0 or ep == args.epochs:
            m = evaluate(model, dl_va, device)
            line += (f"　| val Dice {m['dice']:.4f}　recall {m['recall']:.2f}"
                     f"　precision {m['precision']:.2f}　(TP {m['tp']} FP {m['fp']} FN {m['fn']})")
            if m["dice"] > best:
                best = m["dice"]
                torch.save({"model": model.state_dict(), "fold": args.fold, "enhance": args.enhance,
                            "mask_res": args.mask_res, "mask_loss": args.mask_loss,
                            "metrics": m}, ckpt)
                line += "　← 存檔"
        print(line, flush=True)

    print(f"\n{'最終模型' if final else f'最佳 val Dice {best:.4f}'}　權重 → {ckpt}")


if __name__ == "__main__":
    main()
