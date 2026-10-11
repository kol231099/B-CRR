"""解析度天花板：模型完美時，光是「縮放再貼回」就會造成多少誤差。不訓練任何模型。

放在 koi/setA/final/scripts/ 執行。把每顆 GT 遮罩當成完美的模型輸出，走一遍各管線的
幾何流程，再與 GT 本身比 HD95 / Dice（同 metrics.py 的定義）：

    HRNet 512×256    GT 斜框 + pad 0.2 → 轉正裁切 → 縮成 512×256 → 放回 → 轉回原圖
    HRNet 768×384    同上，輸入 768×384
    HRNet 1024×512   同上，輸入 1024×512
    Mask R-CNN 28    水平框 → 縮成 28×28（遮罩頭解析度）→ 放大貼回
    Mask R-CNN 56    同上，56×56
    融合 512 + 28    兩者機率圖平均後切 0.5（對應現行融合的幾何上限）
    融合 768 + 28    換成 768×384 的 HRNet

判讀：模型實際的 HD95 約 10 px。某一列的天花板若只有 1–2 px，代表那個解析度不是
瓶頸，提高它幾乎沒有好處；天花板之間的差距，就是換解析度最多能拿到的改善。

用法（在 koi/setA/final 底下）：
    python3 scripts/final_res_ceiling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_crops_obb import obb_of, warp_of  # noqa: E402
from metrics import hd95  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if ROOT.name != "final":
    sys.exit(f"⚠ 這支腳本必須放在 koi/setA/final/scripts/ 執行，目前的根目錄是 {ROOT}。")
ANN, PAD = ROOT / "annotations", 0.2


def obb_round(gt, size):
    """回傳轉回原圖的機率圖（未二值化）。"""
    h, w = gt.shape
    M, cw, ch = warp_of(*obb_of(gt.astype(np.uint8)), PAD)
    sub = cv2.warpAffine(gt.astype(np.float32), M, (cw, ch), flags=cv2.INTER_LINEAR)
    down = cv2.resize(sub, size[::-1], interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (cw, ch), interpolation=cv2.INTER_LINEAR)
    return cv2.warpAffine(up, cv2.invertAffineTransform(M), (w, h), flags=cv2.INTER_LINEAR)


def mrcnn_round(gt, res):
    """Mask R-CNN 遮罩頭：框內縮成 res×res 再雙線性放大貼回（torchvision paste 的近似）。"""
    ys, xs = np.nonzero(gt)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = gt[y0:y1, x0:x1].astype(np.float32)
    down = cv2.resize(sub, (res, res), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    out = np.zeros(gt.shape, np.float32)
    out[y0:y1, x0:x1] = up
    return out


def roi(gt, margin=24):
    ys, xs = np.nonzero(gt)
    return (slice(max(ys.min() - margin, 0), ys.max() + margin + 1),
            slice(max(xs.min() - margin, 0), xs.max() + margin + 1))


def main():
    d = json.loads((ANN / "instances_all.json").read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in d["images"]}
    rows = []
    for a in d["annotations"]:
        if a.get("iscrowd"):
            continue
        im = imgs[a["image_id"]]
        gt = np.zeros((im["height"], im["width"]), np.uint8)
        for poly in a["segmentation"]:
            cv2.fillPoly(gt, [np.array(poly, np.int32).reshape(-1, 2)], 1)
        gt = gt.astype(bool)
        if gt.sum() < 50:
            continue
        p512, p768, p1024 = (obb_round(gt, s) for s in ((512, 256), (768, 384), (1024, 512)))
        m28, m56 = mrcnn_round(gt, 28), mrcnn_round(gt, 56)
        preds = {"HRNet 512×256": p512 > 0.5, "HRNet 768×384": p768 > 0.5,
                 "HRNet 1024×512": p1024 > 0.5, "Mask R-CNN 28": m28 > 0.5,
                 "Mask R-CNN 56": m56 > 0.5, "融合 512 + 28": (p512 + m28) / 2 > 0.5,
                 "融合 768 + 28": (p768 + m28) / 2 > 0.5}
        sl = roi(gt)
        g = gt[sl]
        r = {}
        for k, p in preds.items():
            p = p[sl]
            r[k] = (hd95(p, g), 2 * (p & g).sum() / (p.sum() + g.sum()))
        rows.append(r)

    print(f"\n{len(rows)} 顆 GT 牙（final/annotations/instances_all.json）")
    print(f"\n  {'流程':<18}{'HD95 中位':>10}{'HD95 平均':>10}{'HD95 P95':>10}{'Dice 中位':>11}")
    for k in rows[0]:
        h = np.array([r[k][0] for r in rows])
        dc = np.array([r[k][1] for r in rows])
        print(f"  {k:<18}{np.nanmedian(h):>10.2f}{np.nanmean(h):>10.2f}"
              f"{np.nanpercentile(h, 95):>10.2f}{np.median(dc):>11.4f}")
    print("\n  對照：現行融合在 OOF 上實際 HD95 中位 10.00、平均 12.23")


if __name__ == "__main__":
    main()
