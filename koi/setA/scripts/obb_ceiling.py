"""量 HBB 與 OBB 兩種裁切表示法的「資訊上限」，不訓練任何模型。

第二階段吃的是 512x256 的 crop，輸出再貼回原圖。就算網路完美無誤地輸出
「GT 在該 crop 裡的樣子」，貼回去也會有誤差——降採樣、升採樣，OBB 還多了
轉正與轉回兩次重採樣。這個誤差是該表示法的天花板，任何模型都不可能超過。

HBB：crop → resize 512x256 → resize 回 → 貼回
OBB：minAreaRect 轉正 → crop → resize → resize 回 → 轉回 → 貼回

同時報裁切面積與填充率，檢查轉正確實有把牙擺直。

用法：
    py scripts/obb_ceiling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_crops import crop_box  # noqa: E402
from metrics import boundary_iou, hd95  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "annotations"
SIZE = (512, 256)  # (高, 寬)，與 train_seg2.py 一致
PAD = 0.2


def dice(a, b):
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def roundtrip_hbb(gt, bbox, w, h):
    x0, y0, x1, y1 = crop_box(bbox, PAD, w, h)
    sub = gt[y0:y1, x0:x1].astype(np.float32)
    down = cv2.resize(sub, SIZE[::-1], interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0.5
    out = np.zeros_like(gt)
    out[y0:y1, x0:x1] = up
    return out, (x1 - x0) * (y1 - y0)


def obb_transform(gt, w, h):
    """回傳 (M, 裁切寬, 裁切高, 傾角)；M 把原圖座標轉成「牙擺正」的 crop 座標。"""
    cnt = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_NONE)[0]
    cnt = max(cnt, key=cv2.contourArea)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(cnt)
    if rw > rh:                      # 讓牙是「高」的那一邊，與 512x256 的長寬比一致
        rw, rh, ang = rh, rw, ang + 90
    tilt = ((ang + 90) % 180) - 90   # 折到 [-90, 90)，0 = 已經是直的
    cw = max(8, int(round(rw * (1 + 2 * PAD))))
    ch = max(8, int(round(rh * (1 + 2 * PAD))))
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    M[0, 2] += cw / 2 - cx
    M[1, 2] += ch / 2 - cy
    return M, cw, ch, tilt


def roundtrip_obb(gt, w, h):
    M, cw, ch, tilt = obb_transform(gt, w, h)
    sub = cv2.warpAffine(gt.astype(np.float32), M, (cw, ch), flags=cv2.INTER_LINEAR)
    down = cv2.resize(sub, SIZE[::-1], interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (cw, ch), interpolation=cv2.INTER_LINEAR)
    back = cv2.warpAffine(up, cv2.invertAffineTransform(M), (w, h),
                          flags=cv2.INTER_LINEAR) > 0.5
    fill = (sub > 0.5).sum() / max(cw * ch, 1)
    return back, cw * ch, fill, abs(tilt)


def main() -> None:
    rows = []
    for f in range(5):
        coco = json.loads((ANN / f"fold{f}_val.json").read_text(encoding="utf-8"))
        imgs = {i["id"]: i for i in coco["images"]}
        for a in coco["annotations"]:
            if a.get("iscrowd"):
                continue
            im = imgs[a["image_id"]]
            h, w = im["height"], im["width"]
            gt = cv2.fillPoly(np.zeros((h, w), np.uint8),
                              [np.array(a["segmentation"][0], np.int32).reshape(-1, 2)],
                              1).astype(bool)
            if gt.sum() < 50:
                continue
            hb, ha = roundtrip_hbb(gt, a["bbox"], w, h)
            ob, oa, ofill, tilt = roundtrip_obb(gt, w, h)
            x0, y0, x1, y1 = crop_box(a["bbox"], PAD, w, h)
            rows.append({
                "tilt": tilt, "h_area": ha, "o_area": oa,
                "h_fill": gt[y0:y1, x0:x1].sum() / max(ha, 1), "o_fill": ofill,
                "h_dice": dice(hb, gt), "o_dice": dice(ob, gt),
                "h_biou": boundary_iou(hb, gt), "o_biou": boundary_iou(ob, gt),
                "h_hd95": hd95(hb, gt), "o_hd95": hd95(ob, gt),
            })

    def med(k):
        return float(np.median([r[k] for r in rows]))

    t = np.array([r["tilt"] for r in rows])
    print(f"\n{'=' * 72}\n牙齒傾角（GT minAreaRect）　n={len(rows)}\n{'=' * 72}")
    print(f"  中位 {np.median(t):.1f}°　平均 {t.mean():.1f}°　"
          f">15° {100 * (t > 15).mean():.1f}%　>25° {100 * (t > 25).mean():.1f}%")

    print(f"\n{'=' * 72}\n裁切幾何\n{'=' * 72}")
    print(f"  裁切面積中位　HBB {med('h_area'):>9.0f} px²　OBB {med('o_area'):>9.0f} px²　"
          f"縮小 {100 * (1 - med('o_area') / med('h_area')):.1f}%")
    print(f"  填充率中位　　HBB {100 * med('h_fill'):>6.1f}%　　OBB {100 * med('o_fill'):>6.1f}%")

    print(f"\n{'=' * 72}\n來回轉換上限（完美模型也只能做到這樣）\n{'=' * 72}")
    print(f"  {'':10}{'Dice':>10}{'B-IoU':>10}{'HD95':>10}")
    for lab, p in (("HBB", "h"), ("OBB", "o")):
        print(f"  {lab:<10}{med(p + '_dice'):>10.4f}{med(p + '_biou'):>10.4f}"
              f"{med(p + '_hd95'):>10.2f}")

    print(f"\n  對照：兩階段實際做到的　B-IoU 0.6570　HD95 12.01")
    print(f"  對照：Mask R-CNN 實際做到的　B-IoU 0.6510　HD95 10.96")


if __name__ == "__main__":
    main()
