"""每顆標註牙的牙位類型（門牙／犬齒／小臼齒／大臼齒）：先由形狀給「建議」，再由人工確認。

放在 koi/setA/final/scripts/ 執行；只讀標註與影像，不需要模型。

**建議值只看 GT 遮罩的形狀，沒有用任何牙位標註驗證過，一定要人工逐顆確認。**
用到的形狀特徵（全部是比例，與影像解析度無關）：

    冠寬比   牙冠最寬處 ÷ 牙齒全長        大臼齒最寬、小臼齒次之、門牙與犬齒最窄
    多根     根尖側 10–45% 的橫切面有兩段以上分開的牙根   大臼齒
    尖度     牙冠末端寬度 ÷ 牙冠最寬       犬齒尖（小）、門牙切緣平（大）
    上／下顎 牙冠朝下 = 上顎，牙冠朝上 = 下顎

輸出（在 final/ 底下）：
    tooth_types/<影像>.png   整張原圖（不調對比），每顆牙畫綠色輪廓並在牙齒中央標「編號 類型」；
                             右邊並排未畫線的原圖
    tooth_types.csv          每顆牙一列：建議類型、上下顎、形狀特徵，以及空白的 type 欄
                             （填 I 門牙、C 犬齒、P 小臼齒、M 大臼齒，留白 = 同意建議）

用法（在 koi/setA/final 底下）：
    python3 scripts/tooth_types.py
    python3 scripts/tooth_types.py --only 1 37          # 只做這幾張
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
if ROOT.name != "final":
    sys.exit(f"⚠ 這支腳本必須放在 koi/setA/final/scripts/ 執行，目前的根目錄是 {ROOT}。")
ANN = ROOT / "annotations"
NAME = {"I": "門牙", "C": "犬齒", "P": "小臼齒", "M": "大臼齒"}
FONTS = ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
         "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf",
         "C:/Windows/Fonts/msjh.ttc", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]


def gt_mask(a, h, w):
    m = np.zeros((h, w), np.uint8)
    for poly in a["segmentation"]:
        cv2.fillPoly(m, [np.array(poly, np.int32).reshape(-1, 2)], 1)
    return m.astype(bool)


def straighten(m):
    """把牙齒轉成「長軸直立、牙冠朝上」的點陣：回傳 (row 由根尖到牙冠的二值圖, 指向牙冠的軸)。"""
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    mu = pts.mean(0)
    d = pts - mu
    c = np.array([[np.mean(d[:, 0] ** 2), np.mean(d[:, 0] * d[:, 1])],
                  [np.mean(d[:, 0] * d[:, 1]), np.mean(d[:, 1] ** 2)]])
    ax = np.linalg.eigh(c)[1][:, -1]
    proj = d[:, 0] * ax[0] + d[:, 1] * ax[1]
    lat = -d[:, 0] * ax[1] + d[:, 1] * ax[0]
    # 牙冠端 = 較寬的一半
    if (proj < 0).sum() > 10 and (proj >= 0).sum() > 10 and lat[proj < 0].std() > lat[proj >= 0].std():
        ax, proj, lat = -ax, -proj, -lat
    r = np.round(proj - proj.min()).astype(int)
    q = np.round(lat - lat.min()).astype(int)
    grid = np.zeros((r.max() + 1, q.max() + 1), np.uint8)
    grid[r, q] = 1
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return grid.astype(bool), ax


def features(m):
    g, ax = straighten(m)
    L = g.shape[0]
    width = g.sum(1).astype(float)
    t = np.arange(L) / max(L - 1, 1)
    crown_w = width[t >= 0.6].max()
    tip_w = width[(t >= 0.93) & (t <= 0.97)].mean() if ((t >= 0.93) & (t <= 0.97)).any() else crown_w
    rows = g[(t >= 0.10) & (t <= 0.45)]
    multi = 0
    for row in rows:
        segs = np.flatnonzero(np.diff(np.concatenate([[0], row.astype(int), [0]])))
        runs = list(zip(segs[::2], segs[1::2]))
        runs = [s for s in runs if s[1] - s[0] >= 2]
        gaps = [b[0] - a[1] for a, b in zip(runs, runs[1:])]
        multi += any(gp >= 3 for gp in gaps)
    return {"ratio": crown_w / L, "multi": multi / max(len(rows), 1), "tip": tip_w / max(crown_w, 1),
            "length": L, "jaw": "上顎" if ax[1] > 0 else "下顎"}


def suggest(f):
    """門檻取自教科書平均尺寸（近遠心冠寬 ÷ 全長）：大臼齒約 0.45–0.55、小臼齒約 0.30–0.35、
    犬齒約 0.27–0.30、門牙約 0.26–0.36。門牙、犬齒、小臼齒的冠寬比彼此重疊，所以再用牙冠
    末端的尖度區分犬齒；這三類的建議特別不可靠。"""
    if f["multi"] > 0.30 or f["ratio"] > 0.42:
        return "M"
    if f["tip"] < 0.50:
        return "C"
    return "P" if f["ratio"] > 0.31 else "I"


def font(size):
    for p in FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=[], metavar="IMG", help="只做這些影像（主檔名）")
    args = ap.parse_args()

    out_dir = ROOT / "tooth_types"
    out_dir.mkdir(exist_ok=True)
    rows = []
    sets = [("train", "instances_all.json", ROOT / "images"), ("holdout", "holdout.json", ROOT / "holdout")]
    for split, jf, img_dir in sets:
        coco = json.loads((ANN / jf).read_text(encoding="utf-8"))
        per: dict = {}
        for a in coco["annotations"]:
            if not a.get("iscrowd"):
                per.setdefault(a["image_id"], []).append(a)
        for im in sorted(coco["images"], key=lambda i: i["file_name"]):
            stem = Path(im["file_name"]).stem
            if (args.only and stem not in args.only) or im["id"] not in per:
                continue
            bgr = cv2.imread(str(img_dir / im["file_name"]), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"  ⚠ 讀不到 {img_dir / im['file_name']}")
                continue
            h, w = bgr.shape[:2]
            draw_img = bgr.copy()
            labels = []
            for gi, a in enumerate(per[im["id"]]):
                m = gt_mask(a, h, w)
                if m.sum() < 200:
                    continue
                f = features(m)
                s = suggest(f)
                rows.append({"split": split, "image": im["file_name"], "tooth": gi, "suggest": s,
                             "suggest_name": NAME[s], "jaw": f["jaw"], "crown_ratio": round(f["ratio"], 3),
                             "multi_root": round(f["multi"], 2), "tip": round(f["tip"], 2),
                             "length_px": f["length"], "type": ""})
                cs = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
                cv2.drawContours(draw_img, cs, -1, (0, 200, 0), 2)
                ys, xs = np.nonzero(m)
                labels.append((int(xs.mean()), int(ys.mean()), f"#{gi} {NAME[s]}"))   # 標在牙齒正中央
            # 文字用 PIL 畫（OpenCV 不支援中文）
            pil = Image.fromarray(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
            dr = ImageDraw.Draw(pil)
            ft = font(max(18, h // 45))
            for x, y, text in labels:
                if ft is None:
                    text = text.split()[0]
                bb = dr.textbbox((0, 0), text, font=ft)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
                x = int(min(max(x - tw / 2, 4), w - tw - 6))
                dr.rectangle([x - 4, y - 4, x + tw + 4, y + th + 8], fill=(255, 255, 255))
                dr.text((x, y), text, fill=(0, 0, 0), font=ft)
            left = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            gap = np.full((h, 20, 3), 255, np.uint8)
            cv2.imwrite(str(out_dir / f"{split}_{stem}.png"), np.hstack([left, gap, bgr]))

    if not rows:
        sys.exit("沒有任何結果；請確認 images/ 與 holdout/ 是否存在。")
    out = ROOT / "tooth_types.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as fh:   # utf-8-sig：Excel 開中文不亂碼
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    cnt = Counter((r["split"], r["suggest_name"]) for r in rows)
    print(f"\n共 {len(rows)} 顆牙（建議值，未經驗證）")
    for split in ("train", "holdout"):
        print(f"  {split:<8}" + "　".join(f"{NAME[k]} {cnt[(split, NAME[k])]}" for k in "ICPM"))
    print(f"\n標示圖 → {out_dir}/（每張影像一張，左：標註＋類型，右：原圖）")
    print(f"清單　 → {out}（用 Excel 開，在 type 欄填 I/C/P/M 修正；留白 = 同意建議）")


if __name__ == "__main__":
    main()
