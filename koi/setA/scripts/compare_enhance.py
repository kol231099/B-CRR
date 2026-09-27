"""對測試集每張圖產生所有增強版本的對照圖，並量化排名哪個真的讓邊界更明顯。

視覺對照
--------
每張輸入產生一張 koi/enhance_vis/<圖名>.png，格狀排列所有方法，標題附上該版本
的全圖平均梯度。

量化排名（--rank）
------------------
光看「梯度變高」會被騙：銳化與強 CLAHE 會把**雜訊**一起放大，梯度上升但邊界並
沒有更清楚。所以在有標註的 25 張上計算真正該看的指標：

    邊界梯度      沿人工標註輪廓量到的梯度 → 訊號
    背景梯度      牙齒內部與外部隨機取樣點的梯度 → 雜訊
    CNR           兩者的比值。**只有這個比值上升，才代表邊界相對於雜訊更突出。**

同時分開報「牙冠段」與「根尖段」，因為問題只出在根尖段（實測對比只有牙冠的
46%），一個在牙冠有效、在根尖無效的方法對這個任務沒有價值。

用法：
    py koi/scripts/compare_enhance.py            # 產生 20 張對照圖
    py koi/scripts/compare_enhance.py --rank     # 加上量化排名
    py koi/scripts/compare_enhance.py --only 121
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enhance import METHODS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTSET, IMAGES, ANN = ROOT / "testset", ROOT / "images", ROOT / "annotations"
OUT = ROOT / "enhance_vis"
COLS = 4
TILE_W = 460


def grad(g: np.ndarray) -> np.ndarray:
    f = g.astype(np.float32)
    return np.sqrt(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=5) ** 2
                   + cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=5) ** 2)


def montage(path: Path) -> None:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    tiles = []
    for name, fn in METHODS.items():
        out = fn(g)
        v = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        v = cv2.resize(v, (TILE_W, int(TILE_W * v.shape[0] / v.shape[1])))
        b = np.zeros((46, v.shape[1], 3), np.uint8)
        cv2.putText(b, f"{name}  g={grad(out).mean():.0f}", (10, 32), 0, 0.75, (255, 255, 255), 2)
        tiles.append(np.vstack([b, v]))

    H = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((H - t.shape[0], t.shape[1], 3), np.uint8)]) for t in tiles]
    rows = []
    for i in range(0, len(tiles), COLS):
        row = tiles[i:i + COLS]
        while len(row) < COLS:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{path.stem}.png"), np.vstack(rows))


def rank() -> None:
    """在有標註的影像上，量每個方法的邊界訊噪比，分牙冠段與根尖段。"""
    data = json.loads((ANN / "instances_all.json").read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in data["images"]}
    rng = np.random.default_rng(0)
    acc = {k: {"crown": [], "apex": [], "bg": []} for k in METHODS}

    for a in data["annotations"]:
        im = imgs[a["image_id"]]
        g0 = cv2.imread(str(IMAGES / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        poly = np.array(a["segmentation"][0], np.int32).reshape(-1, 2)
        m = cv2.fillPoly(np.zeros(g0.shape, np.uint8), [poly], 1)
        c = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
                key=cv2.contourArea).reshape(-1, 2).astype(np.float64)

        # 長軸方向：用 float64，float32 在十萬點以上的共變異數會溢位
        pts = np.column_stack(np.nonzero(m)[::-1]).astype(np.float64)
        mu = pts.mean(0)
        ax = np.linalg.eigh(np.cov((pts - mu).T))[1][:, -1]
        perp = np.array([-ax[1], ax[0]])
        proj = (pts - mu) @ ax
        shape_ratio = lambda h: (((h - mu) @ perp).var()) / max((((h - mu) @ ax).var()), 1e-6)
        if shape_ratio(pts[proj < 0]) > shape_ratio(pts[proj >= 0]):
            ax = -ax                                   # 讓 ax 指向牙冠
        t = (c - mu) @ ax
        t = (t - t.min()) / max(np.ptp(t), 1e-6)       # 0 = 根尖端, 1 = 牙冠端

        # 背景取樣點：離輪廓夠遠的牙內與牙外像素
        far = cv2.distanceTransform(1 - cv2.dilate(
            cv2.Canny(m * 255, 50, 150), np.ones((3, 3), np.uint8), iterations=6) // 255,
            cv2.DIST_L2, 3) > 25
        ys, xs = np.nonzero(far)
        if len(xs) > 400:
            sel = rng.choice(len(xs), 400, replace=False)
            bg_pts = (ys[sel], xs[sel])
        else:
            bg_pts = (ys, xs)

        yy = np.clip(c[:, 1], 0, g0.shape[0] - 1).astype(int)
        xx = np.clip(c[:, 0], 0, g0.shape[1] - 1).astype(int)
        for name, fn in METHODS.items():
            gm = grad(fn(g0))
            v = gm[yy, xx]
            if (t < 0.3).any():
                acc[name]["apex"].append(float(v[t < 0.3].mean()))
            if (t > 0.7).any():
                acc[name]["crown"].append(float(v[t > 0.7].mean()))
            acc[name]["bg"].append(float(gm[bg_pts].mean()))

    print(f"{'方法':<18}{'牙冠段 CNR':>13}{'根尖段 CNR':>13}{'根尖/牙冠':>12}")
    print("-" * 58)
    rows = []
    for name, v in acc.items():
        bg = np.mean(v["bg"])
        cr, ap = np.mean(v["crown"]) / bg, np.mean(v["apex"]) / bg
        rows.append((name, cr, ap))
    base = dict((r[0], r) for r in rows)["original"]
    for name, cr, ap in sorted(rows, key=lambda r: -r[2]):
        mark = ""
        if name != "original":
            mark = f"  ({ap / base[2] - 1:+.0%} vs original)"
        print(f"{name:<18}{cr:>13.2f}{ap:>13.2f}{ap / cr:>12.2f}{mark}")
    print("\nCNR = 邊界梯度 / 背景梯度。只有這個比值上升才代表邊界相對雜訊更突出；")
    print("單看梯度絕對值會被雜訊放大所騙。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="只做指定影像，逗號分隔")
    ap.add_argument("--rank", action="store_true", help="加做量化排名")
    args = ap.parse_args()

    wanted = {f"{s.strip()}.jpg" for s in args.only.split(",") if s.strip()}
    files = sorted(TESTSET.glob("*.jpg"), key=lambda p: int(p.stem))
    files = [f for f in files if not wanted or f.name in wanted]
    for f in files:
        montage(f)
        print(f"  {f.name} → enhance_vis/{f.stem}.png")
    print(f"\n完成 {len(files)} 張，{len(METHODS)} 種版本 → {OUT}")

    if args.rank:
        print()
        rank()


if __name__ == "__main__":
    main()
