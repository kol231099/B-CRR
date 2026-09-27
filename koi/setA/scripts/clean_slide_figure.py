"""把投影片上的示意圖搬到深色底：X 光原樣保留，白底黑字反相。

投影片的圖有三種雜物：頂端的藍色標題帶、X 光外圍那圈白色圓角外框、以及四周
的白邊。直接整張反相會把 X 光變成負片，所以先把 X 光那一塊框出來原樣保留，
其餘（白底、括號、文字）才做灰階反相。

兩種模式：

    預設        X 光原樣 + 其餘反相成深底亮字（示意圖用，例如 def_crr）
    --photo     整張只留 X 光，外面全部切掉（純照片用，例如 landmarks）

用法：
    py koi/setA/scripts/clean_slide_figure.py in.webp out.webp
    py koi/setA/scripts/clean_slide_figure.py in.webp out.webp --photo --width 900
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

BG = np.array((10, 12, 15), np.float32)     # 頁面深底
INK = 236.0                                 # 反相後的字色
NEAR_WHITE = 246                            # 高於此視為白底


def drop_blue_bar(im: np.ndarray) -> np.ndarray:
    """切掉投影片頂端／底端那條藍色標題帶。"""
    b, g, r = (im[..., i].astype(int) for i in range(3))
    blue = ((b > 110) & (b - r > 45) & (b - g > 45)).mean(1) > 0.35
    h = len(blue)
    t = 0
    while t < h - 1 and blue[t]:
        t += 1
    bt = h - 1
    while bt > t and blue[bt]:
        bt -= 1
    return im[t:bt + 1]


def photo_mask(im: np.ndarray, win: int = 31, dens: float = 0.80):
    """框出 X 光那一塊。

    先用「中間調且低飽和」的密度找到照片核心（線稿只有邊緣符合，密度過不了
    門檻），再沿著非白像素長到照片真正的邊界——直接用密度區域當邊界會偏小，
    照片四周本來就偏暗。
    """
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mx, mn = im.max(2).astype(int), im.min(2).astype(int)
    core = (((g > 40) & (g < 236)) & ((mx - mn) < 40)).astype(np.float32)
    blob = cv2.morphologyEx((cv2.boxFilter(core, -1, (win, win)) > dens).astype(np.uint8),
                            cv2.MORPH_CLOSE, np.ones((win, win), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(blob, 8)
    if n < 2:
        return None
    seed = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))

    # 沿非白區域長到照片邊界
    solid = cv2.morphologyEx((g < NEAR_WHITE).astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(solid, 8)
    ys, xs = np.nonzero(lab == seed)
    ids, counts = np.unique(lab2[ys, xs], return_counts=True)
    ok = [(c, i) for i, c in zip(ids, counts) if i != 0]
    if not ok:
        return (lab == seed), tuple(int(v) for v in st[seed, :4])
    grown = max(ok)[1]
    # 回傳遮罩而不是外框：白色圓角框在外框裡、卻不在遮罩裡，用遮罩貼回去
    # 才不會把白框一起留下來。
    m = cv2.erode((lab2 == grown).astype(np.uint8), np.ones((3, 3), np.uint8))
    return m.astype(bool), tuple(int(v) for v in st2[grown, :4])


def rounded(shape, rect, frac: float = 0.06, ss: int = 4) -> np.ndarray:
    """X 光那一塊的圓角遮罩（0–1）。

    投影片上兩張圖的收邊不一致——一張原本就是圓角、另一張是直角。與其遷就
    來源，統一在這裡收一次邊，兩張就會長得一樣。先放大 ss 倍畫再縮回來，
    邊緣才不會有鋸齒。
    """
    h, w = shape
    x, y, rw, rh = rect
    r = max(6, int(round(frac * min(rw, rh))))
    big = np.zeros((h * ss, w * ss), np.uint8)
    X, Y, W, H, R = (v * ss for v in (x, y, rw, rh, r))
    cv2.rectangle(big, (X + R, Y), (X + W - R, Y + H), 255, cv2.FILLED)
    cv2.rectangle(big, (X, Y + R), (X + W, Y + H - R), 255, cv2.FILLED)
    for cx, cy in ((X + R, Y + R), (X + W - R, Y + R),
                   (X + R, Y + H - R), (X + W - R, Y + H - R)):
        cv2.circle(big, (cx, cy), R, 255, cv2.FILLED, cv2.LINE_AA)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def invert_outside(im: np.ndarray, mask: np.ndarray, rect) -> np.ndarray:
    """遮罩外做灰階反相，遮罩內（X 光）原樣保留，四角統一收邊。"""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inv = (255.0 - g) / 255.0
    bg = BG[None, None, :] * (1 - inv[..., None]) + INK * inv[..., None]
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2)
    a = np.clip(np.minimum(soft, rounded(im.shape[:2], rect)), 0, 1)[..., None]
    return (im.astype(np.float32) * a + bg * (1 - a)).astype(np.uint8)


def trim(im: np.ndarray, keep: np.ndarray, pad: int = 6) -> np.ndarray:
    """依 keep 遮罩裁掉四周多餘的部分，再留一點邊。"""
    ys, xs = np.nonzero(keep)
    h, w = im.shape[:2]
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    return im[y0:y1, x0:x1]


def peel_bright(im: np.ndarray, thr: int = 200, frac: float = 0.5) -> np.ndarray:
    """由外往內剝掉整列／整行偏亮的邊。

    X 光本身是中低灰階，貼齊照片邊界後若還有一整條接近白的，那就是投影片的
    底色漏進來，不是影像內容。
    """
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    # 內部中位數當基準；整列明顯比它亮就是邊界殘影（含反鋸齒那一兩像素）
    base = float(np.median(g[h // 5:h * 4 // 5, w // 5:w * 4 // 5]))
    rows = ((g > thr).mean(1) >= frac) | (g.mean(1) > base + 25)
    cols = ((g > thr).mean(0) >= frac) | (g.mean(0) > base + 25)
    t = 0
    while t < h - 1 and rows[t]:
        t += 1
    b = h - 1
    while b > t and rows[b]:
        b -= 1
    l = 0
    while l < w - 1 and cols[l]:
        l += 1
    r = w - 1
    while r > l and cols[r]:
        r -= 1
    return im[t:b + 1, l:r + 1]


def clean(im: np.ndarray, photo_only: bool) -> np.ndarray:
    im = drop_blue_bar(im)
    found = photo_mask(im)
    if found is None:
        raise SystemExit("找不到 X 光的位置")
    mask, (x, y, w, h) = found
    if photo_only:
        return peel_bright(im[y:y + h, x:x + w])

    out = invert_outside(im, mask, (x, y, w, h))
    # 反相後「有東西」= 比底色亮；據此把四周的白邊裁掉
    keep = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) > 30
    keep[y:y + h, x:x + w] = True
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8)).astype(bool)
    return trim(out, keep)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--photo", action="store_true", help="只留 X 光，其餘切掉")
    ap.add_argument("--width", type=int, default=0, help="輸出寬度，0 為不縮放")
    args = ap.parse_args()

    im = cv2.imread(args.src)
    if im is None:
        raise SystemExit(f"讀不到 {args.src}")
    out = clean(im, args.photo)
    if args.width and out.shape[1] != args.width:
        h = int(out.shape[0] * args.width / out.shape[1])
        out = cv2.resize(out, (args.width, h), interpolation=cv2.INTER_AREA)
    q = [cv2.IMWRITE_WEBP_QUALITY, 92] if args.dst.endswith(".webp") else []
    cv2.imwrite(args.dst, out, q)
    print(f"{args.dst}  {out.shape[1]}x{out.shape[0]}")


if __name__ == "__main__":
    main()
