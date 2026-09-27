"""沿法線精修輪廓：把粗定位後的模板貼到牙齒真正的邊緣上。

粗定位（`align_template.py`）只有平移、旋轉、水平垂直縮放五個自由度，形狀
仍然是模板本身。這一步讓輪廓上的每個點各自沿**法線**移動到影像上的真實邊緣，
形狀才會跟著這顆牙齒走。

做法
----
1. 對每個輪廓點取**向外法線**（相鄰點的切線轉 90 度，並校正成朝外）
2. 沿法線在 ±search_px 內取樣灰階，找**沿外向方向下降最劇烈**的位置——
   那就是牙齒表面（內亮外暗）。牙根處的牙周膜間隙是一條細暗線，會讓這個
   下降更明顯，不會造成干擾
3. 把所有位移量沿輪廓做**環狀平滑**，再以阻尼係數施加，重複數輪

第 3 步是形狀先驗在這裡的作用。單點各自吸附會讓輪廓變成鋸齒，在低對比的
牙根段更會亂跑；平滑強迫相鄰點一起移動，等同於要求「形狀不能突然變得不像
牙齒」。這正是 ASM 的搜尋步驟，但**用平滑取代訓練出來的形狀模型**，因此
不需要任何訓練資料。

位移量的統計（中位數、大位移的比例）就是擬合品質的指標。

用法：
    py scripts/refine_contour.py shape_prior_seg_test --only 13 41
    py scripts/refine_contour.py shape_prior_seg_test -o refined.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.align_template import align_image  # noqa: E402
from scripts.labelme_io import make_figure, save_or_show  # noqa: E402
from scripts.shape_template import build_template  # noqa: E402

# 沿法線搜尋的距離（像素）。太大會跳到鄰牙或骨小樑上，太小則修不動。
SEARCH_PX = 18

# 環狀平滑的視窗（輪廓點數）。模板共 200 點，視窗 15 約佔輪廓的 7.5%。
SMOOTH_WINDOW = 15

# 每輪只走位移量的這個比例，避免一次衝過頭；配合多輪逐步收斂。
DAMPING = 0.5

ROUNDS = 5


def outward_normals(contour: np.ndarray) -> np.ndarray:
    """算每個輪廓點的單位向外法線。

    切線取前後兩點的差（中央差分，比單邊差分穩定），轉 90 度得到法線，
    再用「該點相對形心的方向」決定要朝哪一邊。牙齒輪廓是凸性為主的形狀，
    用形心定向足夠可靠，不必判斷輪廓的環繞方向。
    """
    tangent = np.roll(contour, -1, axis=0) - np.roll(contour, 1, axis=0)
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9

    outward = contour - contour.mean(axis=0)
    flip = np.sign((normal * outward).sum(axis=1))
    flip[flip == 0] = 1.0
    return normal * flip[:, None]


def sample_along(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """雙線性取樣，落在影像外的點回傳 nan。"""
    x, y = points[..., 0], points[..., 1]
    inside = (x >= 0) & (y >= 0) & (x < img.shape[1] - 1) & (y < img.shape[0] - 1)

    x0 = np.clip(np.floor(x), 0, img.shape[1] - 2).astype(int)
    y0 = np.clip(np.floor(y), 0, img.shape[0] - 2).astype(int)
    fx, fy = x - x0, y - y0

    value = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
             + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return np.where(inside, value, np.nan)


def circular_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """沿封閉輪廓做環狀移動平均。"""
    kernel = np.ones(window) / window
    padded = np.concatenate([values[-window:], values, values[:window]])
    return np.convolve(padded, kernel, mode="same")[window:-window]


def best_edge_offset(img: np.ndarray, contour: np.ndarray, normals: np.ndarray,
                     search_px: int) -> np.ndarray:
    """對每個點沿法線找邊緣，回傳應該移動的距離（正值代表往外）。

    在 [-search, +search] 上取樣成一條剖面，取沿外向的一階差分，最負的位置
    就是「由亮轉暗」最劇烈處，也就是牙齒表面。差分用中央差分（相隔兩格）而非
    相鄰兩格，可壓掉單像素雜訊。
    """
    offsets = np.arange(-search_px, search_px + 1, dtype=float)
    # (點數, 取樣數, 2)
    rays = contour[:, None, :] + normals[:, None, :] * offsets[None, :, None]
    profile = sample_along(img, rays)

    gradient = np.full_like(profile, np.nan)
    gradient[:, 1:-1] = profile[:, 2:] - profile[:, :-2]

    # 全是 nan（整條剖面在影像外）時保持不動
    all_nan = np.all(np.isnan(gradient), axis=1)
    gradient[all_nan] = 0.0
    gradient = np.nan_to_num(gradient, nan=0.0)

    return offsets[np.argmin(gradient, axis=1)]


def refine(img: np.ndarray, contour: np.ndarray, search_px: int = SEARCH_PX,
           window: int = SMOOTH_WINDOW, damping: float = DAMPING,
           rounds: int = ROUNDS) -> tuple[np.ndarray, np.ndarray]:
    """反覆沿法線精修輪廓。

    輸入：灰階影像、粗定位後的輪廓。
    輸出：(精修後的輪廓, 每個點相對起始位置的總位移量)。
    """
    blurred = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2.0)
    start = contour.copy()
    current = contour.copy()

    for _ in range(rounds):
        normals = outward_normals(current)
        step = best_edge_offset(blurred, current, normals, search_px)
        step = circular_smooth(step, window) * damping
        current = current + normals * step[:, None]

    return current, np.linalg.norm(current - start, axis=1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", help="含有 .jpg 的資料夾")
    parser.add_argument("--only", nargs="+", help="只處理這幾張")
    parser.add_argument("--template", default="labeled_PA/13.json", help="標準形狀的來源")
    parser.add_argument("--search", type=int, default=SEARCH_PX,
                        help=f"沿法線搜尋的距離（像素），預設 {SEARCH_PX}")
    parser.add_argument("--window", type=int, default=SMOOTH_WINDOW,
                        help=f"環狀平滑的視窗，預設 {SMOOTH_WINDOW}")
    parser.add_argument("--rounds", type=int, default=ROUNDS,
                        help=f"精修輪數，預設 {ROUNDS}")
    parser.add_argument("--top", type=int, default=1,
                        help="每張只精修得分最高的前 N 個擺放，預設 1")
    parser.add_argument("-o", "--out", help="存檔到此路徑，不開視窗")
    args = parser.parse_args()

    folder = Path(args.folder)
    files = sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if args.only:
        files = [f for f in files if f.stem in set(args.only)]
    if not files:
        raise SystemExit(f"在 {folder} 找不到影像")

    template = build_template(args.template)
    cols = min(5, len(files))
    rows = (len(files) + cols - 1) // cols
    plt, fig, axes = make_figure(rows, cols, (3.6 * cols, 5.2 * rows), args.out)

    for index, path in enumerate(files):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        placements = align_image(img, template, 30.0, 170.0, 140.0)
        placements = sorted(placements, key=lambda p: -p.score)[:args.top]

        ax = axes[index // cols][index % cols]
        ax.imshow(img, cmap="gray")

        for placement in placements:
            coarse = template.place(*placement.params)
            refined, shift = refine(img, coarse, args.search, args.window,
                                    rounds=args.rounds)

            print(f"{path.name}  得分 {placement.score:.1f}"
                  f"　位移 中位數 {np.median(shift):.1f} px"
                  f"　90 百分位 {np.percentile(shift, 90):.1f} px"
                  f"　最大 {shift.max():.1f} px")

            closed_c = np.vstack([coarse, coarse[:1]])
            closed_r = np.vstack([refined, refined[:1]])
            ax.plot(closed_c[:, 0], closed_c[:, 1], "--", color="#ff9f45", linewidth=0.8)
            ax.plot(closed_r[:, 0], closed_r[:, 1], "-", color="#00e5ff", linewidth=1.2)

        ax.set_title(f"{path.stem}　橘虛線=粗定位　青=精修", fontsize=8)
        ax.axis("off")

    for blank in range(len(files), rows * cols):
        axes[blank // cols][blank % cols].axis("off")

    save_or_show(plt, args.out, dpi=110)


if __name__ == "__main__":
    main()
