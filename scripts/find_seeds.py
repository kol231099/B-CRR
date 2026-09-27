"""找種子點：每顆牙齒給一個大致的中心位置，供後續模板配準使用。

做法
----
1. Otsu 二值化取出亮區
2. 距離變換——每個前景像素到背景的距離，牙齒中心的值最大
3. 取距離變換的**局部極大值**當種子

用距離變換的極大值而非隨機取樣，有三個好處：位置落在牙齒**中心**而非邊緣；
即使相鄰牙齒連成同一個連通區域，每顆仍各有一個極大值，不必先把它們分開；
而且極大值本身就是該處的**內切圓半徑**，約等於牙齒半寬，可直接當作後續
配準的初始尺度。

這一步只求**大致位置**，不求準確——真正的分割由模板配準完成。

用法：
    py scripts/find_seeds.py shape_prior_seg_test
    py scripts/find_seeds.py shape_prior_seg_test --only 13 1 26 50 41 4
    py scripts/find_seeds.py shape_prior_seg_test -o seeds.png --only 13 1 26 50 41 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.labelme_io import make_figure, save_or_show  # noqa: E402


def binarize(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Otsu 二值化，回傳 (二值遮罩, 使用的門檻值)。"""
    threshold, binary = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.uint8), threshold


def find_seeds(binary: np.ndarray, min_radius: float, max_radius: float, spacing: float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由二值遮罩找出種子點。

    輸入：二值遮罩、內切圓半徑的上下限、兩個種子之間的最小間距。
    輸出：(距離變換, 種子座標 (N,2) 的 (x,y), 各種子的距離值)。

    局部極大值的判定用形態學膨脹：某像素若等於它鄰域內的最大值，即為極大值。
    鄰域大小就是 spacing，所以間距的限制在這一步就完成了，不必事後再篩。

    **半徑上限是必要的**：Otsu 的前景是「牙齒 ∪ 齒槽骨」，兩者相連且骨頭那片
    又寬又大，距離變換的最大值會落在骨頭中央而非牙齒（實測骨頭給出 250~330 px，
    而模板的牙齒半寬只有 139 px）。用一個略大於牙齒半寬的上限即可濾掉。

    種子不必完美：**寧可多抓、不要漏抓**。落在骨頭上的假種子，後續模板配準的
    品質判定會把它們刷掉；真正的牙齒漏掉了才無法補救。
    """
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    size = max(3, int(spacing) | 1)  # 必須是奇數
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    peaks = ((distance >= cv2.dilate(distance, kernel))
             & (distance >= min_radius) & (distance <= max_radius))

    ys, xs = np.nonzero(peaks)
    seeds = np.column_stack([xs, ys]).astype(float)
    values = distance[ys, xs]

    order = np.argsort(values)[::-1]  # 由大到小，先處理最有把握的
    return distance, seeds[order], values[order]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", help="含有 .jpg 的資料夾")
    parser.add_argument("--only", nargs="+", help="只處理這幾張（檔名去掉副檔名）")
    parser.add_argument("--min-radius", type=float, default=30.0,
                        help="種子的最小內切圓半徑（像素），預設 30")
    parser.add_argument("--max-radius", type=float, default=170.0,
                        help="種子的最大內切圓半徑（像素），預設 170（模板半寬 139 的 1.2 倍）")
    parser.add_argument("--spacing", type=float, default=140.0,
                        help="兩種子的最小間距（像素），預設 140（約一個牙齒半寬）")
    parser.add_argument("-o", "--out", help="存檔到此路徑，不開視窗")
    args = parser.parse_args()

    folder = Path(args.folder)
    files = sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if args.only:
        wanted = set(args.only)
        files = [f for f in files if f.stem in wanted]
    if not files:
        raise SystemExit(f"在 {folder} 找不到影像")

    plt, fig, axes = make_figure(len(files), 4, (16, 4.2 * len(files)), args.out)

    for row, path in enumerate(files):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        binary, threshold = binarize(img)
        distance, seeds, values = find_seeds(binary, args.min_radius, args.max_radius,
                                             args.spacing)

        print(f"{path.name}  {img.shape[1]}x{img.shape[0]}　Otsu 門檻 {threshold:.0f}"
              f"　前景 {100 * binary.mean():.0f}%　種子 {len(seeds)} 個"
              f"　半徑 {values.min():.0f}~{values.max():.0f}" if len(seeds)
              else f"{path.name}  無種子")

        axes[row][0].imshow(img, cmap="gray")
        axes[row][0].set_title(f"{path.stem}　原圖", fontsize=9)

        axes[row][1].imshow(binary, cmap="gray")
        axes[row][1].set_title(f"Otsu（門檻 {threshold:.0f}，前景 {100 * binary.mean():.0f}%）",
                               fontsize=9)

        axes[row][2].imshow(distance, cmap="viridis")
        axes[row][2].set_title(f"距離變換（最大 {distance.max():.0f} px）", fontsize=9)

        axes[row][3].imshow(img, cmap="gray")
        for (x, y), v in zip(seeds, values):
            axes[row][3].add_patch(plt.Circle((x, y), v, fill=False,
                                              color="#00e5ff", linewidth=0.7))
            axes[row][3].plot(x, y, "o", color="#ff3b30", markersize=3,
                              markeredgecolor="black", markeredgewidth=0.5)
        axes[row][3].set_title(f"種子 {len(seeds)} 個（圓 = 內切圓）", fontsize=9)

        for col in range(4):
            axes[row][col].axis("off")

    save_or_show(plt, args.out, dpi=100)


if __name__ == "__main__":
    main()
