"""從殘差影像找出邊緣嵴的 A、B 兩點。

想法
----
牙冠上半部的灰階主要被「射線穿過的厚度」支配（實測相關係數 +0.65~+0.82），
先把這個幾何效應扣掉，剩下的殘差才反映材質差異。扣除後左右兩側的鄰接面
會浮出一條**殘差為正**的帶狀區域——射線在鄰接面是沿切線方向穿過琺瑯質層，
路徑中的琺瑯質比「均質牙齒」模型預期的多，所以偏亮。

A、B 就取這兩條帶狀區域的**上端**（最靠咬合面那一端），跟先前用琺瑯質
遮罩找 A/B 的邏輯相同，只是把人工標註換成殘差。

這裡不試圖把牙尖或琺瑯質「分離」出來——二維投影上每個像素都是各組織的
混合，那種分離並不成立。此處只需要一個可重複定義、且上端位置與邊緣嵴
相關的特徵。這個相關性**尚未驗證**，需要人工標註的 A、B 才能確認。

用法：
    py scripts/find_ridge.py labeled_PA
    py scripts/find_ridge.py labeled_PA --threshold 5
    py scripts/find_ridge.py labeled_PA --cut-at 0.25 -o ridge.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.find_cej import cej_points, upper_crown_region  # noqa: E402
from scripts.find_axis_raw import ToothFrame, fit_axis  # noqa: E402
from scripts.labelme_io import (  # noqa: E402
    LABEL_TOOTH,
    label_mask,
    load_annotation,
    load_image,
    MASK_RGBA,
    REGION_RGBA,
    mask_points,
    overlay,
    setup_cjk_font,
)

SIDE_COLOR = {"A": "#ff3b30", "B": "#ffd54a"}


def detrend_by_thickness(
    values: np.ndarray, distance: np.ndarray, n_bins: int = 40
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """扣掉「射線穿過的厚度」造成的灰階變化，留下組織差異。

    X 光在輪廓附近是擦邊穿過牙齒的，路徑短、衰減少所以暗；越往內部路徑
    越長就越亮。這個幾何效應會蓋過琺瑯質與牙本質的密度差（實測相關係數
    達 +0.74~+0.88），不先扣掉就看不到組織結構。

    這裡用無母數的做法：把像素依「到輪廓的距離」分箱，取各箱的**中位數**
    當作該厚度下的預期灰階，再相減。不假設任何函數形式（線性、平方根
    都不假設），也不受少數極端值影響。

    回傳 (殘差, 各箱中心, 各箱中位數)。
    """
    edges = np.linspace(distance.min(), distance.max(), n_bins + 1)
    index = np.clip(np.digitize(distance, edges) - 1, 0, n_bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    expected = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = index == i
        if sel.sum() >= 5:
            expected[i] = np.median(values[sel])

    ok = ~np.isnan(expected)
    baseline = np.interp(distance, centers[ok], expected[ok])
    return values - baseline, centers[ok], expected[ok]


def residual_map(img, mask, region):
    """區域內每個像素扣掉厚度效應後的殘差。"""
    rows, cols = region[:, 1].astype(int), region[:, 0].astype(int)
    values = img[rows, cols].astype(float)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)[rows, cols]
    residual, _, _ = detrend_by_thickness(values, distance)
    return residual, values, distance


def proximal_edge(
    band: np.ndarray, frame: ToothFrame, surface_distance: np.ndarray,
    surface_tolerance: float, bin_px: float = 1.0,
) -> np.ndarray:
    """帶狀區域貼著鄰接面的那條外緣。

    分兩步，各處理一個問題：

    ① **每個高度只取最外側的一點。**帶狀區域的輪廓繞了一整圈，包含面向
       牙本質核心的內緣、以及沿著切割線的下緣，那些都不是鄰接面。沿 y'
       分箱後每箱取 |x'| 最大者，內緣（同高度 |x'| 較小）自動被淘汰，
       水平線段也會塌縮成單一點。結果是單值函數，保證不斷線，而且分箱
       寬度取 1 px 與影像解析度一致，不算可調參數。

    ② **只留貼著牙齒表面的部分。**邊緣嵴是牙齒表面上的構造。某些高度上
       帶狀區域根本沒延伸到鄰接面，那裡的「最外側」其實仍在牙齒深處，
       用到距離變換濾掉即可。容差取 CEJ 寬度（C 到 D 的距離）的一個比例，
       這樣同一個設定在不同大小的牙齒、不同解析度的影像上都等價。
    """
    x_prime, y_prime = frame.to_frame(band)
    n_bins = max(5, int(round(np.ptp(y_prime) / bin_px)))
    edges = np.linspace(y_prime.min(), y_prime.max(), n_bins + 1)
    index = np.clip(np.digitize(y_prime, edges) - 1, 0, n_bins - 1)

    picked = []
    for i in range(n_bins):
        rows = np.flatnonzero(index == i)
        if len(rows):
            picked.append(rows[np.argmax(np.abs(x_prime[rows]))])
    edge = band[picked]

    depth = surface_distance[edge[:, 1].astype(int), edge[:, 0].astype(int)]
    return edge[depth <= surface_tolerance]


def ridge_points(
    region: np.ndarray, residual: np.ndarray, frame: ToothFrame,
    threshold: float, mask: np.ndarray, surface_tolerance: float,
    robust_fraction: float = 0.05, min_top: int = 5,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """取左右兩側鄰接面外緣的上端，作為 A、B。

    帶狀區域用「每側最大的連通元件」取得，可自然濾掉零星雜點，不必再加
    形態學開運算之類的額外參數；外緣的抽取見 proximal_edge。
    """
    keep = residual >= threshold
    selected = region[keep]
    if len(selected) < 20:
        raise ValueError(f"殘差 >= {threshold} 的像素只有 {len(selected)} 個，太少")

    binary = np.zeros(mask.shape, np.uint8)
    binary[selected[:, 1].astype(int), selected[:, 0].astype(int)] = 1

    _, components = cv2.connectedComponents(binary)
    x_prime, _ = frame.to_frame(selected)
    surface_distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    points, bands, edge_lines = {}, [], []
    for name, side in (("A", -1), ("B", +1)):
        on_side = (x_prime < 0) if side < 0 else (x_prime >= 0)
        if on_side.sum() < 10:
            raise ValueError(f"{name} 側通過閾值的像素太少")

        # 該側最大的連通元件
        labels = components[
            selected[on_side][:, 1].astype(int), selected[on_side][:, 0].astype(int)
        ]
        biggest = np.bincount(labels[labels > 0]).argmax()
        band = selected[on_side][labels == biggest]
        bands.append(band)

        edge = proximal_edge(band, frame, surface_distance, surface_tolerance)
        if len(edge) < 3:
            raise ValueError(f"{name} 側的帶狀區域沒有貼著牙齒表面的部分")

        x_edge, y_edge = frame.to_frame(edge)

        edge_lines.append(edge)

        # 不取單一極值像素（容易被鋸齒影響），改取最上面一批的中位數；
        # 但輪廓點可能不多，比例太小會退化成 k=1，所以給一個點數下限。
        k = min(len(edge), max(min_top, int(len(edge) * robust_fraction)))
        top = np.argsort(y_edge)[-k:]
        points[name] = np.median(edge[top], axis=0)

    return points, np.vstack(bands), np.vstack(edge_lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", help="labelme 的 .json 檔，或含有多個 .json 的資料夾")
    parser.add_argument("--cut-at", type=float, default=1.0 / 3.0,
                        help="牙冠取樣範圍，0 = 整個牙冠，預設 1/3")
    parser.add_argument("--drop-apical", type=float, default=0.25, help="傳給 find_cej 的參數")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="殘差閾值，預設 0（比預期亮就算）")
    parser.add_argument("--surface-fraction", type=float, default=0.10,
                        help="外緣點距牙齒表面的容差，取 CEJ 寬度的比例，預設 0.10")
    parser.add_argument("-o", "--out", help="存檔到此路徑，不開視窗")
    args = parser.parse_args()

    target = Path(args.target)
    json_files = sorted(target.glob("*.json")) if target.is_dir() else [target]
    if not json_files:
        raise SystemExit(f"在 {target} 找不到任何 .json 標註檔")

    import matplotlib

    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_cjk_font(plt)

    fig, axes = plt.subplots(2, len(json_files), figsize=(5.2 * len(json_files), 10),
                             squeeze=False)

    for col, jf in enumerate(json_files):
        data = load_annotation(jf)
        img = load_image(data)
        mask = label_mask(data, LABEL_TOOTH)
        points = mask_points(mask)
        frame = fit_axis(points)

        try:
            cej = cej_points(points, frame, args.drop_apical)
            region, _, _ = upper_crown_region(points, frame, cej, args.cut_at)
            residual, _, _ = residual_map(img, mask, region)
            # 容差跟著 CEJ 寬度縮放，才不會隨牙齒大小或影像解析度而改變意義
            cej_width = float(np.hypot(cej["D"][0] - cej["C"][0], cej["D"][1] - cej["C"][1]))
            ridge, bands, edges = ridge_points(region, residual, frame, args.threshold, mask,
                                              args.surface_fraction * cej_width)
        except ValueError as exc:
            print(f"{data['imagePath']}：失敗 - {exc}\n")
            for row in range(2):
                axes[row][col].axis("off")
            axes[0][col].set_title(f"{jf.stem}  失敗", fontsize=11)
            continue

        # 論文的 J：過 A、B 各作一條平行於 CD 的直線，與長軸交於 H、I，取其中點。
        # 注意是沿 **CD 方向** 投影，不是垂直長軸投影——CD 通常是斜的，兩者
        # 在 CD 傾角大且 A、B 離軸距離不對稱時會差到 1.5% 牙長。
        x_a, y_a = frame.to_frame(ridge["A"][None, :])
        x_b, y_b = frame.to_frame(ridge["B"][None, :])
        (xc, yc), (xd, yd) = cej["C"], cej["D"]
        dx, dy = xd - xc, yd - yc
        slope = dy / dx if abs(dx) > 1e-9 else 0.0
        h = float(y_a[0] - x_a[0] * slope)
        i = float(y_b[0] - x_b[0] * slope)
        y_j = (h + i) / 2
        tooth_top = float(frame.to_frame(points)[1].max())

        print(f"{data['imagePath']}")
        print(f"   A = ({ridge['A'][0]:.0f}, {ridge['A'][1]:.0f})   y' = {y_a[0]:.1f}")
        print(f"   B = ({ridge['B'][0]:.0f}, {ridge['B'][1]:.0f})   y' = {y_b[0]:.1f}")
        print(f"   兩側高度差 {abs(y_a[0] - y_b[0]):.1f} px　CD 傾角 {np.degrees(np.arctan2(dy, dx)):+.1f} 度")
        print(f"   J（中點）y' = {y_j:.1f}　距牙冠頂端 {tooth_top - y_j:.1f} px")
        print(f"   通過閾值的帶狀區域共 {len(bands)} 像素　"
              f"CEJ 寬 {cej_width:.0f} px → 表面容差 {args.surface_fraction * cej_width:.1f} px\n")

        ax = axes[0][col]
        ax.imshow(img, cmap="gray")
        overlay(ax, mask.shape, points, MASK_RGBA)
        overlay(ax, mask.shape, bands, REGION_RGBA)
        ax.plot(edges[:, 0], edges[:, 1], ".", color="#00e5ff", markersize=1.2,
                label="輪廓上的帶狀區域")
        for name in ("A", "B"):
            p = ridge[name]
            ax.plot(p[0], p[1], "o", color=SIDE_COLOR[name], markersize=5,
                    markeredgecolor="black", markeredgewidth=0.6)
            ax.annotate(name, (p[0], p[1]), color=SIDE_COLOR[name], fontsize=11,
                        fontweight="bold", xytext=(7, 3), textcoords="offset pixels")
        ax.plot([ridge["A"][0], ridge["B"][0]], [ridge["A"][1], ridge["B"][1]],
                "-", color="#00e5ff", linewidth=1.0, label="A-B")
        j_img = frame.point_at(y_j)
        ax.plot(*j_img, "o", color="#00e5ff", markersize=5,
                markeredgecolor="black", markeredgewidth=0.6, label="J")
        ax.annotate("J", j_img, color="#00e5ff", fontsize=11, fontweight="bold",
                    xytext=(7, 3), textcoords="offset pixels")
        ax.set_title(f"{jf.stem}", fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        ax.axis("off")

        rax = axes[1][col]
        x0 = max(0, int(region[:, 0].min()) - 8)
        y0 = max(0, int(region[:, 1].min()) - 8)
        x1, y1 = int(region[:, 0].max()) + 8, int(region[:, 1].max()) + 8
        canvas = np.full(img.shape, np.nan)
        canvas[region[:, 1].astype(int), region[:, 0].astype(int)] = residual
        lim = float(np.percentile(np.abs(residual), 98))
        im = rax.imshow(canvas[y0:y1, x0:x1], cmap="RdBu_r", vmin=-lim, vmax=lim)
        overlay(rax, img.shape, bands, (0.0, 0.0, 0.0, 0.18), crop=(y0, y1, x0, x1))
        rax.plot(edges[:, 0] - x0, edges[:, 1] - y0, ".", color="black", markersize=1.4)
        for name in ("A", "B"):
            p = ridge[name]
            rax.plot(p[0] - x0, p[1] - y0, "o", color=SIDE_COLOR[name], markersize=5,
                     markeredgecolor="black", markeredgewidth=0.6)
        rax.set_title(f"殘差（閾值 {args.threshold:g}；灰 = 帶狀區域，黑點 = 輪廓上的部分）", fontsize=9)
        rax.axis("off")
        fig.colorbar(im, ax=rax, fraction=0.046)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=115, bbox_inches="tight")
        print(f"已存檔 -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
