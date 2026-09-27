"""切出牙冠，畫出它的灰階直方圖。

範圍怎麼定
----------
先用 find_cej 算出 C、D 兩點，連成 CEJ 線，取它與長軸的交點 L（高度 y'_L）。
牙冠頂端取遮罩的最高處 y'_top。兩者的中間高度

    y'_mid = (y'_L + y'_top) / 2

就是切割的位置。切割線**平行於 CD**（不是水平），通過長軸上 y'_mid 那一點，
保留線以上、且落在牙齒遮罩內的像素。

輸出三列：切出來的範圍、範圍內依灰階上色的樣子（方便把直方圖的峰對回
空間位置）、以及灰階直方圖。

用法：
    py scripts/crown_histogram.py labeled_PA                  # 整個牙冠
    py scripts/crown_histogram.py labeled_PA --cut-at 0.5     # 只取上半個牙冠
    py scripts/crown_histogram.py labeled_PA --bins 96 -o hist.png
    py scripts/crown_histogram.py labeled_PA --cut-at 0.333 --bins 128 -o hist.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.find_axis_raw import fit_axis  # noqa: E402
from scripts.find_cej import cej_points, upper_crown_region  # noqa: E402
from scripts.find_ridge import detrend_by_thickness  # noqa: E402
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", help="labelme 的 .json 檔，或含有多個 .json 的資料夾")
    parser.add_argument("--drop-apical", type=float, default=0.25, help="傳給 find_cej 的參數")
    parser.add_argument("--cut-at", type=float, default=0.0,
                        help="切割位置：0 = 整個牙冠（預設），0.5 = 上半個牙冠")
    parser.add_argument("--bins", type=int, default=64, help="直方圖分箱數，預設 64")
    parser.add_argument("--detrend", action="store_true",
                        help="多畫兩列：扣掉射線路徑長度效應後的殘差與其直方圖")
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

    n_rows = 5 if args.detrend else 3
    fig, axes = plt.subplots(n_rows, len(json_files),
                             figsize=(5.2 * len(json_files), 4.7 * n_rows), squeeze=False)

    for col, jf in enumerate(json_files):
        data = load_annotation(jf)
        img = load_image(data)
        mask = label_mask(data, LABEL_TOOTH)
        points = mask_points(mask)
        frame = fit_axis(points)

        try:
            cej = cej_points(points, frame, args.drop_apical)
        except ValueError as exc:
            print(f"{data['imagePath']}：CEJ 擬合失敗 - {exc}\n")
            for row in range(3):
                axes[row][col].axis("off")
            axes[0][col].set_title(f"{jf.stem}  CEJ 擬合失敗", fontsize=11)
            continue

        region, y_cut, ends = upper_crown_region(points, frame, cej, args.cut_at)
        values = img[region[:, 1].astype(int), region[:, 0].astype(int)].astype(float)

        label = "整個牙冠" if args.cut_at <= 0 else f"牙冠上 {100*(1-args.cut_at):.0f}%"
        print(f"{data['imagePath']}   {label} {len(region)} 像素")
        print(f"   灰階：最小 {values.min():.0f}  最大 {values.max():.0f}  "
              f"平均 {values.mean():.1f}  中位數 {np.median(values):.0f}  標準差 {values.std():.1f}")
        for q in (5, 25, 50, 75, 95):
            print(f"     第 {q:2d} 百分位 = {np.percentile(values, q):.0f}", end="")
        print("\n")

        # --- 第一列：切出來的範圍 ---
        ax = axes[0][col]
        ax.imshow(img, cmap="gray")
        overlay(ax, mask.shape, points, MASK_RGBA)
        overlay(ax, mask.shape, region, REGION_RGBA)
        c_img = frame.to_image(np.array([cej["C"][0]]), np.array([cej["C"][1]]))[0]
        d_img = frame.to_image(np.array([cej["D"][0]]), np.array([cej["D"][1]]))[0]
        ax.plot([c_img[0], d_img[0]], [c_img[1], d_img[1]], "-", color="#00e5ff",
                linewidth=1.0, label="CEJ 線")
        ax.plot([ends[0][0], ends[1][0]], [ends[0][1], ends[1][1]], "--", color="#ffd54a",
                linewidth=1.0, label="切割線（平行 CD）")
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.set_title(f"{jf.stem}   {label} {len(region)} px", fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        ax.axis("off")

        # --- 第二列：範圍內依灰階上色，方便把直方圖的峰對回位置 ---
        vax = axes[1][col]
        x0, x1 = int(region[:, 0].min()) - 8, int(region[:, 0].max()) + 8
        y0, y1 = int(region[:, 1].min()) - 8, int(region[:, 1].max()) + 8
        x0, y0 = max(0, x0), max(0, y0)
        canvas = np.full(img.shape, np.nan)
        canvas[region[:, 1].astype(int), region[:, 0].astype(int)] = values
        im = vax.imshow(canvas[y0:y1, x0:x1], cmap="viridis")
        vax.set_title("依灰階上色", fontsize=10)
        vax.axis("off")
        fig.colorbar(im, ax=vax, fraction=0.046)

        # --- 第三列：灰階直方圖 ---
        hax = axes[2][col]
        hax.hist(values, bins=args.bins, color="#4a7fb5", edgecolor="none")
        for q, style in ((25, ":"), (50, "-"), (75, ":")):
            hax.axvline(np.percentile(values, q), color="#ff3b30", linestyle=style, linewidth=1.0)
        hax.set_xlabel("灰階值")
        hax.set_ylabel("像素數")
        hax.set_title(f"直方圖（{args.bins} 箱）　紅線 = 四分位數", fontsize=10)

        if not args.detrend:
            continue

        # --- 第四、五列：扣掉厚度效應後的殘差 ---
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        distance = dist_map[region[:, 1].astype(int), region[:, 0].astype(int)]
        residual, dc, dm = detrend_by_thickness(values, distance)
        print(f"   厚度效應：灰階 vs 離輪廓距離 r = {np.corrcoef(values, distance)[0, 1]:+.3f}"
              f"　扣除後殘差標準差 = {residual.std():.1f}（原始 {values.std():.1f}）")

        rax = axes[3][col]
        canvas = np.full(img.shape, np.nan)
        canvas[region[:, 1].astype(int), region[:, 0].astype(int)] = residual
        lim = float(np.percentile(np.abs(residual), 98))
        im2 = rax.imshow(canvas[y0:y1, x0:x1], cmap="RdBu_r", vmin=-lim, vmax=lim)
        rax.set_title("扣掉厚度效應後的殘差（紅=比預期亮，藍=比預期暗）", fontsize=9)
        rax.axis("off")
        fig.colorbar(im2, ax=rax, fraction=0.046)

        qax = axes[4][col]
        qax.hist(residual, bins=args.bins, color="#b5564a", edgecolor="none")
        qax.axvline(0, color="black", linewidth=0.8)
        qax.set_xlabel("殘差（灰階）")
        qax.set_ylabel("像素數")
        qax.set_title("殘差直方圖", fontsize=10)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=110, bbox_inches="tight")
        print(f"已存檔 -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
