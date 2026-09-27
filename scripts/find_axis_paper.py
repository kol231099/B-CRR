"""依 Li et al. 2024 的規則精修長軸，並畫出論文 Fig.1 式的作圖。

流程
----
1. 整顆牙齒遮罩做 PCA，得到初始（raw）座標系與牙冠方向  find_axis_raw
2. 在 raw 座標系上求 CEJ 的 C、D                          find_cej
3. 用 CEJ 切出牙冠，結合灰階求邊緣嵴的 A、B                find_ridge
4. 求根尖 G，把 A、B、G 沿 CD 方向投影到長軸得 H、I、K，
   J 取 H、I 中點、L 為 CD 與長軸的交點；裁掉 J 以上的牙冠與
   根尖三分之一的牙根，對剩下的區域重跑 PCA

論文只說長軸取自「邊緣嵴以下的牙冠 + 牙根冠側三分之二」，沒有規定切割
線的幾何。這裡**高度用論文的作法算（沿 CD 投影），但切割線垂直於 raw
主軸**：沿 CD 這種斜線切，會在邊界處左右移除的量不對等，等於給協方差
引入旋轉偏差；垂直切則保持左右對稱。

座標系一律保持正交。沿 CD 的投影只是一行算式（見 axial），包成函式即可；
若把 x' 改成 CD 方向而讓座標系變成斜交，雖然 H/I/K 可以直接讀 y'，但
to_frame 得改解反矩陣、所有形狀分析（半寬、離軸距離、PCA）都會算錯，
還是得另外維護一個正交座標系，得不償失。

用法：
    py scripts/find_axis_paper.py labeled_PA
    py scripts/find_axis_paper.py labeled_PA/13.json
    py scripts/find_axis_paper.py labeled_PA -o axis_paper.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.find_cej import cej_points, upper_crown_region  # noqa: E402
from scripts.find_axis_raw import (  # noqa: E402
    ToothFrame,
    angle_between,
    cross_axis,
    fit_axis,
    pca,
)
from scripts.find_ridge import residual_map, ridge_points  # noqa: E402
from scripts.labelme_io import (  # noqa: E402
    LABEL_TOOTH,
    MASK_RGBA,
    label_mask,
    load_annotation,
    load_image,
    mask_points,
    overlay,
    setup_cjk_font,
)

LANDMARK_COLOR = "#1f6feb"  # A B C D G：直接量到的
DERIVED_COLOR = "#8a8a8a"  # H I J K L：推演出來的
LINE_COLOR = "black"
LINE_WIDTH = 0.6


@dataclass
class PaperAxis:
    """精修後的長軸，以及推導過程中的所有點位。"""

    frame_raw: ToothFrame  # 初始座標系
    frame: ToothFrame  # 精修後的座標系
    landmarks: dict  # A B C D G，影像座標
    levels: dict  # H I J K L，**精修**座標系的軸向高度，量測用這組
    levels_raw: dict  # 同上但在 raw 座標系，只用來裁切與比較
    trim: tuple[float, float]  # 保留的 y' 範圍（raw 座標系，裁切當下的依據）
    cd_slope: float  # CD 在精修座標系中的斜率

    @property
    def crown_length(self) -> float:
        return self.levels["J"] - self.levels["L"]

    @property
    def root_length(self) -> float:
        return self.levels["L"] - self.levels["K"]

    @property
    def crr(self) -> float:
        return self.crown_length / self.root_length

    @property
    def angle(self) -> float:
        return angle_between(self.frame_raw.y_axis, self.frame.y_axis)


def axial(points: np.ndarray, frame: ToothFrame, cd_slope: float) -> np.ndarray:
    """沿 CD 方向投影到長軸之後的高度。

    論文的 H、I、K 都是「過該點作平行於 CD 的直線，與長軸的交點」。在正交
    座標系裡，沿 CD 移動會讓 y' 改變 x' * slope，扣掉即得。

    自我檢查：axial(C) 應等於 axial(D)，因為兩點本來就在 CD 線上。
    """
    x_prime, y_prime = frame.to_frame(points)
    return y_prime - x_prime * cd_slope


def project_landmarks(landmarks: dict, frame: ToothFrame, cd_slope: float) -> dict:
    """把 A、B、C、D、G 沿 CD 方向投影到長軸，得到 H、I、J、K、L。"""
    level = {name: float(axial(p[None, :], frame, cd_slope)[0])
             for name, p in landmarks.items()}
    return {
        "H": level["A"], "I": level["B"], "J": (level["A"] + level["B"]) / 2,
        "K": level["G"], "L": (level["C"] + level["D"]) / 2,
    }


def apex_point(points: np.ndarray, frame: ToothFrame, cd_slope: float,
               fraction: float = 0.01) -> np.ndarray:
    """G：軸向座標最小（最靠根尖）之處。

    取最底部一批的中位數而非單一極值像素，避免被遮罩邊緣的鋸齒或雜點影響。
    """
    depth = axial(points, frame, cd_slope)
    k = max(1, int(len(points) * fraction))
    return np.median(points[np.argsort(depth)[:k]], axis=0)


def refine_axis(
    img: np.ndarray, mask: np.ndarray, drop_apical: float, cut_at: float,
    threshold: float, surface_fraction: float,
    root_fraction: float = 2.0 / 3.0,
) -> PaperAxis:
    points = mask_points(mask)
    frame_raw = fit_axis(points)

    cej = cej_points(points, frame_raw, drop_apical)
    (xc, yc), (xd, yd) = cej["C"], cej["D"]
    cd_slope = (yd - yc) / (xd - xc) if abs(xd - xc) > 1e-9 else 0.0

    region, _, _ = upper_crown_region(points, frame_raw, cej, cut_at)
    residual, _, _ = residual_map(img, mask, region)
    cej_width = float(np.hypot(xd - xc, yd - yc))
    ridge, _, _ = ridge_points(region, residual, frame_raw, threshold, mask,
                               surface_fraction * cej_width)

    landmarks = {
        "A": ridge["A"], "B": ridge["B"],
        "C": frame_raw.to_image(np.array([xc]), np.array([yc]))[0],
        "D": frame_raw.to_image(np.array([xd]), np.array([yd]))[0],
    }
    landmarks["G"] = apex_point(points, frame_raw, cd_slope)

    levels_raw = project_landmarks(landmarks, frame_raw, cd_slope)

    # 裁切：上界為 J，下界保留牙根冠側 root_fraction。切割線垂直於 raw 主軸，
    # 所以直接對 raw 座標系的 y' 設條件。
    root_length = levels_raw["L"] - levels_raw["K"]
    low = levels_raw["K"] + (1 - root_fraction) * root_length
    _, y_prime = frame_raw.to_frame(points)
    keep = (y_prime >= low) & (y_prime <= levels_raw["J"])
    if keep.sum() < 10:
        raise ValueError("裁切後剩下的像素太少")

    origin, components, eigenvalues = pca(points[keep])
    y_axis = components[0]
    # 牙冠方向沿用 raw 的判定：裁切後的區域不是完整牙齒，用它自己的形狀
    # 重新判方向沒有意義。
    if float(np.dot(y_axis, frame_raw.y_axis)) < 0:
        y_axis = -y_axis

    frame = ToothFrame(
        origin=origin, x_axis=cross_axis(y_axis), y_axis=y_axis,
        eigenvalues=eigenvalues, n_points=int(keep.sum()),
        crown_ratio=frame_raw.crown_ratio, root_ratio=frame_raw.root_ratio,
    )

    # 量測要在**精修後的軸**上做——raw 軸只是用來找特徵點與決定裁切範圍的
    # 中間產物。CD 的斜率也得重算：兩個座標系之間有旋轉，同一條 CD 線在
    # 各自座標系裡的斜率不同。
    xc2, yc2 = frame.to_frame(landmarks["C"][None, :])
    xd2, yd2 = frame.to_frame(landmarks["D"][None, :])
    cd_slope_final = (float(yd2[0]) - float(yc2[0])) / (float(xd2[0]) - float(xc2[0])) \
        if abs(float(xd2[0]) - float(xc2[0])) > 1e-9 else 0.0
    levels = project_landmarks(landmarks, frame, cd_slope_final)

    return PaperAxis(frame_raw, frame, landmarks, levels, levels_raw,
                     (low, levels_raw["J"]), cd_slope_final)


# --------------------------------------------------------------------------
# 作圖：參照論文 Fig.1
# --------------------------------------------------------------------------


def mark(ax, point, label, color, size, fontsize, offset):
    """畫一個點與它的字母標籤。

    點加一圈黑邊；字母一律**白字黑邊**，才不會在深淺不一的 X 光背景上融進去
    ——用顏色當字色時，藍字落在暗處、灰字落在亮處都會看不清。顏色的資訊改由
    點本身承載（藍＝論文定義的特徵點，灰＝推演點）。
    """
    import matplotlib.patheffects as pe

    ax.plot(point[0], point[1], "o", color=color, markersize=size,
            markeredgecolor="black", markeredgewidth=0.7)
    ax.annotate(label, (point[0], point[1]), color="white", fontsize=fontsize,
                fontweight="bold", xytext=offset, textcoords="offset pixels",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])


def draw(ax, img, mask, result: PaperAxis):
    frame = result.frame
    points = mask_points(mask)
    _, y_prime = frame.to_frame(points)

    ax.imshow(img, cmap="gray")
    overlay(ax, mask.shape, points, MASK_RGBA)

    def on_axis(level):
        return frame.point_at(level)

    # 長軸
    top, bottom = on_axis(float(y_prime.max())), on_axis(float(y_prime.min()))
    ax.plot([bottom[0], top[0]], [bottom[1], top[1]], "-",
            color=LINE_COLOR, linewidth=LINE_WIDTH)

    # 過 A、B、C、D、G 各作一條平行於 CD 的線，交長軸於 H、I、L、K
    half = float(np.abs(frame.to_frame(points)[0]).max()) * 1.15
    for name, level in (("A", "H"), ("B", "I"), ("C", "L"), ("D", "L"), ("G", "K")):
        p = result.landmarks[name]
        target = on_axis(result.levels[level])
        ax.plot([p[0], target[0]], [p[1], target[1]], "-",
                color=LINE_COLOR, linewidth=LINE_WIDTH)

    # 裁切範圍。切割當下是在 **raw** 座標系裡做的（垂直於 raw 主軸），所以
    # 這裡也必須用 frame_raw 換算回影像座標——用精修軸換算會畫錯位置。
    raw = result.frame_raw
    half_raw = float(np.abs(raw.to_frame(points)[0]).max()) * 1.15
    for level in result.trim:
        a = raw.to_image(np.array([-half_raw]), np.array([level]))[0]
        b = raw.to_image(np.array([half_raw]), np.array([level]))[0]
        ax.plot([a[0], b[0]], [a[1], b[1]], ":", color=LINE_COLOR,
                linewidth=LINE_WIDTH, alpha=0.7)

    for name, p in result.landmarks.items():
        mark(ax, p, name, LANDMARK_COLOR, 4.5, 10, (6, 3))

    # H、I、J 三點的高度很接近（A、B 兩側邊緣嵴差不多高，J 又是兩者中點），
    # 標籤直接疊在一起會看不清，所以左右交錯放。
    label_offset = {"H": (5, -12), "I": (5, 4), "J": (-14, -4), "K": (5, -12), "L": (5, -12)}
    for name, level in result.levels.items():
        mark(ax, on_axis(level), name, DERIVED_COLOR, 3.5, 9, label_offset[name])

    # raw 主軸以灰色虛線保留供對照（實際量測用的是黑色那條精修主軸）
    _, y_raw = result.frame_raw.to_frame(points)
    a = result.frame_raw.point_at(float(y_raw.max()))
    b = result.frame_raw.point_at(float(y_raw.min()))
    ax.plot([b[0], a[0]], [b[1], a[1]], "--", color=DERIVED_COLOR, linewidth=0.8,
            label=f"raw 主軸（相差 {result.angle:.2f}°）")
    ax.legend(fontsize=7, loc="lower right")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", help="labelme 的 .json 檔，或含有多個 .json 的資料夾")
    parser.add_argument("--drop-apical", type=float, default=0.25, help="傳給 find_cej")
    parser.add_argument("--cut-at", type=float, default=1.0 / 3.0, help="傳給 find_ridge")
    parser.add_argument("--threshold", type=float, default=0.0, help="傳給 find_ridge")
    parser.add_argument("--surface-fraction", type=float, default=0.10, help="傳給 find_ridge")
    parser.add_argument("--root-fraction", type=float, default=2.0 / 3.0,
                        help="牙根保留冠側的比例，預設 2/3")
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
    fig, axes = plt.subplots(1, len(json_files), figsize=(5.2 * len(json_files), 9),
                             squeeze=False)

    for col, jf in enumerate(json_files):
        data = load_annotation(jf)
        img = load_image(data)
        mask = label_mask(data, LABEL_TOOTH)

        try:
            result = refine_axis(img, mask, args.drop_apical, args.cut_at,
                                 args.threshold, args.surface_fraction, args.root_fraction)
        except ValueError as exc:
            print(f"{data['imagePath']}：失敗 - {exc}\n")
            axes[0][col].axis("off")
            axes[0][col].set_title(f"{jf.stem}  失敗", fontsize=11)
            continue

        raw = result.levels_raw
        crr_raw = (raw["J"] - raw["L"]) / (raw["L"] - raw["K"])
        print(f"{data['imagePath']}")
        print("   " + "  ".join(f"{n}=({p[0]:.0f},{p[1]:.0f})"
                                for n, p in result.landmarks.items()))
        print("   軸向高度（精修軸）：" + "  ".join(f"{n}={v:.1f}" for n, v in result.levels.items()))
        print(f"   牙冠長 JL={result.crown_length:.1f}  牙根長 KL={result.root_length:.1f}"
              f"  →  CRR={result.crr:.4f}   （若用 raw 軸量：{crr_raw:.4f}）")
        print(f"   裁切保留 y' [{result.trim[0]:.1f}, {result.trim[1]:.1f}]"
              f"　{result.frame.n_points} 像素（原 {len(mask_points(mask))}）")
        print(f"   精修後長軸偏轉 {result.angle:.2f} 度\n")

        draw(axes[0][col], img, mask, result)
        axes[0][col].set_title(f"{jf.stem}    CRR = {result.crr:.3f}", fontsize=11)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=130, bbox_inches="tight")
        print(f"已存檔 -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
