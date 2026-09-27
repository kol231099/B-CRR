"""以主成分分析擬合牙齒自身的座標系。

座標系僅由整顆牙齒的遮罩（標籤 "1"）建立，不依賴其他標註：

    y'  長軸方向，符號固定為**指向牙冠端**
    x'  垂直於 y'，符號固定為牙冠朝上時**畫面向右**

兩者皆為單位向量，原點取遮罩形心。因此把點映射進此座標系是一個
**剛體變換**——只有旋轉與平移，不含縮放、也不重採樣任何像素，
長度與角度完全保持不變。這對本研究很重要，因為 CRR 是兩段長度的
比值，任何縮放失真都會直接污染結果。

在此座標系中計算，本身就等同於「把牙齒轉正」。除非另有需要產生
轉正後的圖片，否則不必真的旋轉任何像素。

哪一端是牙冠
------------
以形心為界、沿 x' 方向把遮罩切成兩半，對每一半計算

    Var(x') / Var(y')

也就是「該半有多寬」相對於「該半有多長」。牙冠半寬而短，牙根半
細而長，所以牙冠半的分數較高——而且分子分母同時往相反方向走，
兩項效果相乘，使兩者的差距被放大。

此指標沒有任何可調參數；又因為是變異數的比值，本身無量綱，所以
差距倍率可當作**絕對的信心度**來讀，不會隨牙齒大小或影像解析度
而改變。

用法：
    py scripts/find_axis_raw.py labeled_PA/7.json
    py scripts/find_axis_raw.py labeled_PA                # 資料夾內所有 .json
    py scripts/find_axis_raw.py labeled_PA --cut          # 畫出切分線
    py scripts/find_axis_raw.py labeled_PA --profile      # 加畫寬度剖面圖
    py scripts/find_axis_raw.py labeled_PA -o axis_raw.png    # 存檔，不開視窗
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.labelme_io import (  # noqa: E402
    LABEL_TOOTH,
    label_mask,
    load_annotation,
    load_image,
    mask_points,
    setup_cjk_font,
)

AXIS_COLOR = "#ff4d4d"
CUT_COLOR = "#ffd54a"
CROWN_TINT = "#4dc3ff"
ROOT_TINT = "#b48cff"


# --------------------------------------------------------------------------
# 擬合出來的座標系
# --------------------------------------------------------------------------


@dataclass
class ToothFrame:
    """牙齒自身的座標系，以及擬合過程中得到的診斷數值。"""

    origin: np.ndarray  # (2,) 形心，影像座標 (x, y)
    x_axis: np.ndarray  # (2,) 單位向量，橫跨牙齒
    y_axis: np.ndarray  # (2,) 單位向量，沿牙齒長軸，指向牙冠
    eigenvalues: np.ndarray  # (2,) 共變異數矩陣特徵值，[長軸, 短軸]
    n_points: int
    crown_ratio: float  # 牙冠半的 Var(x')/Var(y')
    root_ratio: float  # 牙根半的 Var(x')/Var(y')

    @property
    def anisotropy(self) -> float:
        """長軸變異數除以短軸變異數——整顆牙有多細長。

        接近 1.0 代表形狀接近圓形，長軸方向不具意義；細長的牙齒
        會得到很大的值。
        """
        return float(self.eigenvalues[0] / max(self.eigenvalues[1], 1e-12))

    @property
    def margin(self) -> float:
        """牙冠半的分數是牙根半的幾倍，即牙冠端判定的信心度。

        接近 1.0 代表兩端形狀相似，此時的方向判定不可信。
        """
        return float(self.crown_ratio / max(self.root_ratio, 1e-12))

    def to_frame(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把影像座標的點映射進本座標系，回傳 (x', y')。

        y' 往牙冠方向增加，所以根尖位於 y' 最小（最負）處。
        """
        rel = np.atleast_2d(np.asarray(points, dtype=float)) - self.origin
        return rel @ self.x_axis, rel @ self.y_axis

    def to_image(self, x_prime, y_prime) -> np.ndarray:
        """把本座標系的座標映射回影像像素座標。"""
        x_prime = np.asarray(x_prime, dtype=float)
        y_prime = np.asarray(y_prime, dtype=float)
        return self.origin + x_prime[..., None] * self.x_axis + y_prime[..., None] * self.y_axis

    def point_at(self, y_prime: float) -> np.ndarray:
        """長軸上高度為 y_prime 的那個點。"""
        return self.origin + y_prime * self.y_axis


# --------------------------------------------------------------------------
# 擬合
# --------------------------------------------------------------------------


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """兩個單位向量的夾角（度），忽略正負號。"""
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(u, v))), -1.0, 1.0))))


def pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """對 (N, 2) 陣列做 PCA，回傳 (形心, 主成分, 特徵值)。

    components[0] 為長軸、components[1] 為短軸；特徵值以相同的
    降冪順序回傳。

    此處顯式建立 2x2 共變異數矩陣再做特徵分解。一般而言應避免形成
    X^T X，因為該運算會讓條件數平方；改用置中資料的 SVD 可以繞過。
    但那條通則是針對欄數眾多的問題而言。本問題的 X 只有兩欄、條件數
    約為 4，實測兩種做法一致到約 1e-15；而 SVD 還會額外算出一個
    N x 2 的左奇異向量矩陣，算完隨即丟棄。在本專案的遮罩上實測，
    共變異數法快了兩倍有餘。
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError(f"PCA 至少需要 3 個點，只收到 {len(points)} 個")

    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = (centered.T @ centered) / (len(points) - 1)

    # eigh 是給對稱矩陣用的，回傳的特徵值為**升冪**、特徵向量是**行向量**，
    # 這兩點都很容易寫反。
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return centroid, eigenvectors[:, order].T, eigenvalues[order]


def shape_ratio(points: np.ndarray, origin: np.ndarray, major: np.ndarray, minor: np.ndarray) -> float:
    """一群點的「橫向變異數 / 縱向變異數」。

    變異數是對該群點**自身的平均值**計算，而非對長軸計算，所以牙冠
    相對牙根略有偏斜時，量到的仍是散布程度而非偏移量。變異數本身也
    已經除以像素數，因此即使形心沒有把兩半切成等量，兩者仍可直接比較。
    """
    rel = np.asarray(points, dtype=float) - origin
    along = rel @ major
    across = rel @ minor
    return float(np.var(across) / max(np.var(along), 1e-12))


def orient_to_crown(
    points: np.ndarray, centroid: np.ndarray, major: np.ndarray, minor: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """決定長軸的符號，使其指向牙冠端。

    回傳 (y_axis, 牙冠半分數, 牙根半分數)。
    """
    along = (points - centroid) @ major
    positive, negative = points[along >= 0], points[along < 0]
    if len(positive) < 3 or len(negative) < 3:
        raise ValueError("以形心切分後，其中一半的像素數過少")

    ratio_positive = shape_ratio(positive, centroid, major, minor)
    ratio_negative = shape_ratio(negative, centroid, major, minor)

    if ratio_positive >= ratio_negative:
        return major, ratio_positive, ratio_negative
    return -major, ratio_negative, ratio_positive


def cross_axis(y_axis: np.ndarray) -> np.ndarray:
    """回傳垂直於 y_axis、且在畫面上指向右方的單位向量。

    影像的列座標是往下增加的，所以牙冠朝上時 y' = (0, -1)，此式會
    給出 (1, 0)。選到另一個垂直向量會讓整個座標系左右鏡像卻不易察覺，
    因此這個慣例只在這裡決定一次，不在程式其他地方重複判斷。
    """
    return np.array([-y_axis[1], y_axis[0]])


def fit_axis(points: np.ndarray) -> ToothFrame:
    """用整顆牙齒遮罩的所有像素擬合出牙齒座標系。"""
    centroid, components, eigenvalues = pca(points)
    major, minor = components[0], components[1]

    y_axis, crown_ratio, root_ratio = orient_to_crown(points, centroid, major, minor)

    return ToothFrame(
        origin=centroid,
        x_axis=cross_axis(y_axis),
        y_axis=y_axis,
        eigenvalues=eigenvalues,
        n_points=len(points),
        crown_ratio=crown_ratio,
        root_ratio=root_ratio,
    )


# --------------------------------------------------------------------------
# 寬度剖面（供論文版長軸擬合使用）
# --------------------------------------------------------------------------


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """移動平均，邊緣以端點值填補，用來抑制分箱造成的雜訊。"""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="same")[window // 2 : window // 2 + len(values)]


def width_profile(
    coordinate: np.ndarray, n_bins: int = 100, smooth_window: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """形狀在沿軸各個位置上有多寬。

    把沿軸座標分箱後計算每箱的像素數。對填滿的遮罩而言，一個薄切片
    內的像素數正比於該處的寬度；這比量測每個切片的最大最小範圍穩健，
    因為後者會被單一雜訊像素主導。

    回傳 (各箱中心, 平滑後的像素數)。
    """
    edges = np.linspace(coordinate.min(), coordinate.max(), n_bins + 1)
    counts, _ = np.histogram(coordinate, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, _smooth(counts.astype(float), smooth_window)


# --------------------------------------------------------------------------
# 輸出與繪圖
# --------------------------------------------------------------------------


def describe(frame: ToothFrame) -> list[str]:
    lines = [
        f"  y'（指向牙冠）= ({frame.y_axis[0]:+.4f}, {frame.y_axis[1]:+.4f})   "
        f"x' = ({frame.x_axis[0]:+.4f}, {frame.x_axis[1]:+.4f})",
        f"  形心 = ({frame.origin[0]:.1f}, {frame.origin[1]:.1f})   像素數 = {frame.n_points}",
        f"  Var(y')={frame.eigenvalues[0]:.1f}  Var(x')={frame.eigenvalues[1]:.1f}  "
        f"-> 長短軸變異數比 = {frame.anisotropy:.2f}",
        f"  牙冠半 Var(x')/Var(y') = {frame.crown_ratio:.4f}   "
        f"牙根半 = {frame.root_ratio:.4f}   信心度 = {frame.margin:.2f} 倍",
    ]
    if frame.anisotropy < 3:
        lines.append(
            f"  警告：長短軸變異數比 {frame.anisotropy:.2f} 偏低，形狀接近圓形，"
            "長軸方向的可信度不足"
        )
    if frame.margin < 1.5:
        lines.append(
            f"  警告：牙冠／牙根信心度僅 {frame.margin:.2f} 倍，兩端形狀相似，"
            "此處的牙冠端判定不可靠"
        )
    return lines


def draw_frame(ax, frame: ToothFrame, points: np.ndarray, show_cut: bool):
    """畫出長軸；show_cut 為真時另外畫出切分線並把兩半染色。"""
    x_prime, y_prime = frame.to_frame(points)

    if show_cut:
        for mask, color, label in (
            (y_prime >= 0, CROWN_TINT, "牙冠半"),
            (y_prime < 0, ROOT_TINT, "牙根半"),
        ):
            ax.plot(points[mask, 0], points[mask, 1], ",", color=color, alpha=0.35, label=label)
    else:
        ax.plot(points[:, 0], points[:, 1], ",", color="lime", alpha=0.15)

    tip = frame.point_at(float(y_prime.max()))
    tail = frame.point_at(float(y_prime.min()))
    ax.annotate(
        "", xy=tip, xytext=tail,
        arrowprops=dict(arrowstyle="-|>", color=AXIS_COLOR, linewidth=2, mutation_scale=18),
    )
    ax.plot([], [], "-", color=AXIS_COLOR, linewidth=2, label="y'（指向牙冠）")
    ax.plot(*frame.origin, "o", color=AXIS_COLOR, markersize=7, markeredgecolor="black")

    if show_cut:
        half = float(np.abs(x_prime).max()) * 1.15
        a = frame.to_image(np.array([-half]), np.array([0.0]))[0]
        b = frame.to_image(np.array([half]), np.array([0.0]))[0]
        ax.plot([a[0], b[0]], [a[1], b[1]], "--", color=CUT_COLOR, linewidth=2,
                label="x'（切分線）")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", help="labelme 的 .json 檔，或含有多個 .json 的資料夾")
    parser.add_argument("--cut", action="store_true",
                        help="畫出 x' 切分線並將兩半染色")
    parser.add_argument("--profile", action="store_true", help="加畫寬度剖面圖")
    parser.add_argument("-o", "--out", help="存檔到此路徑，不開視窗")
    args = parser.parse_args()

    target = Path(args.target)
    json_files = sorted(target.glob("*.json")) if target.is_dir() else [target]
    if not json_files:
        raise SystemExit(f"在 {target} 找不到任何 .json 標註檔")

    import matplotlib

    # 後端必須在 import pyplot 之前設定，而是否需要無視窗模式取決於
    # 有沒有給 -o，所以 matplotlib 要延後到這裡才 import。
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_cjk_font(plt)

    n_cols = len(json_files) * (2 if args.profile else 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 9))
    axes = np.atleast_1d(axes)

    col = 0
    for jf in json_files:
        data = load_annotation(jf)
        img = load_image(data)
        points = mask_points(label_mask(data, LABEL_TOOTH))
        frame = fit_axis(points)

        print(f"{data['imagePath']}  ({data['imageWidth']}x{data['imageHeight']})")
        print("\n".join(describe(frame)))
        print()

        ax = axes[col]
        col += 1
        ax.imshow(img, cmap="gray")
        draw_frame(ax, frame, points, args.cut)
        ax.set_title(
            f"{jf.stem}    長短軸變異數比 = {frame.anisotropy:.2f}    "
            f"牙冠／牙根信心度 = {frame.margin:.2f} 倍",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="lower right", markerscale=8)
        ax.axis("off")

        if args.profile:
            pax = axes[col]
            col += 1
            _, y_prime = frame.to_frame(points)
            centers, widths = width_profile(y_prime)
            pax.plot(centers, widths, "-", color="black", linewidth=1.5)
            pax.axvline(0, color=CUT_COLOR, linestyle="--", linewidth=2, label="切分線")
            pax.set_xlabel("y'  （根尖 <- 0 -> 牙冠）")
            pax.set_ylabel("每箱像素數（正比於寬度）")
            pax.set_title(f"{jf.stem} 寬度剖面", fontsize=10)
            pax.legend(fontsize=8)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=120, bbox_inches="tight")
        print(f"已存檔 -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
