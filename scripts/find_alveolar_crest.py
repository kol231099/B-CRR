"""找齒槽脊的 E、F 兩點——只用原影像與牙齒遮罩。

想法
----
牙周病把骨頭吃掉了，所以沿著牙根往下走，**牙根外側**的東西會改變：

    齒槽脊以上   骨頭沒了，旁邊是軟組織或空隙   → 暗
    齒槽脊以下   骨頭還在                       → 亮

於是在牙根外側的一條窄帶裡，每個高度記錄一個平均亮度，得到一串「上暗
下亮」的數字，齒槽脊就是它跳躍的位置。

這是本專案第一個**往遮罩外面看**的步驟。A/B/C/D/G 都在牙齒內部或表面
上，遮罩本身就帶著答案；E/F 的資訊卻完全在遮罩之外，遮罩只用來決定
「該去哪裡取樣」。

搜尋範圍
--------
縱向限制在 K（根尖）到 L（CEJ）之間。這是解剖上的硬約束而非參數：齒槽骨
不可能高過 CEJ，也不可能低於根尖。

橫向從牙根表面往外取一條窄帶。起點稍微離開表面，避開遮罩最外圈的部分
體積效應；終點不能太遠，否則會吃到鄰牙——根尖片上鄰牙緊貼是常態，而
鄰牙是亮的，會被誤判成骨頭。兩個距離都以 CEJ 寬的比例表示，無量綱。

找交界
------
用階梯模型的變點擬合：假設交界在某個高度，上半段取一個平均值、下半段取
另一個，算總誤差；每個高度都試一遍，誤差最小者即為交界。這跟 find_cej
是同一個套路，只是那裡每段配一條斜線（CEJ 沒有明顯轉角，只有斜率變化），
這裡每段配一個定值（骨頭要嘛在、要嘛不在，預期是真的階梯）。

用法：
    py scripts/find_alveolar_crest.py labeled_PA
    py scripts/find_alveolar_crest.py labeled_PA --outer-fraction 0.20
    py scripts/find_alveolar_crest.py labeled_PA -o alc.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.find_axis_paper import mark, refine_axis  # noqa: E402
from scripts.find_axis_raw import ToothFrame  # noqa: E402
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

LANDMARK_COLOR = "#1f6feb"
DERIVED_COLOR = "#8a8a8a"
SIDE_COLOR = {"E": "#d64545", "F": "#e0a020"}


@dataclass
class CrestFit:
    """單側的掃描剖面與擬合結果。"""

    y: np.ndarray  # 各高度（raw 座標系的 y'）
    brightness: np.ndarray  # 該高度牙根外側窄帶的平均亮度
    level: float  # 擬合出的齒槽脊高度
    bright: float  # 交界以下（有骨）的平均亮度
    dark: float  # 交界以上（無骨）的平均亮度
    sse: float
    sse_single: float  # 單一定值模型的誤差，用來衡量階梯有多明顯
    band: np.ndarray  # 取樣過的像素座標，供作圖

    @property
    def contrast(self) -> float:
        return self.bright - self.dark

    @property
    def improvement(self) -> float:
        """階梯模型比單一定值好多少倍。接近 1 代表根本沒有階梯。"""
        return float(self.sse_single / max(self.sse, 1e-12))


def scan_outside(
    img: np.ndarray, points: np.ndarray, frame: ToothFrame, side: int,
    low: float, high: float, inner: float, outer: float, bin_px: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """沿牙根外側掃描，回傳 (各高度, 平均亮度, 取樣點)。

    每個高度先找出該側牙根表面的位置（該高度 |x'| 最大的遮罩像素），再從
    表面往外 inner~outer 像素取樣。往外的方向直接用 x' —— 牙根大致平行
    長軸，x' 已經很接近表面法線。
    """
    x_prime, y_prime = frame.to_frame(points)
    on_side = (x_prime < 0) if side < 0 else (x_prime >= 0)
    xs, ys = np.abs(x_prime[on_side]), y_prime[on_side]

    n_bins = max(5, int(round((high - low) / bin_px)))
    edges = np.linspace(low, high, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    levels, values, sampled = [], [], []
    for i in range(n_bins):
        in_bin = (ys >= edges[i]) & (ys < edges[i + 1])
        if not in_bin.any():
            continue
        surface = float(xs[in_bin].max())

        offsets = np.arange(inner, outer + 1e-9)
        probe = frame.to_image(side * (surface + offsets),
                              np.full(len(offsets), centers[i]))
        rows = np.round(probe[:, 1]).astype(int)
        cols = np.round(probe[:, 0]).astype(int)
        inside = (rows >= 0) & (rows < img.shape[0]) & (cols >= 0) & (cols < img.shape[1])
        if inside.sum() < 2:
            continue

        levels.append(centers[i])
        values.append(float(img[rows[inside], cols[inside]].mean()))
        sampled.append(np.column_stack([cols[inside], rows[inside]]))

    if len(levels) < 10:
        raise ValueError("可掃描的高度太少")
    return np.array(levels), np.array(values), np.vstack(sampled)


def fit_step(y: np.ndarray, brightness: np.ndarray, min_frac: float = 0.1) -> tuple:
    """掃描每個可能的交界高度，取殘差最小者。

    y 往牙冠方向遞增，所以交界**以下**（低 y、靠根尖）應該是有骨的亮段，
    **以上**是無骨的暗段。要求「亮段確實比暗段亮」——這是解剖事實而非
    可調參數，少了它，最小平方法可能找到上下顛倒的解。
    """
    n = len(y)
    lo, hi = max(2, int(n * min_frac)), min(n - 2, int(n * (1 - min_frac)))
    single = float(((brightness - brightness.mean()) ** 2).sum())

    best = None
    for i in range(lo, hi):
        below, above = brightness[:i], brightness[i:]
        bright, dark = below.mean(), above.mean()
        if bright <= dark:
            continue
        sse = float(((below - bright) ** 2).sum() + ((above - dark) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(y[i]), float(bright), float(dark))

    if best is None:
        raise ValueError("找不到「下亮上暗」的階梯解")
    return (*best, single)


def find_crest(img, mask, result, inner: float, outer: float) -> dict[str, CrestFit]:
    points = mask_points(mask)
    frame = result.frame_raw
    low, high = result.levels_raw["K"], result.levels_raw["L"]

    fits = {}
    for name, side in (("E", -1), ("F", +1)):
        y, brightness, band = scan_outside(img, points, frame, side, low, high, inner, outer)
        sse, level, bright, dark, single = fit_step(y, brightness)
        fits[name] = CrestFit(y, brightness, level, bright, dark, sse, single, band)
    return fits


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", help="labelme 的 .json 檔，或含有多個 .json 的資料夾")
    parser.add_argument("--inner-fraction", type=float, default=0.03,
                        help="取樣起點離牙根表面多遠，取 CEJ 寬的比例，預設 0.03")
    parser.add_argument("--outer-fraction", type=float, default=0.15,
                        help="取樣終點離牙根表面多遠，取 CEJ 寬的比例，預設 0.15")
    parser.add_argument("--drop-apical", type=float, default=0.25)
    parser.add_argument("--cut-at", type=float, default=1.0 / 3.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--surface-fraction", type=float, default=0.10)
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
    fig, axes = plt.subplots(2, len(json_files), figsize=(5.2 * len(json_files), 12),
                             squeeze=False)

    for col, jf in enumerate(json_files):
        data = load_annotation(jf)
        img = load_image(data)
        mask = label_mask(data, LABEL_TOOTH)

        try:
            result = refine_axis(img, mask, args.drop_apical, args.cut_at,
                                 args.threshold, args.surface_fraction)
            width = float(np.hypot(result.landmarks["D"][0] - result.landmarks["C"][0],
                                   result.landmarks["D"][1] - result.landmarks["C"][1]))
            fits = find_crest(img, mask, result,
                              args.inner_fraction * width, args.outer_fraction * width)
        except ValueError as exc:
            print(f"{data['imagePath']}：失敗 - {exc}\n")
            for row in range(2):
                axes[row][col].axis("off")
            axes[0][col].set_title(f"{jf.stem}  失敗", fontsize=11)
            continue

        raw = result.levels_raw
        root = raw["L"] - raw["K"]
        print(f"{data['imagePath']}   CEJ 寬 {width:.0f} px　取樣帶 "
              f"{args.inner_fraction * width:.1f}~{args.outer_fraction * width:.1f} px")
        for name in ("E", "F"):
            f = fits[name]
            loss = (raw["L"] - f.level) / root
            print(f"   {name}: 齒槽脊 y'={f.level:7.1f}   有骨側亮度 {f.bright:.1f}"
                  f"　無骨側 {f.dark:.1f}　對比 {f.contrast:5.1f}"
                  f"　階梯改善 {f.improvement:5.2f} 倍   骨流失 {100 * loss:.1f}%")
        mean_loss = np.mean([(raw["L"] - fits[n].level) / root for n in ("E", "F")])
        print(f"   → 平均骨流失比 ABLR = {mean_loss:.3f}\n")

        # --- 影像 ---
        ax = axes[0][col]
        ax.imshow(img, cmap="gray")
        overlay(ax, mask.shape, mask_points(mask), MASK_RGBA)
        for name in ("E", "F"):
            overlay(ax, mask.shape, fits[name].band, (1.0, 0.55, 0.0, 0.18))

        frame = result.frame_raw
        half = float(np.abs(frame.to_frame(mask_points(mask))[0]).max()) * 1.3
        for lvl, label in ((raw["L"], "L"), (raw["K"], "K")):
            a = frame.to_image(np.array([-half]), np.array([lvl]))[0]
            b = frame.to_image(np.array([half]), np.array([lvl]))[0]
            ax.plot([a[0], b[0]], [a[1], b[1]], ":", color=DERIVED_COLOR, linewidth=0.6)

        for name in ("E", "F"):
            f = fits[name]
            side = -1 if name == "E" else 1
            xs = np.abs(frame.to_frame(mask_points(mask))[0])
            ys = frame.to_frame(mask_points(mask))[1]
            near = np.abs(ys - f.level) < 3
            surface = float(xs[near].max()) if near.any() else half * 0.5
            p = frame.to_image(np.array([side * surface]), np.array([f.level]))[0]
            mark(ax, p, name, LANDMARK_COLOR, 4.5, 10, (6, 3) if side > 0 else (-16, 3))
            a = frame.to_image(np.array([-half]), np.array([f.level]))[0]
            b = frame.to_image(np.array([half]), np.array([f.level]))[0]
            ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=SIDE_COLOR[name],
                    linewidth=0.7, alpha=0.8)

        ax.set_title(f"{jf.stem}    ABLR = {mean_loss:.3f}", fontsize=11)
        ax.axis("off")

        # --- 剖面 ---
        pax = axes[1][col]
        for name in ("E", "F"):
            f = fits[name]
            pax.plot(f.y, f.brightness, "-", color=SIDE_COLOR[name], linewidth=1.0,
                     label=f"{name} 側外緣亮度")
            step = np.where(f.y < f.level, f.bright, f.dark)
            pax.plot(f.y, step, "--", color=SIDE_COLOR[name], linewidth=0.9, alpha=0.9)
            pax.axvline(f.level, color=SIDE_COLOR[name], linestyle=":", linewidth=1.2)
        pax.axvline(raw["L"], color=DERIVED_COLOR, linestyle="-", linewidth=0.8, alpha=0.6)
        pax.set_xlabel("y'　（根尖 ← → CEJ）")
        pax.set_ylabel("牙根外側平均亮度")
        pax.set_title("虛線 = 擬合的階梯，點線 = 齒槽脊，灰線 = CEJ 高度", fontsize=9)
        pax.legend(fontsize=8)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=120, bbox_inches="tight")
        print(f"已存檔 -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
