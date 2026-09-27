"""方法 A2：牙根直線外插 + 鉸鏈擬合，從牙齒遮罩找出 CEJ 的 C、D 兩點。

想法
----
把牙齒看成「一個錐狀牙根，上面額外貼了一層琺瑯質牙冠」。

第一階段先只用牙根那段資料，擬合出「半寬隨高度變化」的直線，並往牙冠
方向外插——這條線代表「如果這顆牙從頭到尾都只是牙根，每個高度該有多寬」。

第二階段把實際寬度減去這條基準線，得到差值曲線：CEJ 以下應該貼著零，
CEJ 以上因為牙冠隆起而往上爬。用鉸鏈函數 max(0, k*(y' - y0)) 去擬合，
解出來的 y0 就是牙冠開始隆起的位置。

先有雞還是先有蛋：第一階段需要知道哪段是牙根，但那要等第二階段算完。
所以用迭代——先用根側一半粗估，得到 y0 後只用 y0 以下重新擬合基準線，
再算一次，直到 y0 不再移動。

已知的系統性偏差
----------------
琺瑯質在 CEJ 處的厚度定義上是零，往上才逐漸增厚。所以緊鄰 CEJ 上方那段
幾乎沒有隆起，差值要爬過一段距離、超過雜訊水準才看得出來。鉸鏈抓到的是
「隆起變得可測量」的位置，**必定偏向牙冠側**。

因此同時輸出第二個估計值：只取差值明顯高於雜訊的區段擬合直線，再往下
外插到差值為零之處。那個零點才接近真實的 CEJ。

只用遮罩、不用多邊形頂點：分割模型輸出的是遮罩，推論時沒有頂點可用。
寬度剖面是數每個高度有幾個像素算出來的，不碰邊界幾何，所以遮罩邊緣的
階梯狀鋸齒影響很小。

用法：
    py scripts/find_cej.py labeled_PA/13.json
    py scripts/find_cej.py labeled_PA
    py scripts/find_cej.py labeled_PA --drop-apical 0.15
    py scripts/find_cej.py labeled_PA -o cej.png
    py scripts/find_cej.py labeled_PA --drop-apical 0.5 -o cej.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.find_axis_raw import ToothFrame, fit_axis  # noqa: E402
from scripts.labelme_io import (  # noqa: E402
    LABEL_ENAMEL,
    LABEL_TOOTH,
    label_mask,
    load_annotation,
    load_image,
    MASK_RGBA,
    mask_points,
    overlay,
    setup_cjk_font,
)

SIDE_COLOR = {"C": "#ff3b30", "D": "#ffd54a"}


@dataclass
class SideFit:
    """單側的擬合結果，欄位齊全到足以完整重畫整個過程。"""

    y: np.ndarray  # 各高度（y'），往牙冠遞增
    w: np.ndarray  # 該高度的半寬
    root_a: float  # 牙根基準線截距
    root_b: float  # 牙根基準線斜率
    deviation: np.ndarray  # 實際半寬減去基準線
    onset: float  # 鉸鏈轉折處（牙冠開始隆起）
    slope_k: float  # 鉸鏈斜率，即牙冠隆起速率
    noise: float  # 牙根段殘差標準差，作為雜訊水準
    zero_cross: float | None  # 外插回差值為零之處，較接近真實 CEJ
    crown_a: float | None  # 牙冠段那條線（差值空間）的截距
    crown_b: float | None  # 同上的斜率；與牙根基準線的交點即 zero_cross
    iterations: int
    converged: bool

    def root_line(self, y=None) -> np.ndarray:
        y = self.y if y is None else y
        return self.root_a + self.root_b * y

    def hinge(self, y=None) -> np.ndarray:
        y = self.y if y is None else y
        return np.maximum(0.0, self.slope_k * (y - self.onset))

    def half_width_at(self, level: float) -> float:
        return float(np.interp(level, self.y, self.w))


def half_width_profile(
    points: np.ndarray, frame: ToothFrame, side: int
) -> tuple[np.ndarray, np.ndarray]:
    """單側在各高度的半寬。

    side 為 -1 取左側（x' < 0）、+1 取右側。分箱寬度固定為 1 像素，與
    影像解析度一致——分得更細不會增加資訊，更粗則會損失資訊。每箱取離
    長軸最遠的距離，也就是輪廓的位置。
    """
    x_prime, y_prime = frame.to_frame(points)
    keep = (x_prime < 0) if side < 0 else (x_prime >= 0)
    x_prime, y_prime = np.abs(x_prime[keep]), y_prime[keep]

    n_bins = max(10, int(round(np.ptp(y_prime))))
    edges = np.linspace(y_prime.min(), y_prime.max(), n_bins + 1)
    index = np.clip(np.digitize(y_prime, edges) - 1, 0, n_bins - 1)

    widths = np.zeros(n_bins)
    np.maximum.at(widths, index, x_prime)
    counts = np.bincount(index, minlength=n_bins)

    centers = (edges[:-1] + edges[1:]) / 2.0
    ok = counts > 0
    return centers[ok], widths[ok]


def prepare_side(
    points: np.ndarray, frame: ToothFrame, side: int, drop_apical: float
) -> tuple[np.ndarray, np.ndarray]:
    """取出要拿去擬合的那段剖面。

    只保留**該側最寬處（外形高點）以下**的部分：再往咬合面走寬度會反過來
    變小，那是第三種走勢，不屬於本模型。這個範圍剛好也對應論文所說的
    「邊緣嵴以下的牙冠」。

    drop_apical 再捨去根尖端一段。根尖會急遽收斂到一點，那段並不符合
    「牙根是直錐體」的模型；若納入擬合，會把基準線的斜率拉陡，外插上去
    甚至會高過實際寬度，使差值全部變成負的而完全無解。預設捨去三分之一，
    沿用論文決定長軸時「忽略根尖三分之一」的同一慣例。
    """
    y, w = half_width_profile(points, frame, side)
    order = np.argsort(y)
    y, w = y[order], w[order]

    y, w = y[y <= y[int(np.argmax(w))]], w[y <= y[int(np.argmax(w))]]

    if drop_apical > 0:
        keep = y >= y.min() + drop_apical * np.ptp(y)
        y, w = y[keep], w[keep]

    return y, w


def hinge_onset(y: np.ndarray, deviation: np.ndarray, min_root_frac: float = 0.5):
    """掃描每個可能的轉折位置，取殘差最小者。

    鉸鏈模型 max(0, k*(y - y0)) 沒有常數項——轉折以下的差值理應為零，
    因為基準線就是用那段資料擬合出來的。固定 y0 之後模型對 k 是線性的，
    可直接解出最佳 k，所以只需要掃描 y0。

    兩個約束都來自解剖，不是為了調參數：

    * **k 必須為正。**牙冠是往外隆起的，差值只能往上爬。少了這個約束，
      最小平方法會找到「k 為負、轉折落在牙根深處」的自洽錯誤解——實測
      十個側面中有五個掉進這個陷阱。
    * **轉折以下至少要佔剖面的一半。**牙根本來就佔牙齒長度約七成，而這
      條剖面又已經切掉了外形高點以上的牙冠，牙根佔比只會更高。少了這個
      約束，轉折可以滑到根尖附近，使得「牙根段」只剩根尖那一小截，基準
      線便被根尖的急遽收斂帶歪。
    """
    n = len(y)
    lo, hi = max(2, int(n * min_root_frac)), n - 2
    best = None
    for i in range(lo, hi):
        basis = np.maximum(0.0, y - y[i])
        denom = float(basis @ basis)
        if denom < 1e-12:
            continue
        k = float(basis @ deviation) / denom
        if k <= 0:
            continue
        sse = float(((k * basis - deviation) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(y[i]), k)
    if best is None:
        raise ValueError("找不到符合約束（k > 0）的鉸鏈解")
    return best


def fit_side(y: np.ndarray, w: np.ndarray, max_iterations: int = 10, tol: float = 0.5) -> SideFit:
    """迭代求解牙根基準線與鉸鏈轉折點。"""
    onset = y.min() + 0.5 * np.ptp(y)  # 起始猜測：根側一半都算牙根
    root_a = root_b = 0.0
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        root = y <= onset
        if root.sum() < 5:
            break
        design = np.column_stack([np.ones(int(root.sum())), y[root]])
        (root_a, root_b), *_ = np.linalg.lstsq(design, w[root], rcond=None)

        deviation = w - (root_a + root_b * y)
        _, new_onset, slope_k = hinge_onset(y, deviation)

        if abs(new_onset - onset) < tol:
            onset = new_onset
            converged = True
            break
        onset = new_onset

    deviation = w - (root_a + root_b * y)
    _, onset, slope_k = hinge_onset(y, deviation)

    # 牙根段的殘差散布，作為「差值多大才算真的隆起」的雜訊尺度。
    root = y <= onset
    noise = float(np.std(deviation[root])) if root.sum() >= 3 else 0.0

    # 只取明顯高於雜訊的區段擬合直線，往下外插到差值為零，藉此避開
    # 「琺瑯質在 CEJ 附近太薄、偵測不到」造成的偏牙冠側系統性偏差。
    zero_cross = crown_a = crown_b = None
    clear = deviation > 3 * noise
    if noise > 0 and clear.sum() >= 5:
        design = np.column_stack([np.ones(int(clear.sum())), y[clear]])
        (intercept, slope), *_ = np.linalg.lstsq(design, deviation[clear], rcond=None)
        if abs(slope) > 1e-9:
            zero_cross = float(-intercept / slope)
            crown_a, crown_b = float(intercept), float(slope)

    return SideFit(
        y=y, w=w, root_a=float(root_a), root_b=float(root_b), deviation=deviation,
        onset=float(onset), slope_k=float(slope_k), noise=noise,
        zero_cross=zero_cross, crown_a=crown_a, crown_b=crown_b,
        iterations=iterations, converged=converged,
    )


def cej_points(points: np.ndarray, frame: ToothFrame, drop_apical: float) -> dict:
    """C、D 兩點在本座標系中的 (x', y')。

    優先採用外插零點——它避開了 CEJ 附近琺瑯質太薄造成的偏差；若算不出來
    才退回鉸鏈轉折點。
    """
    out = {}
    for name, side in (("C", -1), ("D", +1)):
        y, w = prepare_side(points, frame, side, drop_apical)
        fit = fit_side(y, w)
        level = fit.zero_cross if fit.zero_cross is not None else fit.onset
        out[name] = (side * fit.half_width_at(level), level)
    return out


def upper_crown_region(
    points: np.ndarray, frame: ToothFrame, cej: dict, fraction: float = 0.0
) -> tuple[np.ndarray, float, tuple]:
    """切出牙冠。

    切割線平行於 CD，位置由 fraction 決定：沿長軸從 CEJ 線與長軸的交點 L
    走向牙冠頂端，0 代表切在 L（取整個牙冠），0.5 代表取上半個牙冠。

    回傳 (該區域的遮罩像素座標, 切割高度 y'_cut, 切割線的兩個端點)。
    """
    (xc, yc), (xd, yd) = cej["C"], cej["D"]

    # CEJ 線與長軸（x' = 0）的交點高度
    s = -xc / (xd - xc) if abs(xd - xc) > 1e-9 else 0.5
    y_l = yc + s * (yd - yc)

    x_prime, y_prime = frame.to_frame(points)
    y_top = float(y_prime.max())
    y_cut = y_l + fraction * (y_top - y_l)

    # 切割線通過長軸上的 (0, y_cut)，方向與 CD 相同
    dx, dy = xd - xc, yd - yc
    side_of = lambda x, y: (x - 0.0) * dy - (y - y_cut) * dx
    reference = side_of(0.0, y_top)  # 牙冠頂端所在的那一側才是要保留的
    keep = side_of(x_prime, y_prime) * reference > 0

    half = float(np.abs(x_prime).max()) * 1.2
    norm = np.hypot(dx, dy)
    ends = (
        frame.to_image(np.array([-half * dx / norm]), np.array([y_cut - half * dy / norm]))[0],
        frame.to_image(np.array([half * dx / norm]), np.array([y_cut + half * dy / norm]))[0],
    )
    return points[keep], y_cut, ends


def enamel_reference(data: dict, frame: ToothFrame) -> dict[str, float] | None:
    """由琺瑯質標註推出的真實 CEJ 高度，作為驗證基準。

    琺瑯質往牙頸方向逐漸變薄到消失，該側琺瑯質的最根尖端就是 CEJ。
    只有還留著琺瑯質標註的影像才有這個基準。
    """
    try:
        enamel = mask_points(label_mask(data, LABEL_ENAMEL))
    except ValueError:
        return None
    x_prime, y_prime = frame.to_frame(enamel)
    return {"C": float(y_prime[x_prime < 0].min()), "D": float(y_prime[x_prime >= 0].min())}


def report(name: str, fit: SideFit, truth: float | None, tooth_length: float) -> list[str]:
    status = "收斂" if fit.converged else "**未收斂**"
    lines = [
        f"  {name} 側（迭代 {fit.iterations} 次，{status}）",
        f"     牙根基準線：半寬 = {fit.root_a:.2f} + {fit.root_b:.4f} * y'"
        f"     雜訊 σ = {fit.noise:.2f} px",
        f"     鉸鏈轉折 y' = {fit.onset:7.1f}   隆起速率 k = {fit.slope_k:.4f}",
    ]
    if fit.zero_cross is not None:
        lines.append(f"     外插零點 y' = {fit.zero_cross:7.1f}")
    if truth is not None:
        def err(v):
            return f"{v - truth:+6.1f} px（{100 * (v - truth) / tooth_length:+.2f}% 牙長）"
        lines.append(f"     真實 CEJ y' = {truth:7.1f}")
        lines.append(f"     → 鉸鏈誤差 {err(fit.onset)}")
        if fit.zero_cross is not None:
            lines.append(f"     → 外插誤差 {err(fit.zero_cross)}")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", help="labelme 的 .json 檔，或含有多個 .json 的資料夾")
    parser.add_argument("--drop-apical", type=float, default=1.0 / 2.0,
                        help="捨去根尖端的比例，預設 1/3（沿用論文忽略根尖三分之一的慣例）")
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

    fig, axes = plt.subplots(3, len(json_files), figsize=(6.2 * len(json_files), 15),
                             squeeze=False)

    for col, jf in enumerate(json_files):
        data = load_annotation(jf)
        img = load_image(data)
        mask = label_mask(data, LABEL_TOOTH)
        points = mask_points(mask)
        frame = fit_axis(points)
        tooth_length = float(np.ptp(frame.to_frame(points)[1]))
        truth = enamel_reference(data, frame)

        fits, corners, failures = {}, {}, {}
        for name, side in (("C", -1), ("D", +1)):
            y, w = prepare_side(points, frame, side, args.drop_apical)
            try:
                fit = fit_side(y, w)
            except ValueError as exc:
                failures[name] = str(exc)
                continue
            fits[name] = fit
            corners[name] = frame.to_image(
                np.array([side * fit.half_width_at(fit.onset)]), np.array([fit.onset])
            )[0]

        print(f"{data['imagePath']}   牙長 {tooth_length:.0f} px")
        for name in ("C", "D"):
            if name in failures:
                print(f"  {name} 側：擬合失敗 - {failures[name]}")
            else:
                print("\n".join(report(name, fits[name], truth[name] if truth else None, tooth_length)))
        print()

        # --- 第一列：影像與定位結果 ---
        ax = axes[0][col]
        ax.imshow(img, cmap="gray")
        overlay(ax, mask.shape, points, MASK_RGBA)
        for name in corners:
            p = corners[name]
            ax.plot(p[0], p[1], "o", color=SIDE_COLOR[name], markersize=4,
                    markeredgecolor="black", markeredgewidth=0.6)
            ax.annotate(name, (p[0], p[1]), color=SIDE_COLOR[name], fontsize=10,
                        fontweight="bold", xytext=(7, 3), textcoords="offset pixels")
        if len(corners) == 2:
            ax.plot([corners["C"][0], corners["D"][0]], [corners["C"][1], corners["D"][1]],
                    "-", color="#00e5ff", linewidth=1.0, label="CEJ 線")
        ax.set_title(f"{jf.stem}   牙長 {tooth_length:.0f} px", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        ax.axis("off")

        # --- 第二列：半寬剖面與牙根基準線 ---
        pax = axes[1][col]
        for name in fits:
            fit, color = fits[name], SIDE_COLOR[name]
            pax.plot(fit.y, fit.w, "-", color=color, linewidth=1.0, label=f"{name} 側實際半寬")
            pax.plot(fit.y, fit.root_line(), "--", color=color, linewidth=0.9, alpha=0.85,
                     label=f"{name} 側牙根基準線（外插）")
            root = fit.y <= fit.onset
            pax.plot(fit.y[root], fit.w[root], "-", color=color, linewidth=2.5, alpha=0.30)
            pax.axvline(fit.onset, color=color, linestyle=":", linewidth=1.2)
            if truth:
                pax.axvline(truth[name], color=color, linestyle="-", linewidth=0.9, alpha=0.5)
        pax.set_xlabel("y'　（根尖 ← → 牙冠）")
        pax.set_ylabel("半寬（像素）")
        pax.set_title("粗線 = 用來擬合基準線的牙根段；點線 = 鉸鏈轉折；細實線 = 真實 CEJ",
                      fontsize=9)
        pax.legend(fontsize=7)

        # --- 第三列：差值曲線與鉸鏈 ---
        dax = axes[2][col]
        for name in fits:
            fit, color = fits[name], SIDE_COLOR[name]
            dax.plot(fit.y, fit.deviation, "-", color=color, linewidth=1.0,
                     label=f"{name} 側差值（實際 - 基準線）")
            dax.plot(fit.y, fit.hinge(), "--", color=color, linewidth=0.9, alpha=0.9,
                     label=f"{name} 側鉸鏈擬合")
            dax.axhspan(-3 * fit.noise, 3 * fit.noise, color=color, alpha=0.07)
            dax.axvline(fit.onset, color=color, linestyle=":", linewidth=1.2)
            if fit.zero_cross is not None:
                dax.axvline(fit.zero_cross, color=color, linestyle="-.", linewidth=1.1)
            if truth:
                dax.axvline(truth[name], color=color, linestyle="-", linewidth=0.9, alpha=0.5)
        dax.axhline(0, color="black", linewidth=0.6)
        dax.set_xlabel("y'　（根尖 ← → 牙冠）")
        dax.set_ylabel("差值（像素）")
        dax.set_title("點線 = 鉸鏈轉折；點劃線 = 外插零點；細實線 = 真實 CEJ；淡色帶 = ±3σ 雜訊",
                      fontsize=9)
        dax.legend(fontsize=7)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=110, bbox_inches="tight")
        print(f"已存檔 -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
