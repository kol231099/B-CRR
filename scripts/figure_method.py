"""產生論文的兩張方法流程圖：從牙齒遮罩到 CRR。

拆成兩張是因為單張塞不下，而且切點很自然——前後正好是兩件不同的事：

    圖一　牙齒座標系與 CEJ　　除了輸入那格，**全部只用形狀**
    圖二　特徵點與測量　　　　需要灰階的步驟，以及論文長軸與最終測量

圖一整張都不碰灰階，這件事本身就是本方法的主張之一：CEJ 不需要影像、也不需要
琺瑯質標注。

版面依循流程圖的通用慣例
------------------------
* 格子上不寫說明文字，全部說明留給圖說（`caption_one`、`caption_two`）
* 字母標在左下角
* 綠色箭頭串起流程，排不下就折行繞回
* 同一步驟若含多張圖，用圓角白框框成一組

用法：
    py scripts/figure_method.py
    py scripts/figure_method.py --source labeled_PA/13.json
    py scripts/figure_method.py --source labeled_PA/147.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.find_alveolar_crest import find_crest  # noqa: E402
from scripts.find_axis_raw import fit_axis  # noqa: E402
from scripts.find_cej import cej_points, fit_side, prepare_side, upper_crown_region  # noqa: E402
from scripts.find_ridge import residual_map, ridge_points  # noqa: E402
from scripts.labelme_io import (  # noqa: E402
    MASK_RGBA,
    REGION_RGBA,
    load_tooth,
    overlay,
    setup_cjk_font,
)
from scripts.measure import draw as draw_result, measure  # noqa: E402

# 畫布放大得多、字級與線寬放大得少，文字與線條因此相對於影像變小，圖表也拿到
# 更多空間。兩者同步放大只會得到一張更大但一樣擁擠的圖。
FIG_SCALE = 2.6
S = 1.7  # 字級與線寬的倍率

WIDTH_IN = 175 / 25.4 * FIG_SCALE
# 字級一律以**實際點數**設定，不再乘 S（那個倍率只留給線寬與標記大小）。
# 級距刻意壓縮到最大與最小差 1.9 倍：拉大之前差到 2.6 倍，圖表裡的刻度與圖例
# 小到得湊近才看得清，格子字母卻大得搶走視線。
LETTER_SIZE = 28       # 格子左下的 A–F、群組框的字母
LANDMARK_SIZE = 16     # 特徵點的字母 A–G、剖面圖上的 C／D
NOTE_SIZE = 14         # y′ (crown)、crown、root
AXIS_SIZE = 14         # 軸標題
TICK_SIZE = 12         # 刻度數字
LEGEND_SIZE = 12       # 圖例

ARROW_COLOR = "#22ab55"
ARROW_WIDTH = 2.4 * S

LANDMARK = "#1f6feb"   # 論文定義的特徵點
DERIVED = "#8a8a8a"    # 推演點與初始長軸
FIT = "#d64545"        # 擬合結果
AUX = "#00a884"        # 主軸、基準線、裁切線
SIDE_C = "#5aa8e0"     # C 側（淺藍）
SIDE_D = "#a882d8"     # D 側（淺紫）
MASK_RED = "#ff0800"   # 淺紅遮罩
ROUNDING = 0.05


# --------------------------------------------------------------------------
# 共用的繪圖工具
# --------------------------------------------------------------------------

def round_corners(ax):
    """把格子裡的影像裁成圓角。"""
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (ROUNDING, ROUNDING), 1 - 2 * ROUNDING, 1 - 2 * ROUNDING,
        boxstyle=f"round,pad={ROUNDING},rounding_size={ROUNDING}",
        transform=ax.transAxes, facecolor="none", edgecolor="none",
    )
    ax.add_patch(patch)
    for image in ax.images:
        image.set_clip_path(patch)


def stroke(width=2.0, colour="black"):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=width * S, foreground=colour)]


def letter(ax, text, on_dark: bool = True):
    """字母標在左下角。"""
    ax.text(0.05, 0.03, text, transform=ax.transAxes, fontsize=LETTER_SIZE,
            fontweight="bold", va="bottom", ha="left",
            color="white" if on_dark else "black",
            path_effects=stroke(2.2) if on_dark else None, zorder=10)


def show_mask(ax, mask, colour=None):
    """純形狀的格子：黑底，遮罩為白色（或指定顏色）。"""
    from matplotlib.colors import to_rgb

    rgb = np.zeros((*mask.shape, 3))
    rgb[mask > 0] = to_rgb(colour) if colour else (1.0, 1.0, 1.0)
    ax.imshow(rgb)


def show_image(ax, img, mask):
    """用到灰階的格子：原片加上淺綠遮罩。"""
    ax.imshow(img, cmap="gray")
    ys, xs = np.nonzero(mask)
    overlay(ax, mask.shape, np.column_stack([xs, ys]), MASK_RGBA)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def apply_crop(fig, axes, region, shape, margin: float = 0.13):
    """依**實際**的 axes 長寬比裁切，讓牙齒填滿格子。

    `region` 是要框住的點集 (N,2)；不同格子可以框不同的東西（整顆牙，或只有
    琺瑯質那條帶）。

    裁切框的長寬比必須跟格子一致，否則 imshow 會維持像素比例而在兩側留白，
    看起來像格子大小不一。格子的實際尺寸要等版面排完才知道，所以得先 draw
    一次拿到真實位置，再回頭設 xlim/ylim。裁切框還要夾回影像範圍內，否則
    牙齒靠近感光片邊緣時格子裡會出現一塊底色。
    """
    fig.canvas.draw()
    height_px, width_px = shape
    x0, x1 = float(region[:, 0].min()), float(region[:, 0].max())
    y0, y1 = float(region[:, 1].min()), float(region[:, 1].max())
    pad = margin * max(x1 - x0, y1 - y0)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad

    def clamp(lo, hi, limit):
        span = hi - lo
        if span >= limit:
            return 0.0, float(limit)
        if lo < 0:
            return 0.0, span
        if hi > limit:
            return limit - span, float(limit)
        return lo, hi

    for ax in axes:
        # 必須用 gridspec 的**儲存格**，不能用 ax.get_position()。後者回傳的是
        # 已被 aspect="equal" 縮過的框，而縮放依據是**當下**的資料範圍（此刻
        # 還是整張影像），拿它算長寬比會把裁切框撐回原來的大小。
        box = ax.get_subplotspec().get_position(fig)
        aspect = (box.width * fig.get_figwidth()) / (box.height * fig.get_figheight())
        width, height = x1 - x0, y1 - y0
        if width / height < aspect:
            extra = aspect * height - width
            left, right = clamp(x0 - extra / 2, x1 + extra / 2, width_px)
            top, bottom = clamp(y0, y1, height_px)
        else:
            extra = width / aspect - height
            left, right = clamp(x0, x1, width_px)
            top, bottom = clamp(y0 - extra / 2, y1 + extra / 2, height_px)
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        round_corners(ax)


def style_chart(ax, xlabel: str, ylabel: str, legend_loc: str = "best"):
    ax.tick_params(labelsize=TICK_SIZE, length=2 * S, pad=1)
    ax.set_xlabel(xlabel, fontsize=AXIS_SIZE, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=AXIS_SIZE, labelpad=2)
    for side in ax.spines.values():
        side.set_linewidth(0.5 * S)
    if legend_loc:
        ax.legend(fontsize=LEGEND_SIZE, frameon=False, loc=legend_loc,
                  handlelength=1.4, borderpad=0.2, labelspacing=0.3)


def label_point(ax, point, name, colour=LANDMARK, offset=(5, 4)):
    """畫特徵點與它的字母。

    字母一律**白字黑邊**——用顏色當字色時，字會隨背景明暗而融進去。顏色的
    資訊由點本身承載就夠了。
    """
    ax.plot(point[0], point[1], "o", color=colour, markersize=3.4 * S,
            markeredgecolor="black", markeredgewidth=0.5 * S, zorder=6)
    ax.annotate(name, point, textcoords="offset points",
                xytext=(offset[0] * S, offset[1] * S), fontsize=LANDMARK_SIZE,
                color="white", fontweight="bold", zorder=7,
                path_effects=stroke(2.0))


def group_box(fig, axes, label: str, pad: float = 0.010):
    """把同屬一個步驟的多張圖框成一組，字母標在框的左下角。

    範圍要用 **tight bbox**（含刻度、軸標題、色標），不能用 ax.get_position()
    ——後者只有繪圖區的矩形，框出來會切掉軸上的數字。

    但回傳時要另外附上**內容的垂直中心**：tight bbox 會因為刻度與軸標題而往
    下長出一截，用它的中心當箭頭的錨點，箭頭就會歪掉。框歸框、錨點歸錨點。
    """
    from matplotlib.patches import FancyBboxPatch

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    boxes = [ax.get_tightbbox(renderer).transformed(inverse) for ax in axes]
    x0 = min(b.x0 for b in boxes) - pad
    x1 = max(b.x1 for b in boxes) + pad
    y0 = min(b.y0 for b in boxes) - pad
    y1 = max(b.y1 for b in boxes) + pad

    fig.add_artist(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        transform=fig.transFigure, facecolor="white",
        edgecolor="#777777", linewidth=0.7 * S, zorder=0,
    ))
    fig.text(x0 + 0.010, y0 + 0.008, label, fontsize=LETTER_SIZE,
             fontweight="bold", va="bottom", ha="left", color="white",
             path_effects=stroke(2.6), zorder=11)

    content = [ax.get_position() for ax in axes]
    centre = (min(b.y0 for b in content) + max(b.y1 for b in content)) / 2
    return x0, y0, x1, y1, centre


def arrow(fig, start, end, elbow=None):
    """流程箭頭。elbow 給定時走折線（折行繞回下一列）。"""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path as MplPath

    style = dict(arrowstyle=f"-|>,head_length={4.5 * S},head_width={3 * S}",
                 color=ARROW_COLOR, linewidth=ARROW_WIDTH, zorder=12)
    if elbow is None:
        fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
                                       shrinkA=0, shrinkB=0, **style))
        return
    vertices = [start, *elbow, end]
    path = MplPath(vertices, [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 1))
    fig.add_artist(FancyArrowPatch(path=path, transform=fig.transFigure,
                                   shrinkA=0, shrinkB=0, **style))


def connect(fig, boxes, pairs, gap=0.006):
    """依 (來源, 目標) 的順序畫箭頭，走向由兩格的相對位置自動判斷。"""
    for src, dst in pairs:
        sx0, sy0, sx1, sy1, s_cy = boxes[src]
        dx0, dy0, dx1, dy1, d_cy = boxes[dst]
        if dx0 >= sx1 - 0.002:                          # 目標在右邊：水平
            # 兩端取同一個高度，否則刻度造成的中心差會讓箭頭斜掉
            level = (s_cy + d_cy) / 2
            arrow(fig, (sx1 + gap, level), (dx0 - gap, level))
        elif dy1 <= sy0 + 0.002 and abs((dx0 + dx1) - (sx0 + sx1)) < 0.16:
            centre = (dx0 + dx1) / 2                    # 目標在正下方：垂直
            arrow(fig, (centre, sy0 - gap), (centre, dy1 + gap))
        else:                                           # 折行繞回
            turn = min(sx1 + 0.012, 0.995)
            # 橫向那段必須高於目標格子的上緣，否則會穿過同一列的其他格子；
            # 而且要高出一段，最後「往下插進去」那節才夠長畫得出箭頭。
            gap_y = min(max((sy0 + dy1) / 2, dy1 + 0.035), sy0 - 0.008)
            centre = (dx0 + dx1) / 2
            arrow(fig, (sx1 + gap, s_cy), (centre, dy1 + gap),
                  elbow=[(turn, s_cy), (turn, gap_y), (centre, gap_y)])


def axis_arrow(ax, frame, points, colour, label, shape, overshoot=0.07):
    """畫長軸，並在牙冠端加箭頭與標籤。

    兩端各伸出一段，強調它是一條貫穿整顆牙的軸而非某段線；箭頭指向牙冠端，
    表示這個座標系的 y′ **朝牙冠遞增**——方向不是隨意選的，而是由牙冠半與
    牙根半的形狀差異判定出來的。
    """
    from matplotlib.patches import FancyArrowPatch

    _, y_prime = frame.to_frame(points)
    span = float(np.ptp(y_prime))
    tail = frame.point_at(float(y_prime.min()) - overshoot * span)
    head = frame.point_at(float(y_prime.max()) + overshoot * span)
    # 牙齒常常貼著感光片邊緣，伸出的那一截會落到影像外而被裁掉——夾回來。
    height, width = shape
    head = np.clip(head, [6, 6], [width - 7, height - 7])
    tail = np.clip(tail, [6, 6], [width - 7, height - 7])

    # head_length/head_width 的單位是**點**，不隨畫布放大，所以在大尺寸的圖上
    # 得自己加大，否則箭頭小到看不出來。
    ax.add_patch(FancyArrowPatch(
        tail, head, arrowstyle=f"-|>,head_length={6 * S},head_width={2.8 * S}",
        color=colour, linewidth=0.9 * S, shrinkA=0, shrinkB=0, zorder=5))
    # 標籤用 **axes 座標**擺在右上角。用資料座標會落到裁切框外，而 annotate
    # 的文字不受 axes 裁切，會直接畫到格子外面去。牙齒經裁切後一律置中且
    # 接近垂直，固定位置就在箭頭旁邊。
    ax.text(0.60, 0.92, label, transform=ax.transAxes, fontsize=NOTE_SIZE,
            color="white", fontweight="bold", ha="left", va="center", zorder=7,
            path_effects=stroke(2.4))


def half_mask_strip(ax, mask, frame, side, y_range, resolution: float = 1.6):
    """把遮罩沿長軸剖半、轉成橫躺，並讓**兩半的外緣朝同一側**。

    縱軸取 |x′|（離長軸的距離），所以兩半都以長軸貼著格子底部、外緣朝上，
    可以直接對照，不必在腦中鏡像一次。這也讓縱軸恰好等於「半寬」——擬合出的
    牙根基準線與牙冠隆起線因此能原樣疊在遮罩上，看得到它們與輪廓的關係。

    不用旋轉影像的方式製作——旋轉會插值、邊緣變糊；直接把遮罩像素的
    (x′, y′) 座標丟進格點即可。格點要**略粗於原像素**（resolution > 1），
    否則座標系轉過一個角度後會有格子取不到樣本，遮罩內部出現一片小孔。
    """
    coords = np.column_stack(np.nonzero(mask)[::-1]).astype(float)
    x_prime, y_prime = frame.to_frame(coords)
    keep = (x_prime < 0) if side < 0 else (x_prime >= 0)
    x_prime, y_prime = np.abs(x_prime[keep]), y_prime[keep]

    x_hi = float(x_prime.max())
    n_row = int(x_hi / resolution) + 1
    n_col = int((y_range[1] - y_range[0]) / resolution) + 1
    grid = np.zeros((n_row, n_col), bool)
    rows = np.clip((x_prime / resolution).astype(int), 0, n_row - 1)
    cols = np.clip(((y_prime - y_range[0]) / resolution).astype(int), 0, n_col - 1)
    grid[rows, cols] = True

    rgb = np.zeros((n_row, n_col, 3))
    rgb[grid] = 1.0
    ax.imshow(rgb, origin="lower", aspect="auto",
              extent=(y_range[0], y_range[1], 0.0, x_hi))
    ax.set_xlim(*y_range)
    return x_hi


# --------------------------------------------------------------------------
# 圖一：牙齒座標系與 CEJ
# --------------------------------------------------------------------------

def panel_frame(ax, mask, points, frame):
    """B　長軸與它的方向。"""
    show_mask(ax, mask)
    axis_arrow(ax, frame, points, AUX, "y′ (crown)", mask.shape)


def panel_cej(panels, mask, points, frame, drop_apical):
    """C　CEJ：牙根基準線與牙冠隆起線的交點。

    `panels` 是 ((chart, strip), (chart, strip))，D 側在上、C 側在下。**兩側各自
    一組**——疊在同一張圖上時四條擬合線互相穿插，看不出哪條配哪條；分開之後
    每一側的剖面圖與它自己的剖半遮罩上下相鄰，可以直接對照。

    四張圖共用同一個橫軸。兩張剖面圖也共用同一個縱軸範圍，兩側才比得出差異。
    擬合線同時畫在剖面圖與遮罩上——縱軸都是半寬，線可以原樣疊上去，看得到它
    與牙齒輪廓的關係。
    """
    _, y_all = frame.to_frame(points)
    y_range = (float(y_all.min()), float(y_all.max()))

    fits = {}
    for name, side in (("D", +1), ("C", -1)):
        y, w = prepare_side(points, frame, side, drop_apical)
        fits[name] = (fit_side(y, w), y, w)

    widths = np.concatenate([w for _, _, w in fits.values()])
    span = float(np.ptp(widths))
    w_range = (float(widths.min()) - 0.08 * span, float(widths.max()) + 0.08 * span)

    for (name, side, colour), (chart, strip) in zip(
            (("D", +1, SIDE_D), ("C", -1, SIDE_C)), panels):
        fit, y, w = fits[name]
        x_hi = half_mask_strip(strip, mask, frame, side, y_range)

        chart.plot(y, w, "-", color=colour, linewidth=1.1 * S, label=f"{name} 側半寬")

        root_span = np.array([y.min(), y_range[1]])
        crown_span = (np.array([fit.zero_cross - 0.22 * np.ptp(y), y.max()])
                      if fit.crown_a is not None else None)

        for target, is_chart in ((chart, True), (strip, False)):
            target.plot(root_span, fit.root_a + fit.root_b * root_span, "--",
                        color=colour, linewidth=0.9 * S,
                        label="牙根基準線（外插）" if is_chart else None)
            if crown_span is None:
                continue
            target.plot(crown_span,
                        (fit.root_a + fit.root_b * crown_span)
                        + (fit.crown_a + fit.crown_b * crown_span),
                        ":", color=colour, linewidth=1.2 * S,
                        label="牙冠隆起線（回插）" if is_chart else None)
            level = fit.zero_cross
            target.plot(level, fit.root_a + fit.root_b * level, "o", color=colour,
                        markersize=4.5 * S, markeredgecolor="black",
                        markeredgewidth=0.6 * S, zorder=8)
            target.annotate(name, (level, fit.root_a + fit.root_b * level),
                            textcoords="offset points", xytext=(7 * S, -14 * S),
                            fontsize=LANDMARK_SIZE, color="white", fontweight="bold",
                            zorder=9, path_effects=stroke(2.0))

        chart.set_xlim(*y_range)
        chart.set_ylim(*w_range)
        chart.set_xticklabels([])
        style_chart(chart, "", "半寬 (px)", "upper left")

        strip.set_ylim(0.0, x_hi * 1.04)
        strip.set_yticks([])
        strip.set_xticks([])
        for spine in strip.spines.values():
            spine.set_visible(False)
        strip.set_facecolor("black")
        round_corners(strip)

    # 四張圖共用橫軸，刻度與標題只在最下面那張標一次
    bottom = panels[-1][1]
    bottom.set_xticks(panels[-1][0].get_xticks())
    bottom.set_xlim(*y_range)
    bottom.tick_params(labelsize=TICK_SIZE, length=2 * S, pad=1,
                       bottom=True, labelbottom=True)
    bottom.set_xlabel("沿長軸的高度 y′ (px)", fontsize=AXIS_SIZE, labelpad=2)


def panel_cd(ax, mask, points, frame, cej):
    """D　C、D 兩點與它們的連線——這條線把牙齒分成牙冠與牙根。"""
    show_mask(ax, mask)
    pts = {}
    for name in ("C", "D"):
        x_p, y_p = cej[name]
        pts[name] = frame.to_image(np.array([x_p]), np.array([y_p]))[0]

    a, b = pts["C"], pts["D"]
    direction = (b - a) / (np.linalg.norm(b - a) + 1e-9)
    tail, head = a - direction * 45, b + direction * 45
    ax.plot([tail[0], head[0]], [tail[1], head[1]], "-", color=LANDMARK,
            linewidth=1.5 * S, zorder=4)
    for name in ("C", "D"):
        label_point(ax, pts[name], name, LANDMARK, offset=(6, -15))

    # crown 在線的牙冠側、root 在牙根側；哪一側是牙冠由座標系決定，不是猜的
    centre = (a + b) / 2
    for text, sign in (("crown", +1), ("root", -1)):
        ax.annotate(text, centre + frame.y_axis * sign * 95, fontsize=NOTE_SIZE,
                    color="white", ha="center", va="center", fontweight="bold",
                    zorder=7, path_effects=stroke(2.2))


def build_figure_one(plt, img, mask, points, frame, cej, args):
    """圖一：牙齒座標系與 CEJ。除了 A 之外全部只用形狀。

    四格排成**一列**。牙齒是細長形，排成多列會留下大片空白；一列到底則動線
    單純（三支水平箭頭），每格也拿得到最大的高度。C 的剖面圖與兩條剖半遮罩
    在同一格內上下堆疊，總高度與其他格子齊平。
    """
    fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * 0.40))
    # C 那組是「D 的剖面圖、D 的遮罩、C 的剖面圖、C 的遮罩」四列。
    # 底部要留給共用的橫軸刻度與標題，留太少會被畫布切掉。
    gs = fig.add_gridspec(4, 14, height_ratios=[1.25, 0.75, 1.25, 0.75],
                          hspace=0.14, wspace=0.55,
                          left=0.012, right=0.988, top=0.975, bottom=0.10)

    ax_a = fig.add_subplot(gs[0:4, 0:2])
    ax_b = fig.add_subplot(gs[0:4, 2:4])
    ch_d = fig.add_subplot(gs[0, 5:11])
    strip_d = fig.add_subplot(gs[1, 5:11])
    ch_c = fig.add_subplot(gs[2, 5:11])
    strip_c = fig.add_subplot(gs[3, 5:11])
    ax_d = fig.add_subplot(gs[0:4, 12:14])

    show_image(ax_a, img, mask)
    panel_frame(ax_b, mask, points, frame)
    panel_cej(((ch_d, strip_d), (ch_c, strip_c)), mask, points, frame,
              args.drop_apical)
    panel_cd(ax_d, mask, points, frame, cej)

    images = [ax_a, ax_b, ax_d]
    for ax in images:
        bare(ax)
    apply_crop(fig, images, points, mask.shape)
    for ax, name in ((ax_a, "A"), (ax_b, "B"), (ax_d, "D")):
        letter(ax, name)

    boxes = {name: (b.x0, b.y0, b.x1, b.y1, (b.y0 + b.y1) / 2) for name, b in
             (("A", ax_a.get_position()), ("B", ax_b.get_position()),
              ("D", ax_d.get_position()))}
    boxes["C"] = group_box(fig, [ch_d, strip_d, ch_c, strip_c], "C")
    connect(fig, boxes, [("A", "B"), ("B", "C"), ("C", "D")])
    return fig


# --------------------------------------------------------------------------
# 圖二：特徵點與測量
# --------------------------------------------------------------------------

def ridge_pieces(img, mask, points, frame, cej, cut_at, threshold, surface_fraction):
    """邊緣嵴那幾格共用的中間產物，只算一次。"""
    region, _, _ = upper_crown_region(points, frame, cej, cut_at)
    residual, _, _ = residual_map(img, mask, region)
    cd = float(np.hypot(cej["D"][0] - cej["C"][0], cej["D"][1] - cej["C"][1]))
    ridge, bands, edges = ridge_points(region, residual, frame, threshold, mask,
                                       surface_fraction * cd)
    return region, residual, ridge, bands, edges


def panel_residual(ax, chart, img, mask, region, residual, bands, threshold):
    """A　殘差：原片疊綠遮罩與正殘差，旁邊是牙冠的殘差分布圖。

    殘差圖沿用 `find_ridge.py` 的畫法：紅藍發散色階，紅為比該厚度的預期更亮
    （即琺瑯質），藍為更暗。
    """
    show_image(ax, img, mask)
    overlay(ax, mask.shape, region[residual >= threshold], REGION_RGBA)

    x0 = max(0, int(region[:, 0].min()) - 8)
    y0 = max(0, int(region[:, 1].min()) - 8)
    x1, y1 = int(region[:, 0].max()) + 8, int(region[:, 1].max()) + 8
    canvas = np.full(img.shape, np.nan)
    canvas[region[:, 1].astype(int), region[:, 0].astype(int)] = residual
    lim = float(np.percentile(np.abs(residual), 98))
    image = chart.imshow(canvas[y0:y1, x0:x1], cmap="RdBu_r", vmin=-lim, vmax=lim)
    overlay(chart, img.shape, bands, (0.0, 0.0, 0.0, 0.18), crop=(y0, y1, x0, x1))
    bare(chart)
    return image  # colorbar 需要它；傳 None 進 colorbar 會得到一條無關的預設色階


def panel_edges(ax_mask, ax_img, img, mask, ridge, edges, bands):
    """B　鄰接面外緣與它的上端 A、B。左為遮罩，右為疊在原片上。

    淺紅畫的是**殘差為正的區域**（比該厚度的預期亮，即琺瑯質），不是整顆牙。
    這兩格只取那條帶的範圍放大——整顆牙的位置脈絡在 A 已經給過了，這裡要看的
    是外緣的細節與 A、B 落在哪裡。
    """
    from matplotlib.colors import to_rgb

    canvas = np.zeros((*mask.shape, 3))
    band = bands.astype(int)
    canvas[band[:, 1], band[:, 0]] = to_rgb(MASK_RED)
    ax_mask.imshow(canvas)

    ax_img.imshow(img, cmap="gray")
    overlay(ax_img, mask.shape, bands, (0.88, 0.33, 0.30, 0.45))

    for ax in (ax_mask, ax_img):
        ax.plot(edges[:, 0], edges[:, 1], ".", color="#00e5ff", markersize=1.3 * S)
        for name in ("A", "B"):
            label_point(ax, ridge[name], name, LANDMARK, offset=(6, -15))


def panel_crest(ax, chart, img, mask, axis, crest):
    """C　齒槽脊：牙根外側的亮度階梯。"""
    show_image(ax, img, mask)
    raw = axis.frame_raw
    coords = np.column_stack(np.nonzero(mask)[::-1]).astype(float)
    x_prime, y_prime = raw.to_frame(coords)

    for name, side, colour in (("E", -1, SIDE_C), ("F", +1, SIDE_D)):
        fit = crest[name]
        overlay(ax, mask.shape, fit.band, REGION_RGBA)

        chart.plot(fit.y, fit.brightness, "-", color=colour, linewidth=0.8 * S,
                   label=f"{name} 側亮度")
        chart.plot([fit.y.min(), fit.level, fit.level, fit.y.max()],
                   [fit.bright, fit.bright, fit.dark, fit.dark],
                   ":", color=colour, linewidth=1.2 * S, label=f"{name} 側階梯擬合")

        near = np.abs(y_prime - fit.level) < 3
        surface = float(np.abs(x_prime[near]).max()) if near.any() else 0.0
        point = raw.to_image(np.array([side * surface]), np.array([fit.level]))[0]
        label_point(ax, point, name, LANDMARK, offset=(6, -15))

    style_chart(chart, "沿長軸的高度 y′ (px)", "外側平均亮度", "lower left")


def panel_trim(ax, mask, points, axis):
    """D　裁切：捨去邊緣嵴以上的牙冠與牙根的根側 1/3。

    被捨去的部分仍以暗灰畫出，讀者才看得出「切掉了什麼」。
    """
    raw = axis.frame_raw
    _, y_prime = raw.to_frame(points)
    low, high = axis.trim
    keep = (y_prime >= low) & (y_prime <= high)

    canvas = np.zeros((*mask.shape, 3))
    dropped = points[~keep].astype(int)
    kept = points[keep].astype(int)
    canvas[dropped[:, 1], dropped[:, 0]] = 0.28
    canvas[kept[:, 1], kept[:, 0]] = 1.0
    ax.imshow(canvas)

    half = float(np.abs(raw.to_frame(points)[0]).max()) * 1.3
    for level in (low, high):
        mid = raw.point_at(level)
        a, b = mid + raw.x_axis * half, mid - raw.x_axis * half
        ax.plot([a[0], b[0]], [a[1], b[1]], "--", color=AUX, linewidth=1.2 * S)


def panel_new_axis(ax, mask, points, axis):
    """E　論文長軸：由裁切後的區域重新求得。"""
    show_mask(ax, mask)
    axis_arrow(ax, axis.frame, points, AUX, "y′ (crown)", mask.shape)


def build_figure_two(plt, img, mask, points, frame, cej, result, crest, args):
    """圖二：特徵點與測量。"""
    region, residual, ridge, bands, edges = ridge_pieces(
        img, mask, points, frame, cej, args.cut_at, args.threshold,
        args.surface_fraction)

    #   第一列  A（牙齒＋殘差圖）　　B（遮罩＋原片）
    #   第二列  C（牙齒＋亮度圖）　　D　E　F
    # 兩列都排到滿；D、E、F 之間各空一欄，箭頭才有長度可畫。底部要留給 C 那張
    # 亮度圖的橫軸標題。
    fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * 0.72))
    # 第 3 列是**空白的走道**，專門留給折行箭頭轉彎。兩列直接相鄰時，箭頭最後
    # 「往下插進下一格」那段會短到比箭頭本身還小，頭部就會畫在前一段（水平段）
    # 上、看起來朝左。不能用加大 hspace 解決——那會連帶把 B 的兩條橫幅也拉開。
    gs = fig.add_gridspec(5, 18, height_ratios=[1, 1, 0.38, 1, 1],
                          hspace=0.20, wspace=0.55,
                          left=0.012, right=0.988, top=0.975, bottom=0.058)

    ax_a, ch_a = fig.add_subplot(gs[0:2, 0:2]), fig.add_subplot(gs[0:2, 3:9])
    # B 的兩格是**寬扁的橫幅**：琺瑯質那條帶本來就寬扁，塞進細高的格子只會
    # 在補長寬比時補出一大片黑。上下堆疊也讓遮罩與原片能直接對照。
    ax_b1, ax_b2 = fig.add_subplot(gs[0, 11:18]), fig.add_subplot(gs[1, 11:18])
    ax_c, ch_c = fig.add_subplot(gs[3:5, 0:2]), fig.add_subplot(gs[3:5, 3:9])
    ax_d = fig.add_subplot(gs[3:5, 10:12])
    ax_e = fig.add_subplot(gs[3:5, 13:15])
    ax_f = fig.add_subplot(gs[3:5, 16:18])

    residual_image = panel_residual(ax_a, ch_a, img, mask, region, residual,
                                    bands, args.threshold)
    bar = fig.colorbar(residual_image, ax=ch_a, fraction=0.035, pad=0.015)
    bar.ax.tick_params(labelsize=TICK_SIZE, length=2 * S, pad=1)
    bar.outline.set_linewidth(0.5 * S)
    bar.set_label("殘差（灰階）", fontsize=AXIS_SIZE, labelpad=3)
    # 色標是獨立的 axes，得一起框進 A 那一組，否則會被框線切到
    panel_edges(ax_b1, ax_b2, img, mask, ridge, edges, bands)
    panel_crest(ax_c, ch_c, img, mask, result.axis, crest)
    panel_trim(ax_d, mask, points, result.axis)
    panel_new_axis(ax_e, mask, points, result.axis)
    draw_result(ax_f, img, mask, result, scale=1.8)
    ax_f.set_title("")

    for ax in (ax_a, ax_b1, ax_b2, ax_c, ax_d, ax_e, ax_f):
        bare(ax)
    apply_crop(fig, [ax_a, ax_c, ax_d, ax_e, ax_f], points, mask.shape)
    # B 的兩格只框琺瑯質那條帶，放大看外緣與 A、B
    apply_crop(fig, [ax_b1, ax_b2], bands, mask.shape, margin=0.22)
    for ax, name in ((ax_d, "D"), (ax_e, "E"), (ax_f, "F")):
        letter(ax, name)

    boxes = {name: (b.x0, b.y0, b.x1, b.y1, (b.y0 + b.y1) / 2) for name, b in
             (("D", ax_d.get_position()), ("E", ax_e.get_position()),
              ("F", ax_f.get_position()))}
    boxes["A"] = group_box(fig, [ax_a, ch_a, bar.ax], "A")
    boxes["B"] = group_box(fig, [ax_b1, ax_b2], "B")
    boxes["C"] = group_box(fig, [ax_c, ch_c], "C")
    connect(fig, boxes, [("A", "B"), ("B", "C"), ("C", "D"),
                         ("D", "E"), ("E", "F")])
    return fig


# --------------------------------------------------------------------------
# 圖說
# --------------------------------------------------------------------------

def caption_one(result) -> str:
    raw = result.axis.frame_raw
    return (
        "圖 1　牙齒座標系的建立與釉牙骨質界的定位。除 (A) 之外全部只用遮罩的形狀，"
        "不使用灰階影像。"
        "(A) 輸入：根尖片與牙齒遮罩。"
        "(B) 以主成分分析求長軸；箭頭指向牙冠端，方向由牙冠半與牙根半的 "
        f"Var(x′)/Var(y′) 判定（本例 {raw.crown_ratio:.2f} 對 {raw.root_ratio:.2f}，"
        f"相差 {raw.crown_ratio / raw.root_ratio:.1f} 倍）。"
        "(C) 釉牙骨質界：上為兩側的半寬剖面，下為沿長軸剖半、外緣朝同側橫躺的遮罩，"
        "三者共用同一橫軸。牙根段的基準線（虛線）往牙冠方向外插、牙冠隆起段的線"
        "（點線）往牙根方向回插，兩線交點即 C、D。C 側以藍、D 側以紫區分。"
        "(D) C、D 兩點的連線把牙齒分為牙冠與牙根。"
    )


def caption_two(result) -> str:
    return (
        "圖 2　特徵點的定位與測量。"
        "(A) 邊緣嵴：扣除射線穿透厚度造成的灰階變化後取殘差，殘差為正者（紅）即琺瑯質；"
        "右圖為牙冠的殘差分布（紅為較該厚度的預期亮、藍為較暗），灰為通過閾值的帶狀區域。"
        "(B) 取帶狀區域中貼著牙齒表面的鄰接面外緣（深藍點），其上端即 A、B；"
        "左為遮罩，右為疊在原片上。"
        "(C) 齒槽脊：沿牙根外側的窄帶取平均亮度，以階梯模型擬合亮度陡降處得 E、F；"
        "右圖為兩側的亮度剖面與擬合。"
        "(D) 依論文定義裁切：捨去 J 以上的牙冠與牙根的根側 1/3（暗灰為捨去的部分）。"
        "(E) 由裁切後的區域重新求主軸，得論文長軸。"
        "(F) 全部特徵點與其沿長軸的投影點，"
        f"CRR = {result.crr:.3f}、ABLR = {result.ablr:.3f}、B-CRR = {result.b_crr:.3f}。"
        "藍點為原論文定義的特徵點 A–G，灰點為由其推導出的 H、I、J、K、L、Q、R、S。"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", default="labeled_PA/13.json", help="要畫的那顆牙")
    parser.add_argument("--prefix", default="fig_method", help="輸出檔名前綴")
    parser.add_argument("--drop-apical", type=float, default=0.25)
    parser.add_argument("--cut-at", type=float, default=1.0 / 3.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--surface-fraction", type=float, default=0.10)
    parser.add_argument("--inner-fraction", type=float, default=0.03)
    parser.add_argument("--outer-fraction", type=float, default=0.15)
    parser.add_argument("--root-fraction", type=float, default=2.0 / 3.0)
    args = parser.parse_args()

    _, img, mask, points = load_tooth(args.source)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_cjk_font(plt)

    frame = fit_axis(points)
    cej = cej_points(points, frame, args.drop_apical)
    result = measure(Path(args.source).stem, img, mask, args.drop_apical, args.cut_at,
                     args.threshold, args.surface_fraction, args.inner_fraction,
                     args.outer_fraction, args.root_fraction)
    landmarks = result.axis.landmarks
    cd_width = float(np.hypot(landmarks["D"][0] - landmarks["C"][0],
                              landmarks["D"][1] - landmarks["C"][1]))
    crest = find_crest(img, mask, result.axis, args.inner_fraction * cd_width,
                       args.outer_fraction * cd_width)

    one = build_figure_one(plt, img, mask, points, frame, cej, args)
    one.savefig(f"{args.prefix}1.png", dpi=120)
    two = build_figure_two(plt, img, mask, points, frame, cej, result, crest, args)
    two.savefig(f"{args.prefix}2.png", dpi=120)

    print(f"已存檔 -> {args.prefix}1.png、{args.prefix}2.png")
    print()
    print(caption_one(result))
    print()
    print(caption_two(result))


if __name__ == "__main__":
    main()
