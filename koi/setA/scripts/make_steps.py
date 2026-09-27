"""把整條流程的**每一步中間產物**畫成圖，供網站的流程卡片使用。

網站上每個步驟要有一張「這一步做完長什麼樣」的實測圖。前兩步是我這邊的
模型，後面幾步是組員的幾何推導——但組員的 `measure()` 是一次算完的，中間
沒有可以看的成品。所以這支把 `refine_axis()` 的內容按步驟拆開重跑一次，
每跑完一段就畫一張。**呼叫的是組員的同一批函式**，因此結果與 `measure()`
完全一致，只是多了中間輸出。

步驟切法的原則：一個步驟 = 一個方法。兩個點若由同一支函式、同一套原理求出
（C 與 D、A 與 B、E 與 F），就是同一個步驟。

    01  OBB          Mask R-CNN 的遮罩 → minAreaRect → 原圖上的旋轉框
    02  HRNet-w32    轉正 crop 進 U-Net(HRNet-w32) → 牙齒遮罩貼回原圖
    03  初始長軸      整顆遮罩 PCA → 座標系 + 牙冠朝向      find_axis_raw
    04  C D          半寬剖面分段線性 → CEJ                 find_cej
    05  A B          厚度-亮度殘差 → 邊緣嵴                  find_ridge
                     另出 05b：殘差場本身，用深色底的發散色圖重畫
    06  G + 精修長軸  根尖中位數 → 依文獻裁切重做 PCA → H I J K L
    07  E F          牙根外側亮度階梯 → 齒槽脊頂 → R Q S    find_alveolar_crest
    08  量測          CRR / ABLR / B-CRR

輸出是深色背景的 PNG，配色與網站一致，不需要再做後製。

用法：
    py koi/setA/scripts/make_steps.py --image 113
    py koi/setA/scripts/make_steps.py --all --out koi/setA/steps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                      # .../CRR_PA
sys.path.insert(0, str(HERE))               # 我的 koi 腳本
sys.path.insert(0, str(REPO))               # 組員的 scripts 套件

import matplotlib                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib import font_manager         # noqa: E402

from eval_seg2_holdout import predict       # noqa: E402
from make_crops_obb import obb_of, warp_of  # noqa: E402
from postprocess import clean_mask          # noqa: E402
from train_maskrcnn import CKPT, ROOT, build_model  # noqa: E402
from train_seg2 import build_seg2, split_tag        # noqa: E402

from scripts.find_alveolar_crest import find_crest          # noqa: E402
from scripts.find_axis_paper import (                       # noqa: E402
    PaperAxis, apex_point, axial, project_landmarks,
)
from scripts.find_axis_raw import (                         # noqa: E402
    ToothFrame, cross_axis, fit_axis, pca,
)
from scripts.find_cej import cej_points, upper_crown_region  # noqa: E402
from scripts.find_ridge import residual_map, ridge_points    # noqa: E402
from scripts.measure import Measurement, mask_points_of      # noqa: E402

# 與 scratchpad/e2e_crr.py、組員 measure.py 的預設值相同
DEFAULTS = dict(drop_apical=0.25, cut_at=1 / 3, threshold=0.0,
                surface_fraction=0.10, inner_fraction=0.03,
                outer_fraction=0.15, root_fraction=2 / 3)

SEG_TAG = "unet_tu-hrnet_w32"
DET_THR, PAD = 0.35, 0.2

# 網站配色
BG, PANEL = "#07080A", "#0D1013"
INK, INK2, INK3 = "#EAEFF4", "#96A2AE", "#5B6672"
ACC, CYAN, AMBER, ROSE = "#5FD3A6", "#4FC3E8", "#E5B45F", "#E57373"
MASK_RGBA = (0.37, 0.83, 0.65, 0.30)


def use_cjk_font() -> None:
    """挑一個系統裡有的中文字型，否則圖上的中文會變成方框。"""
    for name in ("PingFang TC", "Heiti TC", "Noto Sans CJK TC",
                 "Arial Unicode MS", "Songti SC"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------
# 第一階段：Mask R-CNN → OBB → HRNet-w32
# --------------------------------------------------------------------------


def load_detector():
    ck = torch.load(CKPT / "original" / "maskrcnn_final.pt",
                    map_location="cpu", weights_only=False)
    m = build_model(False, ck.get("mask_res", 28))
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def load_segmenters():
    """五折全載入做集成。測試集不在任何一折的訓練資料裡，集成是合法的。"""
    arch, enc = split_tag(SEG_TAG)
    models = []
    for f in range(5):
        m = build_seg2(arch, enc, pretrained=False)
        m.load_state_dict(torch.load(
            ROOT / "checkpoints_obb" / "seg2" / SEG_TAG / f"fold{f}.pt",
            map_location="cpu", weights_only=False)["model"])
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def detect_obb(det, gray: np.ndarray) -> list[tuple]:
    """回傳依信心度排序的 [(cx, cy, 短邊, 長邊, 角度), ...]。"""
    t = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1)
    out = det([t])[0]
    keep = out["scores"].numpy() >= DET_THR
    masks = out["masks"].numpy()[keep, 0] > 0.5
    boxes = []
    for m in masks:
        m8 = clean_mask(m).astype(np.uint8)
        if m8.any():
            boxes.append(obb_of(m8))
    return boxes


@torch.no_grad()
def segment_in_obb(segs, gray: np.ndarray, box: tuple) -> np.ndarray:
    """把 OBB 轉正成 crop，送進 HRNet-w32，再把遮罩貼回原圖座標。"""
    h, w = gray.shape
    M, cw, ch = warp_of(*box, PAD)
    crop = cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR)
    prob = predict(segs, crop, use_tta=False)
    prob = cv2.resize(prob, (cw, ch), interpolation=cv2.INTER_LINEAR)
    back = cv2.warpAffine(prob, cv2.invertAffineTransform(M), (w, h),
                          flags=cv2.INTER_LINEAR) > 0.5
    return clean_mask(back).astype(np.uint8)


# --------------------------------------------------------------------------
# 作圖的共用零件
# --------------------------------------------------------------------------


def canvas(img: np.ndarray, zoom: float = 1.0, box=None):
    """開一張只有影像的深色畫布；box=(x0, y0, x1, y1) 時只看那塊。"""
    if box is not None:
        x0, y0, x1, y1 = box
    else:
        x0, y0, x1, y1 = 0, 0, img.shape[1], img.shape[0]
    fig = plt.figure(figsize=((x1 - x0) / 100 * zoom, (y1 - y0) / 100 * zoom),
                     dpi=150, facecolor=PANEL)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(PANEL)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    return fig, ax


def tint(ax, mask: np.ndarray, rgba=MASK_RGBA) -> None:
    layer = np.zeros((*mask.shape, 4), np.float32)
    layer[mask.astype(bool)] = rgba
    ax.imshow(layer, interpolation="nearest")


def bbox_of(mask: np.ndarray, pad_frac: float = 0.10):
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    px = int((xs.max() - xs.min()) * pad_frac) + 12
    py = int((ys.max() - ys.min()) * pad_frac) + 12
    return (max(0, xs.min() - px), max(0, ys.min() - py),
            min(w, xs.max() + px), min(h, ys.max() + py))


def dot(ax, p, label, color, r=5.0, fs=12, off=(7, -9)) -> None:
    ax.plot(p[0], p[1], "o", color=color, markersize=r,
            markeredgecolor=BG, markeredgewidth=1.0, zorder=6)
    if label:
        ax.annotate(label, p, textcoords="offset points", xytext=off,
                    color=color, fontsize=fs, fontweight="bold", zorder=7,
                    path_effects=_halo())


def _halo():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=2.6, foreground=BG)]


def axis_line(ax, frame: ToothFrame, points: np.ndarray, color=ACC,
              lw=1.6, ls="-", label=None) -> None:
    _, y = frame.to_frame(points)
    a, b = frame.point_at(float(y.min())), frame.point_at(float(y.max()))
    ax.plot([a[0], b[0]], [a[1], b[1]], ls, color=color, linewidth=lw,
            zorder=5, label=label)


def level_line(ax, frame: ToothFrame, cd_slope: float, level: float,
               half: float, color, ls="--", lw=1.1) -> None:
    """過長軸上某個高度、平行於 CD 的那條線。"""
    xs = np.array([-half, half])
    ys = level + xs * cd_slope
    p = frame.to_image(xs, ys)
    ax.plot(p[:, 0], p[:, 1], ls, color=color, linewidth=lw, zorder=4)


def caption(ax, text: str, color=INK2, size=10) -> None:
    """左上角的說明字。用 axes 座標，長字串才不會超出畫布被裁掉。"""
    ax.text(0.02, 0.985, text, transform=ax.transAxes, ha="left", va="top",
            color=color, fontsize=size, fontweight="bold", zorder=8,
            path_effects=_halo())


# 深色底用的發散色圖：負殘差走青、正殘差走磚紅，中間留一個中灰。
# matplotlib 內建的 coolwarm 中心是純白，貼在深色頁面上會把 0 畫成最亮的地方；
# 但中心若直接用背景色，近零的雜訊又會變成一片黑洞，所以取中灰。
def _diverging():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "resid_dark", ["#1F6E9E", "#4FC3E8", "#A9B4BE", "#E5785F", "#B93A2B"])


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=PANEL, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  {path.name}")


# --------------------------------------------------------------------------
# 逐步重跑 refine_axis，每段畫一張
# --------------------------------------------------------------------------


def run_steps(name: str, img: np.ndarray, mask: np.ndarray, out: Path,
              boxes: list, all_masks: list) -> Measurement:
    p = DEFAULTS
    points = mask_points_of(mask)
    box = bbox_of(mask)
    half = float(np.abs(fit_axis(points).to_frame(points)[0]).max()) * 1.35

    # ---- 01 OBB ----------------------------------------------------------
    fig, ax = canvas(img)
    for i, b in enumerate(boxes):
        quad = cv2.boxPoints(((b[0], b[1]), (b[2], b[3]), b[4]))
        quad = np.vstack([quad, quad[:1]])
        c = ACC if i == 0 else CYAN
        ax.plot(quad[:, 0], quad[:, 1], "-", color=c, linewidth=2.0, zorder=5)
        ax.plot(b[0], b[1], "+", color=c, markersize=9, zorder=6)
        ax.annotate(f"tooth {i + 1}", (quad[:, 0].min(), quad[:, 1].min()),
                    textcoords="offset points", xytext=(2, -8), color=c,
                    fontsize=11, fontweight="bold", path_effects=_halo())
    save(fig, out / "01_obb.png")

    # ---- 01b 旋轉框 vs 水平框（放在卡片背面講原理用）-------------------
    fig, ax = canvas(img)
    for i, (b, m) in enumerate(zip(boxes, all_masks)):
        quad = cv2.boxPoints(((b[0], b[1]), (b[2], b[3]), b[4]))
        quad = np.vstack([quad, quad[:1]])
        x, y, w_, h_ = cv2.boundingRect(m)
        ax.plot([x, x + w_, x + w_, x, x], [y, y, y + h_, y + h_, y], "-",
                color=AMBER, linewidth=1.6, alpha=.9, zorder=4)
        ax.plot(quad[:, 0], quad[:, 1], "-", color=ACC, linewidth=2.0, zorder=5)
        if i == 0:
            ratio = (w_ * h_) / max(b[2] * b[3], 1e-9)
            caption(ax, f"水平框面積是旋轉框的 {ratio:.2f} 倍", color=AMBER)
    ax.plot([], [], "-", color=AMBER, label="水平框 HBB")
    ax.plot([], [], "-", color=ACC, label="旋轉框 OBB")
    leg = ax.legend(loc="lower right", fontsize=10, facecolor=PANEL,
                    edgecolor="#2B353F", labelcolor=INK)
    leg.get_frame().set_alpha(.92)
    save(fig, out / "01b_obb_vs_hbb.png")

    # ---- 02 HRNet-w32 遮罩 ----------------------------------------------
    fig, ax = canvas(img)
    for i, m in enumerate(all_masks):
        tint(ax, m, MASK_RGBA if i == 0 else (0.31, 0.76, 0.91, 0.28))
        cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        for c in cnts:
            ax.plot(c[:, 0, 0], c[:, 0, 1], "-",
                    color=ACC if i == 0 else CYAN, linewidth=1.4, zorder=5)
    save(fig, out / "02_mask.png")

    # ---- 03 初始長軸 -----------------------------------------------------
    frame_raw = fit_axis(points)
    x_prime, y_prime = frame_raw.to_frame(points)
    fig, ax = canvas(img, box=box)
    for sel, rgba in ((y_prime >= 0, (0.31, 0.76, 0.91, 0.30)),
                      (y_prime < 0, (0.90, 0.70, 0.37, 0.26))):
        layer = np.zeros((*mask.shape, 4), np.float32)
        layer[points[sel, 1].astype(int), points[sel, 0].astype(int)] = rgba
        ax.imshow(layer, interpolation="nearest")
    axis_line(ax, frame_raw, points)
    a = frame_raw.to_image(np.array([-half]), np.array([0.0]))[0]
    b = frame_raw.to_image(np.array([half]), np.array([0.0]))[0]
    ax.plot([a[0], b[0]], [a[1], b[1]], "--", color=INK2, linewidth=1.2, zorder=4)
    ax.plot(*frame_raw.origin, "o", color=ACC, markersize=6,
            markeredgecolor=BG, zorder=6)
    caption(ax, f"ρ 牙冠 {frame_raw.crown_ratio:.3f} ／ 牙根 {frame_raw.root_ratio:.3f}")
    save(fig, out / "03_axis.png")

    # ---- 04 C、D（CEJ）---------------------------------------------------
    cej = cej_points(points, frame_raw, p["drop_apical"])
    (xc, yc), (xd, yd) = cej["C"], cej["D"]
    cd_slope = (yd - yc) / (xd - xc) if abs(xd - xc) > 1e-9 else 0.0
    C = frame_raw.to_image(np.array([xc]), np.array([yc]))[0]
    D = frame_raw.to_image(np.array([xd]), np.array([yd]))[0]

    fig, ax = canvas(img, box=box)
    tint(ax, mask)
    axis_line(ax, frame_raw, points, color=INK3, lw=1.2)
    ax.plot([C[0], D[0]], [C[1], D[1]], "-", color=CYAN, linewidth=1.6, zorder=5)
    dot(ax, C, "C", CYAN)
    dot(ax, D, "D", CYAN)
    save(fig, out / "04_cej.png")

    # ---- 05 A、B（邊緣嵴）-----------------------------------------------
    region, _, _ = upper_crown_region(points, frame_raw, cej, p["cut_at"])
    residual, _, _ = residual_map(img, mask, region)
    cej_width = float(np.hypot(xd - xc, yd - yc))
    ridge, _, _ = ridge_points(region, residual, frame_raw, p["threshold"], mask,
                              p["surface_fraction"] * cej_width)

    fig, ax = canvas(img, box=box)
    pos = region[residual > 0] if len(region) == len(residual) else region
    layer = np.zeros((*mask.shape, 4), np.float32)
    layer[pos[:, 1].astype(int), pos[:, 0].astype(int)] = (0.90, 0.45, 0.45, 0.42)
    ax.imshow(layer, interpolation="nearest")
    ax.plot([C[0], D[0]], [C[1], D[1]], "-", color=INK3, linewidth=1.1, zorder=4)
    dot(ax, C, "C", INK3, r=3.5, fs=10)
    dot(ax, D, "D", INK3, r=3.5, fs=10)
    dot(ax, ridge["A"], "A", ROSE)
    dot(ax, ridge["B"], "B", ROSE)
    save(fig, out / "05_ridge.png")

    # ---- 05b 殘差圖（卡片背面講原理用）--------------------------------
    field = np.full(mask.shape, np.nan, np.float32)
    field[region[:, 1].astype(int), region[:, 0].astype(int)] = residual
    ys, xs = np.nonzero(~np.isnan(field))
    pad = 10
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
    vmax = float(np.nanpercentile(np.abs(residual), 99)) or 1.0

    fig = plt.figure(figsize=((x1 - x0) / 100 * 3.2, (y1 - y0) / 100 * 3.2),
                     dpi=150, facecolor=PANEL)
    ax = fig.add_axes([0, 0, 0.88, 1]); ax.set_facecolor(PANEL)
    im2 = ax.imshow(field, cmap=_diverging(), vmin=-vmax, vmax=vmax,
                    interpolation="bilinear")
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    dot(ax, ridge["A"], "A", INK, r=7, fs=20, off=(11, -14))
    dot(ax, ridge["B"], "B", INK, r=7, fs=20, off=(11, -14))
    cax = fig.add_axes([0.905, 0.14, 0.018, 0.72])
    cb = fig.colorbar(im2, cax=cax)
    cb.outline.set_edgecolor("#2B353F"); cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(colors=INK2, labelsize=17, width=1.0, length=5, pad=5)
    cb.set_label("殘差（灰階）", color=INK2, fontsize=18, labelpad=12)
    save(fig, out / "05b_residual.png")

    # ---- 06 G + 依文獻精修長軸 ------------------------------------------
    landmarks = {"A": ridge["A"], "B": ridge["B"], "C": C, "D": D}
    landmarks["G"] = apex_point(points, frame_raw, cd_slope)
    levels_raw = project_landmarks(landmarks, frame_raw, cd_slope)

    root_length = levels_raw["L"] - levels_raw["K"]
    low = levels_raw["K"] + (1 - p["root_fraction"]) * root_length
    keep = (y_prime >= low) & (y_prime <= levels_raw["J"])
    origin, components, eigenvalues = pca(points[keep])
    y_axis = components[0]
    if float(np.dot(y_axis, frame_raw.y_axis)) < 0:
        y_axis = -y_axis
    frame = ToothFrame(origin=origin, x_axis=cross_axis(y_axis), y_axis=y_axis,
                       eigenvalues=eigenvalues, n_points=int(keep.sum()),
                       crown_ratio=frame_raw.crown_ratio,
                       root_ratio=frame_raw.root_ratio)
    xc2, yc2 = frame.to_frame(landmarks["C"][None, :])
    xd2, yd2 = frame.to_frame(landmarks["D"][None, :])
    cd_final = ((float(yd2[0]) - float(yc2[0])) / (float(xd2[0]) - float(xc2[0]))
                if abs(float(xd2[0]) - float(xc2[0])) > 1e-9 else 0.0)
    levels = project_landmarks(landmarks, frame, cd_final)
    axis = PaperAxis(frame_raw, frame, landmarks, levels, levels_raw,
                     (low, levels_raw["J"]), cd_final)

    fig, ax = canvas(img, box=box)
    layer = np.zeros((*mask.shape, 4), np.float32)
    layer[points[~keep, 1].astype(int), points[~keep, 0].astype(int)] = (0.36, 0.41, 0.45, 0.22)
    layer[points[keep, 1].astype(int), points[keep, 0].astype(int)] = MASK_RGBA
    ax.imshow(layer, interpolation="nearest")
    axis_line(ax, frame_raw, points, color=INK3, lw=1.1, ls="--")
    axis_line(ax, frame, points[keep], color=ACC, lw=1.8)
    for key, col in (("J", AMBER), ("L", AMBER), ("K", AMBER)):
        level_line(ax, frame, cd_final, levels[key], half, col)
        dot(ax, frame.point_at(levels[key]), key, col, r=3.5, fs=10)
    dot(ax, landmarks["G"], "G", CYAN)
    caption(ax, f"與初始主軸夾角 {axis.angle:.2f}°")
    save(fig, out / "06_apex.png")

    # ---- 07 E、F（齒槽脊頂）--------------------------------------------
    crest = find_crest(img, mask, axis,
                       p["inner_fraction"] * cej_width,
                       p["outer_fraction"] * cej_width)
    lm = dict(landmarks)
    lv = dict(levels)
    for key, side, level_key in (("E", -1, "R"), ("F", +1, "Q")):
        level_raw = crest[key].level
        xp, yp = axis.frame_raw.to_frame(points)
        near = np.abs(yp - level_raw) < 3
        surface = float(np.abs(xp[near]).max()) if near.any() else 0.0
        pt = axis.frame_raw.to_image(np.array([side * surface]),
                                     np.array([level_raw]))[0]
        lm[key] = pt
        lv[level_key] = float(axial(pt[None, :], axis.frame, axis.cd_slope)[0])
    lv["S"] = (lv["R"] + lv["Q"]) / 2

    fig, ax = canvas(img, box=box)
    tint(ax, mask)
    axis_line(ax, frame, points, color=INK3, lw=1.1)
    # 點留在長軸上的實際高度，只把三個標籤上下錯開：Q 在最上、R 在最下，
    # 各自往自己那一側再推一點，字就不會疊在一起（這張只差 11 px）。
    for key, col, off in (("Q", CYAN, (9, 10)), ("S", ACC, (11, -3)),
                          ("R", CYAN, (9, -17))):
        level_line(ax, frame, cd_final, lv[key], half, col,
                   ls="-" if key == "S" else "--", lw=1.6 if key == "S" else 1.0)
        dot(ax, frame.point_at(lv[key]), key, col, r=3.5, fs=11, off=off)
    dot(ax, lm["E"], "E", AMBER)
    dot(ax, lm["F"], "F", AMBER)
    save(fig, out / "07_alc.png")

    # ---- 08 量測 ---------------------------------------------------------
    result = Measurement(name, axis, lm, lv, crest)
    fig, ax = canvas(img, box=box)
    tint(ax, mask)
    axis_line(ax, frame, points, color=INK, lw=1.2)
    for key, col in (("J", AMBER), ("S", ACC), ("K", CYAN)):
        level_line(ax, frame, cd_final, lv[key], half, col, ls="-", lw=1.5)
    for key in ("H", "I", "L", "R", "Q"):
        level_line(ax, frame, cd_final, lv[key], half, INK3, ls=":", lw=0.9)
    for key in "ABCDEFG":
        dot(ax, lm[key], key, CYAN, r=4.0, fs=10)
    for key, col in (("J", AMBER), ("S", ACC), ("K", CYAN)):
        dot(ax, frame.point_at(lv[key]), key, col, r=4.0, fs=11, off=(-16, -6))
    caption(ax, f"CRR {result.crr:.4f}\nABLR {result.ablr:.4f}\n"
                f"B-CRR {result.b_crr:.4f}", color=ACC, size=11)
    save(fig, out / "08_measure.png")
    return result


# --------------------------------------------------------------------------


def process(path: Path, det, segs, out_root: Path, tooth: int) -> dict | None:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"讀不到 {path}")
    boxes = detect_obb(det, gray)
    if not boxes:
        print(f"{path.name}: 偵測不到牙齒，跳過")
        return None
    # 由左至右排序，讓「tooth 1」在圖上與編號一致
    order = np.argsort([b[0] for b in boxes])
    boxes = [boxes[i] for i in order]
    masks = [segment_in_obb(segs, gray, b) for b in boxes]
    masks = [m for m in masks if m.any()]
    if tooth >= len(masks):
        print(f"{path.name}: 只有 {len(masks)} 顆牙，--tooth {tooth} 超出範圍")
        return None

    out = out_root / path.stem
    print(f"{path.name}  {len(masks)} 顆牙，畫第 {tooth + 1} 顆")
    boxes = boxes[tooth:tooth + 1] + boxes[:tooth] + boxes[tooth + 1:]
    masks = masks[tooth:tooth + 1] + masks[:tooth] + masks[tooth + 1:]
    try:
        r = run_steps(path.stem, gray, masks[0], out, boxes, masks)
    except Exception as exc:                       # noqa: BLE001
        print(f"{path.name}: 幾何推導失敗 — {exc}")
        return None
    print(f"  CRR {r.crr:.4f}  ABLR {r.ablr:.4f}  B-CRR {r.b_crr:.4f}  "
          f"自我檢查差 {r.consistency:.2e}")
    row = {"tooth": path.stem, "teeth_found": len(masks),
           "CRR": round(r.crr, 4), "ABLR": round(r.ablr, 4),
           "max_BLR": round(r.max_blr, 4), "B_CRR": round(r.b_crr, 4),
           "consistency": f"{r.consistency:.1e}"}
    row.update({k: round(v, 1) for k, v in r.levels.items()})
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="測試集裡的檔名（可不帶副檔名）")
    ap.add_argument("--all", action="store_true", help="跑整個測試集")
    ap.add_argument("--tooth", type=int, default=0, help="畫第幾顆牙，由左至右從 0 起算")
    ap.add_argument("--src", default=str(ROOT / "testset"))
    ap.add_argument("--out", default=str(ROOT / "steps"))
    ap.add_argument("--csv", help="把所有成功案例的指標寫成一份 CSV")
    args = ap.parse_args()

    use_cjk_font()
    src, out = Path(args.src), Path(args.out)
    paths = sorted(src.glob("*.jpg")) if args.all else \
        [src / (args.image if args.image.endswith(".jpg") else args.image + ".jpg")]

    print("載入模型…")
    det, segs = load_detector(), load_segmenters()
    rows = [r for r in (process(p, det, segs, out, args.tooth) for p in paths)
            if r is not None]
    print(f"\n{len(rows)} / {len(paths)} 張完整跑完整條流程")
    if args.csv and rows:
        import csv as _csv
        cols = ["tooth", "teeth_found", "CRR", "ABLR", "max_BLR", "B_CRR",
                "consistency", "H", "I", "J", "K", "L", "R", "Q", "S"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            wr = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        print(f"指標寫入 {args.csv}")


if __name__ == "__main__":
    main()
