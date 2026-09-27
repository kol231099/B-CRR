"""重畫第一階段的架構圖：原生深色 SVG，不是把白底圖反相。

網站上那張本來是拿投影片的白底 SVG 反相來的，色票、線寬、虛線框都是原檔
留下來的，湊在一起很雜。這支照同一份結構重畫一次：統一的圓角與線寬、依所屬
區塊配色（偵測＝青、encoder＝綠、decoder＝琥珀、通用＝石板灰）、一種箭頭、
容器用同一種低對比虛線框。內容與 hrnet_w32_OBB.svg 完全一致，只換畫法。

三張內嵌縮圖預設沿用 hrnet_w32_OBB.svg 裡原本那三張（原圖、偵測出兩顆牙的
旋轉框、兩顆牙的遮罩），已經抽出來存在 figs/insets/。加 --from-model 則改用
指定測試片當場跑一次的結果。

輸出
----
    koi/setA/figs/arch.svg    向量原始檔（縮圖以 base64 WebP 內嵌，單檔可攜）

用法：
    py koi/setA/scripts/make_arch.py
    py koi/setA/scripts/make_arch.py --as-code
    py koi/setA/scripts/make_arch.py --from-model --image 191 --tooth 1
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_crops_obb import warp_of                                    # noqa: E402
from make_steps import (                                              # noqa: E402
    DET_THR, PAD, detect_obb, load_detector, load_segmenters,
)
from eval_seg2_holdout import predict                                 # noqa: E402
from train_maskrcnn import ROOT                                       # noqa: E402

# ── 色票（與網站同一組）────────────────────────────────────────────────
BG     = "#0A0C0F"
INK    = "#E8EDF2"
INK_2  = "#8C99A6"
INK_3  = "#55606B"
LINE   = "#222A33"
WIRE   = "#5C6874"
CYAN   = "#4FC3E8"
GREEN  = "#5FD3A6"
AMBER  = "#E5B45F"
SLATE  = "#8FA3B5"
FONT   = ("Archivo, system-ui, -apple-system, 'Segoe UI', Roboto, "
          "'Helvetica Neue', Arial, sans-serif")
MONO   = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

# 節點底色：在 #11161B 上疊一點該區塊的色相
FILL = {CYAN: "#101B22", GREEN: "#101E1A", AMBER: "#1D1810", SLATE: "#141A20"}


# ── SVG 零件 ──────────────────────────────────────────────────────────
def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h, self.body = w, h, []

    def add(self, s: str) -> None:
        self.body.append("  " + s)

    # 節點 --------------------------------------------------------------
    def node(self, x, y, w, h, label, color=SLATE, sub=None, r=11):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
                 f'fill="{FILL[color]}" stroke="{color}" stroke-opacity=".55" '
                 f'stroke-width="1.25"/>')
        lines = label.split("|")
        n = len(lines)
        cy = y + h / 2 - (n - 1) * 9 + (0 if sub is None else -6)
        for i, ln in enumerate(lines):
            self.add(f'<text x="{x + w / 2}" y="{cy + i * 18 + 5.5}" fill="{INK}" '
                     f'font-family="{FONT}" font-size="14.5" font-weight="500" '
                     f'text-anchor="middle">{esc(ln)}</text>')
        if sub:
            self.add(f'<text x="{x + w / 2}" y="{y + h - 11}" fill="{color}" '
                     f'fill-opacity=".85" font-family="{MONO}" font-size="9.5" '
                     f'letter-spacing="1.1" text-anchor="middle">{esc(sub)}</text>')

    # 虛線容器 -----------------------------------------------------------
    def group(self, x, y, w, h, label, color, size=9.5, strong=False):
        """strong=True 是三個主要階段的標題，字大一級並加一顆領頭的圓點。"""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" '
                 f'fill="{color}" fill-opacity="{.05 if strong else .028}" '
                 f'stroke="{color}" stroke-opacity="{.46 if strong else .30}" '
                 f'stroke-width="1" stroke-dasharray="5 5"/>')
        per = size * 0.62 + 1.7
        lead = 26 if strong else 11
        tw = len(label) * per + lead + 13
        ph = size * 2.0
        self.add(f'<rect x="{x + 20}" y="{y - ph / 2}" width="{tw}" height="{ph}" '
                 f'rx="{ph / 2}" fill="{BG}"/>')
        if strong:
            self.add(f'<circle cx="{x + 20 + 14}" cy="{y}" r="3.6" fill="{color}"/>')
        self.add(f'<text x="{x + 20 + lead}" y="{y + size * 0.37}" fill="{color}" '
                 f'font-family="{MONO}" font-size="{size}" font-weight="500" '
                 f'letter-spacing="1.7">{esc(label)}</text>')

    # 連線 --------------------------------------------------------------
    def arrow(self, x1, y1, x2, y2, dash=False, color=WIRE, head="ar"):
        d = ' stroke-dasharray="4 5"' if dash else ""
        self.add(f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{color}" '
                 f'stroke-width="1.4"{d} marker-end="url(#{head})"/>')

    def route(self, sx, sy, tx, ty, midy, dash=False, color=WIRE, head="ar", r=11):
        """下 → 橫 → 下的直角走線，轉角倒圓。"""
        s = 1 if tx > sx else -1
        rr = min(r, abs(tx - sx) / 2, abs(midy - sy), abs(ty - midy)) or 0
        d = (f"M {sx} {sy} L {sx} {midy - rr} "
             f"Q {sx} {midy} {sx + s * rr} {midy} "
             f"L {tx - s * rr} {midy} "
             f"Q {tx} {midy} {tx} {midy + rr} L {tx} {ty}")
        da = ' stroke-dasharray="4 5"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.4" '
                 f'stroke-linecap="round"{da} marker-end="url(#{head})"/>')

    def poly(self, pts, dash=False, color=WIRE, head="ar", r=12):
        """經過一串轉折點的直角走線，轉角倒圓。"""
        d = f"M {pts[0][0]} {pts[0][1]}"
        for i in range(1, len(pts) - 1):
            (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
            rr = min(r, abs(x1 - x0) + abs(y1 - y0), abs(x2 - x1) + abs(y2 - y1)) / 1.0
            rr = min(rr, abs(x1 - x0) or rr, abs(y1 - y0) or rr,
                     abs(x2 - x1) or rr, abs(y2 - y1) or rr)
            ax = x1 - rr * (1 if x1 > x0 else -1 if x1 < x0 else 0)
            ay = y1 - rr * (1 if y1 > y0 else -1 if y1 < y0 else 0)
            bx = x1 + rr * (1 if x2 > x1 else -1 if x2 < x1 else 0)
            by = y1 + rr * (1 if y2 > y1 else -1 if y2 < y1 else 0)
            d += f" L {ax} {ay} Q {x1} {y1} {bx} {by}"
        d += f" L {pts[-1][0]} {pts[-1][1]}"
        da = ' stroke-dasharray="4 5"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.4" '
                 f'stroke-linecap="round"{da} marker-end="url(#{head})"/>')

    # 內嵌影像 -----------------------------------------------------------
    def image(self, x, y, w, h, blob: bytes, caption=None):
        b64 = base64.b64encode(blob).decode()
        self.add(f'<rect x="{x - 3}" y="{y - 3}" width="{w + 6}" height="{h + 6}" '
                 f'rx="8" fill="none" stroke="{LINE}" stroke-width="1"/>')
        self.add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
                 f'preserveAspectRatio="xMidYMid slice" '
                 f'href="data:image/webp;base64,{b64}"/>')
        if caption:
            self.add(f'<text x="{x + w / 2}" y="{y + h + 20}" fill="{INK_2}" '
                     f'font-family="{MONO}" font-size="10" letter-spacing="1.1" '
                     f'text-anchor="middle">{esc(caption)}</text>')

    def title(self, x, y, text, size=13, color=INK_2, anchor="start", mono=True):
        f = MONO if mono else FONT
        ls = ' letter-spacing="1.8"' if mono else ""
        self.add(f'<text x="{x}" y="{y}" fill="{color}" font-family="{f}" '
                 f'font-size="{size}"{ls} text-anchor="{anchor}">{esc(text)}</text>')

    def dump(self) -> str:
        heads = "\n".join(
            f'    <marker id="{i}" viewBox="0 0 10 10" refX="8.5" refY="5" '
            f'markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
            f'<path d="M0 0.8 L9.5 5 L0 9.2 z" fill="{c}"/></marker>'
            for i, c in (("ar", WIRE), ("arg", GREEN), ("arc", CYAN), ("ara", AMBER)))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'xmlns:xlink="http://www.w3.org/1999/xlink" '
                f'viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'role="img" aria-label="Mask R-CNN 產生旋轉框，轉正後的 crop 進入 '
                f'HRNet-w32 encoder 與 U-Net decoder，輸出牙齒遮罩">\n'
                f'  <defs>\n{heads}\n  </defs>\n'
                f'  <rect width="{self.w}" height="{self.h}" fill="{BG}"/>\n'
                + "\n".join(self.body) + "\n</svg>\n")


# ── 三張真的縮圖 ──────────────────────────────────────────────────────
def webp_of(img: np.ndarray, q: int = 88) -> bytes:
    """縮圖以 WebP 內嵌；同樣畫質下 SVG 檔案大小約是 PNG 的三分之一。"""
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, q])
    if not ok:
        raise RuntimeError("WebP 編碼失敗")
    return buf.tobytes()


def original_insets() -> tuple[bytes, bytes, bytes]:
    """原圖檔裡那三張。三張都是 340×495，所以版面用同一個框。"""
    d = ROOT / "figs" / "insets"
    return tuple(webp_of(cv2.imread(str(d / n)), 90)
                 for n in ("1_input.png", "2_obb.png", "3_mask.png"))


@torch.no_grad()
def model_insets(name: str, tooth: int) -> tuple[bytes, bytes, bytes]:
    """回傳（原圖, 轉正的 crop, 遮罩）的 WebP，以及 crop 的長寬比。"""
    gray = cv2.imread(str(ROOT / "testset" / f"{name}.jpg"), cv2.IMREAD_GRAYSCALE)
    det, segs = load_detector(), load_segmenters()
    boxes = detect_obb(det, gray)
    boxes.sort(key=lambda b: b[0])
    box = boxes[tooth]

    M, cw, ch = warp_of(*box, PAD)
    crop = cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR)
    prob = cv2.resize(predict(segs, crop, use_tta=False), (cw, ch),
                      interpolation=cv2.INTER_LINEAR)
    mask = (prob > 0.5).astype(np.uint8)

    # 原圖上標出旋轉框
    shot = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    quad = cv2.boxPoints(((box[0], box[1]), (box[2], box[3]), box[4]))
    cv2.polylines(shot, [np.int32(quad)], True, (232, 195, 79), 4, cv2.LINE_AA)

    cropc = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    out = np.zeros_like(cropc)
    out[mask.astype(bool)] = (235, 235, 235)

    def fit(im, w):
        h = int(im.shape[0] * w / im.shape[1])
        return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return webp_of(fit(shot, 340)), webp_of(fit(cropc, 240)), webp_of(fit(out, 240))


# ── 兩種畫法 ──────────────────────────────────────────────────────────
# 預設照學長看過的那版
UP_LABEL = "Transposed|Conv"
BLOCK_PARTS = ["Concatenation", "DoubleConv", UP_LABEL]
DOUBLECONV = ["Conv", "BN", "ReLU", "Conv", "BN"]
HEAD_K = "3×3"


def as_code() -> None:
    """改成與 segmentation_models_pytorch 的實作一致。

    UnetDecoderBlock.forward 實際上是
        F.interpolate → concat(skip) → Conv3×3-BN-ReLU → Conv3×3-BN-ReLU
    所以三處要改：沒有轉置卷積、上採樣在最前面不是最後、DoubleConv 末端
    還有一個 ReLU。
    """
    global UP_LABEL, BLOCK_PARTS, DOUBLECONV
    UP_LABEL = "Upsample|×2"
    BLOCK_PARTS = [UP_LABEL, "Concatenation", "DoubleConv"]
    DOUBLECONV = ["Conv", "BN", "ReLU", "Conv", "BN", "ReLU"]


# ── 版面 ──────────────────────────────────────────────────────────────
def build(pa: bytes, obb: bytes, mask: bytes) -> Svg:
    W, H = 1240, 880
    s = Svg(W, H)
    iw, ih = 112, 163                       # 三張縮圖都是 340×495，共用一個框

    # ── Stage 1：Mask R-CNN ──────────────────────────────────────────
    gy, gh = 74, 186
    s.group(252, gy, 720, gh, "STAGE 1 · MASK R-CNN", CYAN, 13.5, True)
    iy = gy + (gh - ih) / 2
    s.image(46, iy, iw, ih, pa)
    s.title(46 + iw / 2, iy - 13, "INPUT PA", 9.5, INK_2, "middle")

    nw, nh, nx = 148, 66, 276
    ny = gy + (gh - nh) / 2
    for i, (lab, sub) in enumerate((("ResNet-50", "BACKBONE"), ("FPN", "NECK"),
                                    ("RPN", "PROPOSALS"), ("RoI heads", "MASK"))):
        s.node(nx + i * (nw + 30), ny, nw, nh, lab, CYAN, sub)
        if i:
            s.arrow(nx + i * (nw + 30) - 27, ny + nh / 2,
                    nx + i * (nw + 30) - 5, ny + nh / 2)
    s.arrow(46 + iw + 8, gy + gh / 2, 248, gy + gh / 2)

    ix = 1030
    s.arrow(976, gy + gh / 2, ix - 9, gy + gh / 2, color=AMBER, head="ara")
    s.image(ix, iy, iw, ih, obb)
    # 箭頭上的空隙只有 50 px，放不下字，標題往上疊兩行
    s.title(ix + iw / 2, iy - 29, "minAreaRect", 9.5, AMBER, "middle")
    s.title(ix + iw / 2, iy - 13, "旋轉框 OBB", 9.5, INK_2, "middle")

    # ── Stage 2：HRNet-w32 encoder ───────────────────────────────────
    ey, eh = 326, 140
    s.group(40, ey, 1160, eh, "STAGE 2 · HRNET-W32 ENCODER", GREEN, 13.5, True)
    ew, ehh = 204, 70
    egap = (1160 - 60 - 5 * ew) / 4
    enc_y = ey + 30
    enc_x = [70 + i * (ew + egap) for i in range(5)]
    for x, (lab, sub) in zip(enc_x, (("Stem", None), ("HRNet|stage 1", None),
                                     ("HRNet|stage 2", None), ("HRNet|stage 3", None),
                                     ("HRNet|stage 4", None))):
        s.node(x, enc_y, ew, ehh, lab, SLATE if lab == "Stem" else GREEN, sub)
    for i in range(4):
        s.arrow(enc_x[i] + ew + 3, enc_y + ehh / 2, enc_x[i + 1] - 5,
                enc_y + ehh / 2)
    # 從左側進 Stem。若照原本從正上方落下，會正好壓過區塊標題那一行字。
    bus, lane = 288, 20
    cy = enc_y + ehh / 2
    s.poly([(ix + iw / 2, iy + ih + 6), (ix + iw / 2, bus), (lane, bus),
            (lane, cy), (enc_x[0] - 6, cy)], color=AMBER, head="ara")
    s.title(ix + iw / 2 - 16, bus - 10, "warp 轉正", 9.5, AMBER, "end")

    # ── U-Net decoder ────────────────────────────────────────────────
    dy, dh = 578, 132
    s.group(252, dy, 948, dh, "U-NET DECODER", AMBER, 13.5, True)
    dw, dhh = 134, 68
    dgap = (948 - 40 - 6 * dw) / 5
    dec_y = dy + 30
    dec_x = [272 + i * (dw + dgap) for i in range(6)]          # 左→右
    s.node(dec_x[0], dec_y, dw, dhh, "Conv", SLATE, HEAD_K)
    for i in (1, 2, 3, 4):
        s.node(dec_x[i], dec_y, dw, dhh, "U-Net|block", AMBER)
    s.node(dec_x[5], dec_y, dw, dhh, UP_LABEL, AMBER)
    for i in range(5):                                         # 資料由右往左
        s.arrow(dec_x[i + 1] - 5, dec_y + dhh / 2, dec_x[i] + dw + 3,
                dec_y + dhh / 2)

    # 跳接：encoder 由深到淺對到 decoder 由右到左，走線不交叉
    ec = [x + ew / 2 for x in enc_x]
    dc = [x + dw / 2 for x in dec_x]
    s.route(ec[4], enc_y + ehh + 4, dc[5], dec_y - 8, 520, color=GREEN, head="arg")
    for src, dst, y in ((3, 4, 498), (2, 3, 516), (1, 2, 534), (0, 1, 552)):
        s.route(ec[src], enc_y + ehh + 4, dc[dst], dec_y - 8, y,
                dash=True, color=GREEN, head="arg")

    my = dec_y + dhh / 2 - ih / 2
    s.arrow(dec_x[0] - 5, dec_y + dhh / 2, 46 + iw + 14, dec_y + dhh / 2)
    s.image(46, my, iw, ih, mask)
    s.title(46 + iw / 2, my - 13, "TOOTH MASK", 9.5, INK_2, "middle")

    # ── 細部：U-Net block 與 DoubleConv ──────────────────────────────
    by, bh = 776, 80
    for gx, gw, name, items in ((40, 556, "U-NET BLOCK", BLOCK_PARTS),
                                (644, 556, "DOUBLECONV", DOUBLECONV)):
        s.group(gx, by, gw, bh, name, INK_3)
        n = len(items)
        piw = (gw - 40 - (n - 1) * 22) / n
        for i, lab in enumerate(items):
            x = gx + 20 + i * (piw + 22)
            s.node(x, by + 15, piw, 50, lab, SLATE, r=9)
            if i:
                s.arrow(x - 19, by + 40, x - 5, by + 40)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-model", action="store_true",
                    help="縮圖改用模型當場跑一次的結果，而不是原圖檔裡那三張")
    ap.add_argument("--image", default="113", help="--from-model 時用測試集的哪一張")
    ap.add_argument("--tooth", type=int, default=0, help="第幾顆牙，由左至右從 0 起算")
    ap.add_argument("--out", default=str(ROOT / "figs" / "arch.svg"))
    ap.add_argument("--as-code", action="store_true",
                    help="照 smp 的實際實作畫（Upsample×2、DoubleConv 末端多一個 "
                         "ReLU），而不是學長看過的那版")
    args = ap.parse_args()

    if args.as_code:
        as_code()
    if args.from_model:
        print(f"跑 {args.image}.jpg 取縮圖…（偵測門檻 {DET_THR}）")
        shots = model_insets(args.image, args.tooth)
    else:
        shots = original_insets()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(*shots).dump(), encoding="utf-8")
    print(f"{out}  {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
