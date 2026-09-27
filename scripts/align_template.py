"""粗定位：把標準形狀擺到種子上，用網格搜尋找出最佳的擺法。

五個參數：平移 x、平移 y、旋轉、水平縮放、垂直縮放。

目標函數
--------
    J = 模板內部的平均灰階 − 模板外圈的平均灰階

外圈是把模板往外膨脹一圈再扣掉模板本身所得的環。這個函數會**自動偏好正確
的尺寸**，不必額外加懲罰項：

    模板太小，縮在牙齒中央   → 內部亮、外圈也亮 → 差值小
    模板太大，超出牙齒       → 內部混進暗背景   → 差值小
    剛好貼合                 → 內部亮、外圈暗   → 差值最大

搜尋策略
--------
由粗到細的網格搜尋：先用大步長掃過整個範圍，取最佳的一組，再在它附近用一半
的步長掃一次，重複數輪。**沒有梯度、完全確定性**，每一步都是「試幾個候選、
取分數最好的」——跟 find_cej、find_alveolar_crest 的變點擬合是同一個模式，
只是參數從一個變成五個。

旋轉範圍必須涵蓋整個 360°：測試集裡牙冠朝上與朝下的影像都有。

用法：
    py scripts/align_template.py shape_prior_seg_test --only 13
    py scripts/align_template.py shape_prior_seg_test -o align.png
    py scripts/align_template.py shape_prior_seg_test -o align.png --only 13 1 4 21 
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.find_seeds import binarize, find_seeds  # noqa: E402
from scripts.labelme_io import make_figure, save_or_show  # noqa: E402
from scripts.shape_template import ShapeTemplate, build_template  # noqa: E402


@dataclass
class Placement:
    """一次擺放：五個參數與它的得分。"""

    tx: float
    ty: float
    angle: float
    scale_x: float
    scale_y: float
    score: float

    @property
    def params(self) -> tuple:
        return (self.tx, self.ty, self.angle, self.scale_x, self.scale_y)


# 牙齒偏離垂直的最大容許角度（度）。根尖片上牙齒沿咬合方向站立，超過這個
# 傾斜度的擺放在解剖上不可能出現。
MAX_TILT = 40.0

# 水平與垂直縮放的比值上限。模板已經是一顆真實牙齒的形狀，實際牙齒之間雖有
# 胖瘦差異，長寬比卻相當穩定，不會壓成細長條。少了這個限制，搜尋會把模板
# 擠成寬 0.43 長 1.04 的窄條貼到根管那類亮線上——得分很高，形狀卻不是牙齒。
MAX_ANISOTROPY = 1.3

def score_placement(img: np.ndarray, template: ShapeTemplate, params: tuple,
                    ring_fraction: float = 0.04) -> float:
    """算一次擺放的得分：模板內部與外圈灰階的兩樣本 t 統計量。

    輸入：灰階影像、模板、五個參數 (tx, ty, angle, scale_x, scale_y)。
    輸出：得分（越大越貼合）；模板落到影像外太多時回傳 -inf。

    只在模板的外接矩形範圍內作業，不掃整張圖——搜尋要跑上千次，這是必要的。
    """
    contour = template.place(*params)

    x0, y0 = np.floor(contour.min(axis=0)).astype(int)
    x1, y1 = np.ceil(contour.max(axis=0)).astype(int)
    pad = int(max(x1 - x0, y1 - y0) * ring_fraction) + 2
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad

    # **必須完全落在影像內。**允許部分超出時，外圈會壓在影像邊框的黑色區域
    # 上，對比虛高，搜尋就會把模板推向邊緣——實測最佳解會落在 x=824（影像
    # 寬 825）這種位置。感測器的黑色圓角也是同樣的陷阱。
    if x0 < 0 or y0 < 0 or x1 >= img.shape[1] or y1 >= img.shape[0]:
        return float("-inf")

    cx0, cy0, cx1, cy1 = x0, y0, x1, y1
    roi = img[cy0:cy1, cx0:cx1]
    local = contour - np.array([cx0, cy0])

    # 外圈用「放大後的輪廓減掉原輪廓」，不用 cv2.dilate。膨脹的核心大小得跟
    # 牙齒尺寸成正比（約 100 px），那樣的形態學運算在上萬次評分裡是主要瓶頸；
    # 改成對輪廓本身放大，只要多做一次 fillPoly，快兩個數量級。
    centre = local.mean(axis=0)
    outer = (local - centre) * (1.0 + ring_fraction) + centre

    filled = np.zeros(roi.shape, np.uint8)
    cv2.fillPoly(filled, [local.astype(np.int32)], 1)
    cv2.fillPoly(filled, [outer.astype(np.int32)], 2, offset=(0, 0))
    # 先畫內圈再畫外圈會蓋掉內圈，所以反過來畫：外圈標 2、再把內圈標回 1
    cv2.fillPoly(filled, [local.astype(np.int32)], 1)

    inside = roi[filled == 1]
    ring = roi[filled == 2]
    if inside.size < 50 or ring.size < 50:
        return float("-inf")

    # 用 t 統計量而非單純的平均差。平均差**與面積無關**，因此系統性偏好小解：
    # 小模板貼在局部高對比處，平均值不會被稀釋；蓋住整顆牙的模板卻得把對比
    # 較弱的牙根段一起平均進去，反而輸掉。實測 18.jpg 的第一名是 0.34x0.40
    # （53.5 分），尺寸正確的 0.64x0.69 只拿到 50.7 分——正確解一直都在，
    # 只是被這個偏差壓下去。
    #
    # t 隨樣本數以 sqrt(n) 成長，所以對比撐得住時大模板會勝出；對比若只是
    # 局部僥倖，變異數大、樣本少，t 就撐不起來。分母用 Welch 形式，不假設
    # 內外的變異數相同（牙齒內部本來就比周圍均勻）。
    spread = np.sqrt(inside.var() / inside.size + ring.var() / ring.size)
    if spread < 1e-6:
        return float("-inf")

    return float((inside.mean() - ring.mean()) / spread)


def coarse_to_fine(img: np.ndarray, template: ShapeTemplate, start: tuple,
                   spans: tuple, steps: tuple, bounds: tuple, rounds: int = 3,
                   ring_fraction: float = 0.04) -> Placement:
    """由粗到細的網格搜尋。

    輸入：起始參數、各參數的搜尋半徑 spans、初始步長 steps、各參數的
          (下限, 上限) bounds、精修輪數。
    輸出：得分最高的 Placement。

    每一輪掃過以目前最佳解為中心的網格，然後把搜尋半徑與步長都減半，再掃一次。

    **界限是必要的**。少了它，搜尋會找到退化解：把模板壓成一條細長條
    （實測縮放 0.03 x 2.08）貼在某條亮線上，或讓縮放變成負值而把形狀鏡像。
    這些在數學上得分很高，在解剖上毫無意義。界限把搜尋限制在「還像一顆牙」
    的範圍內，是形狀先驗的一部分。長寬比另外用 MAX_ANISOTROPY 限制——那不是
    單一參數的上下界能表達的，必須在網格裡逐點檢查。
    """
    def clip(params):
        return tuple(float(np.clip(v, lo, hi)) for v, (lo, hi) in zip(params, bounds))

    start = clip(start)
    best = Placement(*start, score_placement(img, template, start, ring_fraction))

    for _ in range(rounds):
        grids = [np.clip(np.arange(c - s, c + s + 1e-9, st), lo, hi) if st > 0
                 else np.array([c])
                 for c, s, st, (lo, hi) in zip(best.params, spans, steps, bounds)]
        for tx in grids[0]:
            for ty in grids[1]:
                for angle in grids[2]:
                    for sx in grids[3]:
                        for sy in grids[4]:
                            params = (tx, ty, angle, sx, sy)
                            if not (1 / MAX_ANISOTROPY <= sx / sy <= MAX_ANISOTROPY):
                                continue  # 形狀被壓扁或拉長到不像牙齒
                            value = score_placement(img, template, params, ring_fraction)
                            if value > best.score:
                                best = Placement(*params, value)
        spans = tuple(s / 2 for s in spans)
        steps = tuple(st / 2 for st in steps)

    return best


def align_at_seed(img: np.ndarray, template: ShapeTemplate,
                  seed: np.ndarray, radius: float, ring_fraction: float = 0.04) -> Placement:
    """在一個種子上擺放模板。

    初始尺度由種子的內切圓半徑推得——那個半徑約等於該處的牙齒半寬，所以
    半徑除以模板自身的半寬就是合理的縮放倍率，不是憑空給的猜測。

    旋轉只在**接近垂直**的範圍內搜尋。根尖片上的牙齒必定大致沿咬合方向站立——
    下顎牙冠朝上（約 0°）、上顎牙冠朝下（約 180°），不會橫躺。所以只掃這兩族
    各 ±MAX_TILT 度。這是解剖上的事實，不是調出來的參數，作用和 CEJ 擬合裡
    「斜率必須為正」一樣：把數學上得分高、解剖上不可能的解直接排除。
    （實測不加這個約束時，失敗的擺放清一色是 −82°、−135° 這類橫躺解。）
    """
    scale = radius / template.half_width

    # 初始掃描必須同時掃角度**與沿軸的平移**。種子落在牙冠上，但模板的形心
    # 在牙齒中段（實測距牙冠端約 41%），直接把形心放到種子上會讓整個模板往
    # 牙冠方向偏移一大截；在那個錯誤位置上比較各角度的得分，選出來的角度也
    # 會是錯的。
    shift = template.length * scale * 0.4
    angles = np.concatenate([np.arange(-MAX_TILT, MAX_TILT + 1e-9, 5.0),
                             np.arange(180 - MAX_TILT, 180 + MAX_TILT + 1e-9, 5.0)])
    best = None
    for angle in angles:
        theta = np.radians(angle)
        for offset in (-shift, 0.0, shift):
            params = (float(seed[0]) + offset * np.sin(theta),
                      float(seed[1]) - offset * np.cos(theta),
                      float(angle), scale, scale)
            value = score_placement(img, template, params, ring_fraction)
            if best is None or value > best.score:
                best = Placement(*params, value)

    # 角度的界限跟著初選的那一族走（朝上或朝下），精修時不得跨族亂跑
    upright = 0.0 if abs(best.angle) <= 90 else 180.0
    bounds = (
        (seed[0] - 80, seed[0] + 80),
        (seed[1] - template.length * scale * 0.7, seed[1] + template.length * scale * 0.7),
        (upright - MAX_TILT, upright + MAX_TILT),
        (scale * 0.55, scale * 1.8),  # 縮放必須為正、且不得偏離種子的估計太遠
        (scale * 0.55, scale * 1.8),
    )

    return coarse_to_fine(
        img, template, best.params,
        spans=(30.0, 60.0, 20.0, scale * 0.4, scale * 0.4),
        steps=(15.0, 20.0, 10.0, scale * 0.2, scale * 0.2),
        bounds=bounds,
        rounds=3,
        ring_fraction=ring_fraction,
    )


def align_image(img: np.ndarray, template: ShapeTemplate, min_radius: float,
                max_radius: float, spacing: float, downsample: int = 4,
                ring_fraction: float = 0.04) -> list[Placement]:
    """對一張影像的每個種子各做一次粗定位。

    粗定位在**降採樣後的影像**上進行。搜尋要跑上萬次評分，而粗定位只需要
    大致的位置與尺度，全解析度是浪費——降四倍即少了十六倍的像素。回傳的
    參數已換算回原始解析度。
    """
    binary, _ = binarize(img)
    _, seeds, radii = find_seeds(binary, min_radius, max_radius, spacing)

    small = cv2.resize(img, None, fx=1 / downsample, fy=1 / downsample,
                       interpolation=cv2.INTER_AREA)

    results = []
    for seed, radius in zip(seeds, radii):
        placement = align_at_seed(small, template, seed / downsample,
                                  radius / downsample, ring_fraction)
        results.append(Placement(
            placement.tx * downsample, placement.ty * downsample, placement.angle,
            placement.scale_x * downsample, placement.scale_y * downsample,
            placement.score,
        ))
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", help="含有 .jpg 的資料夾")
    parser.add_argument("--only", nargs="+", help="只處理這幾張")
    parser.add_argument("--template", default="labeled_PA/13.json", help="標準形狀的來源")
    parser.add_argument("--min-radius", type=float, default=30.0)
    parser.add_argument("--max-radius", type=float, default=170.0)
    parser.add_argument("--spacing", type=float, default=140.0)
    parser.add_argument("--ring-fraction", type=float, default=0.04,
                        help="外圈厚度佔模板尺寸的比例，預設 0.04。太厚會越過牙周膜間隙"
                             "那條細暗線、壓到亮骨頭上，使牙根段失去對比")
    parser.add_argument("--downsample", type=int, default=4,
                        help="粗定位時的降採樣倍率，預設 4")
    parser.add_argument("--top", type=int, default=0,
                        help="只畫得分最高的前 N 個，0 表示全部")
    parser.add_argument("-o", "--out", help="存檔到此路徑，不開視窗")
    args = parser.parse_args()

    folder = Path(args.folder)
    files = sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if args.only:
        files = [f for f in files if f.stem in set(args.only)]
    if not files:
        raise SystemExit(f"在 {folder} 找不到影像")

    template = build_template(args.template)
    print(f"模板 {args.template}：長 {template.length:.0f} px，半寬 {template.half_width:.0f} px\n")

    cols = min(len(files), 5)
    rows = (len(files) + cols - 1) // cols
    plt, fig, axes = make_figure(rows, cols, (4.2 * cols, 6.0 * rows), args.out)

    for index, path in enumerate(files):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        placements = align_image(img, template, args.min_radius, args.max_radius,
                                 args.spacing, args.downsample, args.ring_fraction)
        placements.sort(key=lambda p: p.score, reverse=True)
        shown = placements[:args.top] if args.top else placements

        print(f"{path.name}  {len(placements)} 個擺放，得分 "
              f"{placements[-1].score:.1f} ~ {placements[0].score:.1f}")
        for p in shown[:4]:
            print(f"   得分 {p.score:6.1f}  位置 ({p.tx:4.0f},{p.ty:4.0f})  "
                  f"旋轉 {p.angle:+6.1f}°  縮放 {p.scale_x:.2f} x {p.scale_y:.2f}")

        ax = axes[index // cols][index % cols]
        ax.imshow(img, cmap="gray")
        for rank, p in enumerate(shown):
            contour = np.vstack([template.place(*p.params)] * 1)
            contour = np.vstack([contour, contour[:1]])
            colour = "#00e5ff" if rank == 0 else "#ff8c42"
            ax.plot(contour[:, 0], contour[:, 1], "-", color=colour,
                    linewidth=1.2 if rank == 0 else 0.6,
                    alpha=1.0 if rank == 0 else 0.55)
            ax.plot(p.tx, p.ty, "o", color=colour, markersize=3)
        ax.set_title(f"{path.stem}　{len(shown)} 個　最高分 {placements[0].score:.1f}",
                     fontsize=9)
        ax.axis("off")

    for index in range(len(files), rows * cols):
        axes[index // cols][index % cols].axis("off")

    save_or_show(plt, args.out, dpi=100)


if __name__ == "__main__":
    main()
