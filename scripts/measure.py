"""最終程式：從牙齒遮罩與原影像量出 CRR、ABLR、B-CRR。

這支把前面各步驟串起來，是實際要用的版本；其餘 find_* 各支保留為可單獨
執行的診斷工具，用來檢視中間過程。

流程
----
    find_axis_raw      整顆遮罩做 PCA → 初始座標系與牙冠方向
    find_cej           → C、D（CEJ）
    find_ridge         → A、B（邊緣嵴）
    find_axis_paper    → G（根尖）、精修長軸、H I J K L
    find_alveolar_crest → E、F（齒槽脊）→ R、Q、S

指標（皆為沿長軸的長度比值）
--------------------------
    CRR    = JL / KL          牙冠根比
    ABLR    = 1 - SK / LK     平均齒槽骨流失比（S 為兩側骨脊中點）
    Max BLR                   最大骨流失比（近遠心取較嚴重的一側）
    B-CRR   = JS / KS         骨平面牙冠根比

論文另給 B-CRR = (CRR + ABLR) / (1 - ABLR)，代數上與 JS/KS 等價，程式兩種
都算並比對，不符即代表實作有誤——這是免費的自我檢查。

所有指標都是**軸向長度的比值**，而沿固定方向投影到不同方向的軸上時，
長度差的比值不變（分母的 u x c 會消掉）。因此**這些指標與長軸方向無關**，
論文中無法客觀複現的長軸目視判定並不影響結果。長軸仍然重要，但影響是
透過特徵點的偵測（半寬剖面、左右分側都依賴它），而非透過投影。

用法：
    py scripts/measure.py labeled_PA
    py scripts/measure.py labeled_PA --csv results.csv
    py scripts/measure.py labeled_PA -o measure.png
    py scripts/measure.py labeled_PA -o measure.png --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.find_alveolar_crest import find_crest  # noqa: E402
from scripts.find_axis_paper import PaperAxis, axial, mark, refine_axis  # noqa: E402
from scripts.labelme_io import (  # noqa: E402
    MASK_RGBA,
    collect_annotations,
    load_tooth,
    make_figure,
    overlay,
    save_or_show,
)

LANDMARK_COLOR = "#1f6feb"  # A~G：直接量到的
DERIVED_COLOR = "#8a8a8a"  # H~S：推演出來的
LINE_COLOR = "black"
LINE_WIDTH = 0.6


@dataclass
class Measurement:
    """一顆牙的完整量測結果。"""

    name: str
    axis: PaperAxis  # 長軸與 A B C D G、H I J K L
    landmarks: dict  # 併入 E、F 之後的全部特徵點（影像座標）
    levels: dict  # 併入 R、Q、S 之後的全部軸向高度（精修座標系）
    crest: dict  # 兩側齒槽脊的擬合診斷

    @property
    def crr(self) -> float:
        return self.axis.crr

    def _loss_at(self, level: float) -> float:
        """某個高度處的骨流失比：該處到根尖的距離佔整段牙根的比例，取補數。"""
        lv = self.levels
        return 1 - (level - lv["K"]) / (lv["L"] - lv["K"])

    @property
    def ablr(self) -> float:
        """平均齒槽骨流失比 = 1 - SK/LK，S 是兩側骨脊的中點。"""
        return self._loss_at(self.levels["S"])

    @property
    def max_blr(self) -> float:
        """最大骨流失比：近遠心兩側取流失較嚴重者。

        論文的 Table 1、2 同時報告 Average BLR 與 Max BLR，迴歸模型 2 用的
        就是後者。骨流失越嚴重代表骨脊越低，所以取 R、Q 中較低的那個高度。
        """
        return self._loss_at(min(self.levels["R"], self.levels["Q"]))

    @property
    def b_crr(self) -> float:
        """骨平面牙冠根比 = JS / KS。"""
        lv = self.levels
        return (lv["J"] - lv["S"]) / (lv["S"] - lv["K"])

    @property
    def b_crr_formula(self) -> float:
        """論文另給的算式，用來跟 JS/KS 對帳。"""
        return (self.crr + self.ablr) / (1 - self.ablr)

    @property
    def consistency(self) -> float:
        """兩種 B-CRR 算法的差，正常應為 0（浮點誤差等級）。"""
        return abs(self.b_crr - self.b_crr_formula)


def measure(name: str, img, mask, drop_apical: float, cut_at: float, threshold: float,
            surface_fraction: float, inner_fraction: float, outer_fraction: float,
            root_fraction: float) -> Measurement:
    """跑完整條流程。

    輸入：影像、牙齒遮罩，以及各步驟的參數。
    輸出：Measurement，含所有特徵點、軸向高度與三個指標。
    """
    axis = refine_axis(img, mask, drop_apical, cut_at, threshold,
                       surface_fraction, root_fraction)

    cej_width = float(np.hypot(axis.landmarks["D"][0] - axis.landmarks["C"][0],
                               axis.landmarks["D"][1] - axis.landmarks["C"][1]))
    crest = find_crest(img, mask, axis,
                       inner_fraction * cej_width, outer_fraction * cej_width)

    # E、F 目前是以 raw 座標系的高度求出的，換算回影像座標後再投影到精修軸，
    # 才能跟 H I J K L 放在同一套座標裡比較。
    landmarks = dict(axis.landmarks)
    levels = dict(axis.levels)
    for name_, side, level_key in (("E", -1, "R"), ("F", +1, "Q")):
        level_raw = crest[name_].level
        x_prime, y_prime = axis.frame_raw.to_frame(mask_points_of(mask))
        near = np.abs(y_prime - level_raw) < 3
        surface = float(np.abs(x_prime[near]).max()) if near.any() else 0.0
        point = axis.frame_raw.to_image(np.array([side * surface]),
                                        np.array([level_raw]))[0]
        landmarks[name_] = point
        levels[level_key] = float(axial(point[None, :], axis.frame, axis.cd_slope)[0])

    levels["S"] = (levels["R"] + levels["Q"]) / 2
    return Measurement(name, axis, landmarks, levels, crest)


def mask_points_of(mask):
    """遮罩的前景像素座標（與 labelme_io.mask_points 同義，避免循環 import）。"""
    ys, xs = np.nonzero(mask)
    return np.column_stack([xs, ys]).astype(float)


# --------------------------------------------------------------------------
# 作圖：參照論文 Fig.1
# --------------------------------------------------------------------------

# 每個推演點對應的來源點，用來畫「過該點平行於 CD」的投影線
PROJECTION = (("A", "H"), ("B", "I"), ("C", "L"), ("D", "L"),
              ("E", "R"), ("F", "Q"), ("G", "K"))
# 標籤偏移量：H I J 與 R Q S 各自高度相近，左右交錯才不會疊在一起
LABEL_OFFSET = {"H": (5, -12), "I": (5, 4), "J": (-14, -4),
                "R": (5, -12), "Q": (5, 4), "S": (-14, -4),
                "K": (5, -12), "L": (5, -12)}


def draw(ax, img, mask, result: Measurement, scale: float = 1.0) -> None:
    """把整個作圖過程畫出來：遮罩、長軸、投影線、所有點位。

    scale 放大點與字的尺寸，供大尺寸的方法流程圖使用——那裡的畫布比這支程式
    自己的輸出大好幾倍，沿用原尺寸的話字會小到看不清。
    """
    frame = result.axis.frame
    points = mask_points_of(mask)
    _, y_prime = frame.to_frame(points)

    ax.imshow(img, cmap="gray")
    overlay(ax, mask.shape, points, MASK_RGBA)

    top, bottom = frame.point_at(float(y_prime.max())), frame.point_at(float(y_prime.min()))
    ax.plot([bottom[0], top[0]], [bottom[1], top[1]], "-",
            color=LINE_COLOR, linewidth=LINE_WIDTH)

    for src, dst in PROJECTION:
        p, q = result.landmarks[src], frame.point_at(result.levels[dst])
        ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=LINE_COLOR, linewidth=LINE_WIDTH)

    for name, p in result.landmarks.items():
        mark(ax, p, name, LANDMARK_COLOR, 4.5 * scale, 10 * scale,
             (6 * scale, 3 * scale))
    for name, level in result.levels.items():
        mark(ax, frame.point_at(level), name, DERIVED_COLOR, 3.5 * scale,
             9 * scale, tuple(v * scale for v in LABEL_OFFSET[name]))

    ax.set_title(f"{result.name}\nCRR={result.crr:.3f}  ABLR={result.ablr:.3f}  "
                 f"B-CRR={result.b_crr:.3f}", fontsize=10)
    ax.axis("off")


def report(result: Measurement) -> None:
    lv = result.levels
    print(f"{result.name}")
    print("   " + "  ".join(f"{n}=({p[0]:.0f},{p[1]:.0f})" for n, p in result.landmarks.items()))
    print("   軸向高度：" + "  ".join(f"{n}={lv[n]:.1f}" for n in "HIJKLRQS"))
    print(f"   JL={lv['J'] - lv['L']:7.1f}   KL={lv['L'] - lv['K']:7.1f}   →  CRR   = {result.crr:.4f}")
    print(f"   SK={lv['S'] - lv['K']:7.1f}   LK={lv['L'] - lv['K']:7.1f}   →  ABLR  = {result.ablr:.4f}"
          f"　（近心 {result._loss_at(lv['R']):.4f}　遠心 {result._loss_at(lv['Q']):.4f}"
          f"　Max BLR = {result.max_blr:.4f}）")
    print(f"   JS={lv['J'] - lv['S']:7.1f}   KS={lv['S'] - lv['K']:7.1f}   →  B-CRR = {result.b_crr:.4f}")
    flag = "一致" if result.consistency < 1e-9 else f"**不一致，差 {result.consistency:.2e}**"
    print(f"   自我檢查：(CRR+ABLR)/(1-ABLR) = {result.b_crr_formula:.4f}   {flag}")
    for side in ("E", "F"):
        fit = result.crest[side]
        print(f"   {side} 側齒槽脊：對比 {fit.contrast:.1f}　階梯改善 {fit.improvement:.2f} 倍")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", help="labelme 的 .json 檔，或含有多個 .json 的資料夾")
    parser.add_argument("--drop-apical", type=float, default=0.25,
                        help="CEJ：擬合牙根基準線時捨去的根尖比例")
    parser.add_argument("--cut-at", type=float, default=1.0 / 3.0,
                        help="邊緣嵴：牙冠取樣範圍的下界")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="邊緣嵴：殘差閾值")
    parser.add_argument("--surface-fraction", type=float, default=0.10,
                        help="邊緣嵴：外緣點距牙齒表面的容差，取 CEJ 寬的比例")
    parser.add_argument("--inner-fraction", type=float, default=0.03,
                        help="齒槽脊：取樣起點離牙根表面的距離，取 CEJ 寬的比例")
    parser.add_argument("--outer-fraction", type=float, default=0.15,
                        help="齒槽脊：取樣終點離牙根表面的距離，取 CEJ 寬的比例")
    parser.add_argument("--root-fraction", type=float, default=2.0 / 3.0,
                        help="長軸：牙根保留冠側的比例")
    parser.add_argument("--csv", help="把結果寫成 CSV")
    parser.add_argument("-o", "--out", help="圖表存檔路徑，不給則開視窗")
    args = parser.parse_args()

    json_files = collect_annotations(args.target)
    plt, fig, axes = make_figure(1, len(json_files), (5.2 * len(json_files), 9), args.out)

    rows = []
    for col, jf in enumerate(json_files):
        _, img, mask, _ = load_tooth(jf)
        try:
            result = measure(jf.stem, img, mask, args.drop_apical, args.cut_at,
                             args.threshold, args.surface_fraction, args.inner_fraction,
                             args.outer_fraction, args.root_fraction)
        except ValueError as exc:
            print(f"{jf.stem}：失敗 - {exc}\n")
            axes[0][col].set_title(f"{jf.stem}  失敗", fontsize=11)
            axes[0][col].axis("off")
            continue

        report(result)
        draw(axes[0][col], img, mask, result)
        rows.append({
            "tooth": result.name,
            "CRR": round(result.crr, 4),
            "ABLR": round(result.ablr, 4),
            "MaxBLR": round(result.max_blr, 4),
            "B_CRR": round(result.b_crr, 4),
            **{f"{n}_level": round(result.levels[n], 1) for n in "HIJKLRQS"},
        })

    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"已寫入 {len(rows)} 筆 -> {args.csv}")

    save_or_show(plt, args.out, dpi=130)


if __name__ == "__main__":
    main()
