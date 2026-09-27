"""建立形狀先驗：從一顆手工標註的理想小臼齒取出標準形狀。

模板以「自身座標系」儲存：形心為原點、長軸垂直、牙冠朝上（y 為正）。
之後配準時只要對它套上平移、旋轉、水平／垂直縮放即可。

輪廓沿弧長等距重新取樣成固定點數，原因有二：labelme 的頂點在曲率大處密、
平直處疏，直接用會偏；而且之後要沿法線搜尋，點距均勻法線才穩定。

用法：
    py scripts/shape_template.py labeled_PA/13.json
    py scripts/shape_template.py labeled_PA/13.json -o template.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.find_axis_raw import fit_axis  # noqa: E402
from scripts.labelme_io import (  # noqa: E402
    LABEL_TOOTH,
    label_mask,
    load_annotation,
    load_image,
    make_figure,
    mask_points,
    save_or_show,
)

DEFAULT_TEMPLATE = "labeled_PA/13.json"
N_POINTS = 200


@dataclass
class ShapeTemplate:
    """標準形狀，存於自身座標系（形心為原點、長軸垂直、牙冠朝 +y）。"""

    contour: np.ndarray  # (N, 2)，等距取樣後的輪廓點
    length: float  # 沿長軸的全長
    half_width: float  # 離長軸最遠的距離
    source: str

    @property
    def box(self) -> tuple[float, float, float, float]:
        """外接方框 (x_min, x_max, y_min, y_max)。"""
        return (float(self.contour[:, 0].min()), float(self.contour[:, 0].max()),
                float(self.contour[:, 1].min()), float(self.contour[:, 1].max()))

    def place(self, tx: float, ty: float, angle: float,
              scale_x: float, scale_y: float) -> np.ndarray:
        """把模板擺到影像上。

        輸入：平移、旋轉（度）、水平與垂直縮放。
        輸出：變換後的輪廓點（影像座標）。

        順序是**先縮放、再旋轉、最後平移**。縮放不是各向同性的，所以順序
        會影響結果，必須固定下來——先在模板自身的座標系裡縮放（水平＝牙齒
        寬度方向、垂直＝長軸方向），再整體轉到影像的方向。

        縮放時把 y 反號：模板的 +y 是牙冠方向，但影像的 y 軸向下，不反號的話
        angle=0 會讓牙冠朝下，輸出的角度難以判讀。反號後 **angle=0 代表牙冠
        朝上、180° 代表朝下**。
        """
        scaled = self.contour * np.array([scale_x, -scale_y])
        theta = np.radians(angle)
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        return scaled @ rotation.T + np.array([tx, ty])


def resample_contour(contour: np.ndarray, n_points: int) -> np.ndarray:
    """沿弧長把封閉輪廓重新取樣成 n_points 個等距點。"""
    closed = np.vstack([contour, contour[:1]])
    steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    targets = np.linspace(0, arc[-1], n_points, endpoint=False)
    return np.column_stack([np.interp(targets, arc, closed[:, 0]),
                            np.interp(targets, arc, closed[:, 1])])


def build_template(json_path: str | Path, n_points: int = N_POINTS) -> ShapeTemplate:
    """從一份標註建立標準形狀。

    輸入：labelme 的 .json（需含標籤 "1" 的整顆牙齒）。
    輸出：ShapeTemplate。
    """
    data = load_annotation(json_path)
    mask = label_mask(data, LABEL_TOOTH)
    points = mask_points(mask)
    frame = fit_axis(points)

    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(found, key=cv2.contourArea).squeeze().astype(float)
    contour = resample_contour(contour, n_points)

    # 轉進牙齒自身的座標系：形心為原點、長軸垂直、牙冠朝 +y
    x_prime, y_prime = frame.to_frame(contour)
    canonical = np.column_stack([x_prime, y_prime])

    return ShapeTemplate(
        contour=canonical,
        length=float(np.ptp(y_prime)),
        half_width=float(np.abs(x_prime).max()),
        source=str(json_path),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", nargs="?", default=DEFAULT_TEMPLATE,
                        help=f"作為標準形狀的標註檔，預設 {DEFAULT_TEMPLATE}")
    parser.add_argument("-o", "--out", help="存檔到此路徑，不開視窗")
    args = parser.parse_args()

    template = build_template(args.source)
    x0, x1, y0, y1 = template.box
    print(f"模板來源：{template.source}")
    print(f"   輪廓點數 {len(template.contour)}")
    print(f"   長度 {template.length:.0f} px　最大半寬 {template.half_width:.0f} px")
    print(f"   外接方框 x [{x0:.0f}, {x1:.0f}]　y [{y0:.0f}, {y1:.0f}]"
          f"　（{x1 - x0:.0f} x {y1 - y0:.0f}）")

    plt, fig, axes = make_figure(1, 2, (11, 8), args.out)

    data = load_annotation(args.source)
    axes[0][0].imshow(load_image(data), cmap="gray")
    mask = label_mask(data, LABEL_TOOTH)
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    original = max(found, key=cv2.contourArea).squeeze()
    axes[0][0].plot(original[:, 0], original[:, 1], "-", color="#00e5ff", linewidth=1)
    axes[0][0].set_title("來源標註", fontsize=11)
    axes[0][0].axis("off")

    ax = axes[0][1]
    closed = np.vstack([template.contour, template.contour[:1]])
    ax.plot(closed[:, 0], closed[:, 1], "-", color="black", linewidth=1)
    ax.plot(template.contour[:, 0], template.contour[:, 1], ".", color="#1f6feb", markersize=2)
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               color="#d64545", linestyle="--", linewidth=0.8))
    ax.plot([0, 0], [y0, y1], "-", color="#8a8a8a", linewidth=0.6)
    ax.plot([x0, x1], [0, 0], "-", color="#8a8a8a", linewidth=0.6)
    ax.set_aspect("equal")
    ax.set_title(f"標準形狀（{len(template.contour)} 點等距取樣）\n"
                 f"長 {template.length:.0f} px，半寬 {template.half_width:.0f} px",
                 fontsize=10)
    ax.set_xlabel("← 牙冠朝上 →")

    save_or_show(plt, args.out)


if __name__ == "__main__":
    main()
