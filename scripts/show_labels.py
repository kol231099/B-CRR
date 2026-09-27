"""把 labelme 標註疊在原始根尖片上檢視。

本專案的分割標籤約定：
    1 = 整顆牙齒（牙冠 + 牙根）
    2 = 琺瑯質
    3 = 齒槽骨

用法：
    py scripts/show_labels.py labeled_PA/7.json
    py scripts/show_labels.py labeled_PA/7.json --fill
    py scripts/show_labels.py labeled_PA/7.json --fill --only 2
    py scripts/show_labels.py labeled_PA/7.json --crop-y 0 520
    py scripts/show_labels.py labeled_PA/7.json -o out.png     # 存檔，不開視窗
    py scripts/show_labels.py labeled_PA                        # 資料夾內所有 .json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LABEL_NAMES = {"1": "整顆牙齒", "2": "琺瑯質", "3": "齒槽骨"}
# 各標籤的顏色（RGB）
LABEL_COLORS = {"1": (0, 255, 0), "2": (255, 60, 60), "3": (0, 200, 255)}
DEFAULT_COLOR = (255, 255, 0)


def load_annotation(json_path: Path):
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    img_path = json_path.parent / data["imagePath"]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"讀不到 {json_path} 所指向的影像：{img_path}")
    return data, img


def summarise(data: dict) -> list[str]:
    """每個多邊形輸出一行：標籤、頂點數、外接框、像素面積。"""
    h, w = data["imageHeight"], data["imageWidth"]
    lines = [f"{data['imagePath']}  ({w}x{h})"]
    per_label_area: dict[str, int] = {}
    per_label_count: dict[str, int] = {}
    for i, shape in enumerate(data["shapes"]):
        label = shape["label"]
        pts = np.array(shape["points"], dtype=np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        area = int(mask.sum())
        per_label_area[label] = per_label_area.get(label, 0) + area
        per_label_count[label] = per_label_count.get(label, 0) + 1
        x0, y0, x1, y1 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
        name = LABEL_NAMES.get(label, "?")
        lines.append(
            f"  多邊形{i}  標籤={label}（{name}）  頂點數={len(pts)}  "
            f"外接框 x[{x0},{x1}] y[{y0},{y1}]  面積={area}"
        )
    for label in sorted(per_label_count):
        lines.append(
            f"  -> 標籤 {label}（{LABEL_NAMES.get(label, '?')}）："
            f"{per_label_count[label]} 個多邊形，總面積={per_label_area[label]}"
        )
    return lines


def render(data: dict, img: np.ndarray, fill: bool, only: set[str] | None) -> np.ndarray:
    """回傳疊好標註的 RGB 影像。"""
    h, w = data["imageHeight"], data["imageWidth"]
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(float)

    for shape in data["shapes"]:
        label = shape["label"]
        if only and label not in only:
            continue
        color = np.array(LABEL_COLORS.get(label, DEFAULT_COLOR), dtype=float)
        pts = np.array(shape["points"], dtype=np.int32)
        if fill:
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            canvas[mask > 0] = canvas[mask > 0] * 0.55 + color * 0.45
        cv2.polylines(canvas, [pts], isClosed=True, color=tuple(color), thickness=2)

    return np.clip(canvas, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", help="labelme 的 .json 檔，或含有多個 .json 的資料夾")
    parser.add_argument("--fill", action="store_true", help="半透明填色（預設只畫輪廓）")
    parser.add_argument("--only", nargs="+", metavar="LABEL", help="只顯示指定標籤，例如 --only 2")
    parser.add_argument("--crop-y", nargs=2, type=int, metavar=("Y0", "Y1"), help="只顯示 [Y0, Y1) 這幾列")
    parser.add_argument("-o", "--out", help="存檔到此路徑，不開視窗")
    args = parser.parse_args()

    target = Path(args.target)
    json_files = sorted(target.glob("*.json")) if target.is_dir() else [target]
    if not json_files:
        raise SystemExit(f"在 {target} 找不到任何 .json 標註檔")

    only = set(args.only) if args.only else None
    panels, titles = [], []

    for jf in json_files:
        data, img = load_annotation(jf)
        print("\n".join(summarise(data)))
        overlay = render(data, img, args.fill, only)
        if args.crop_y:
            y0, y1 = args.crop_y
            img, overlay = img[y0:y1], overlay[y0:y1]
        panels.extend([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), overlay])
        titles.extend([f"{jf.stem} 原圖", f"{jf.stem} 標註"])

    import matplotlib

    # 後端必須在 import pyplot 之前設定，而是否需要無視窗模式取決於
    # 有沒有給 -o，所以 matplotlib 要延後到這裡才 import。
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.labelme_io import setup_cjk_font

    setup_cjk_font(plt)

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 9))
    if len(panels) == 1:
        axes = [axes]
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], color=np.array(c) / 255, lw=2, label=f"{k} = {LABEL_NAMES.get(k, '?')}")
        for k, c in LABEL_COLORS.items()
        if not only or k in only
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9)
    plt.tight_layout(rect=(0, 0.04, 1, 1))

    if args.out:
        plt.savefig(args.out, dpi=120, bbox_inches="tight")
        print(f"\n已存檔 -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
