"""讀取 labelme 標註，並將其多邊形轉為像素資料。

本專案使用的標籤約定：
    "1" = 整顆牙齒（牙冠 + 牙根）
    "2" = 琺瑯質
    "3" = 齒槽骨
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

LABEL_TOOTH = "1"
LABEL_ENAMEL = "2"
LABEL_BONE = "3"
LABEL_NAMES = {LABEL_TOOTH: "整顆牙齒", LABEL_ENAMEL: "琺瑯質", LABEL_BONE: "齒槽骨"}

# 疊圖用的顏色（RGBA）。一律畫成固定透明度的色層，不要用逐點散布——散點的
# 視覺濃度會隨像素數與圖形縮放而變，同一份設定在大小不同的牙齒上會深淺不一。
MASK_RGBA = (0.30, 1.00, 0.40, 0.13)  # 牙齒遮罩
REGION_RGBA = (1.00, 0.25, 0.20, 0.22)  # 選取出來的區域（牙冠、帶狀區域等）


def overlay(ax, shape, points, rgba, crop=None):
    """把一群像素畫成固定透明度的色層。crop 為 (y0, y1, x0, x1) 時先裁切。"""
    import numpy as _np

    layer = _np.zeros((*shape, 4))
    layer[points[:, 1].astype(int), points[:, 0].astype(int)] = rgba
    if crop is not None:
        y0, y1, x0, x1 = crop
        layer = layer[y0:y1, x0:x1]
    ax.imshow(layer)


def collect_annotations(target: str | Path) -> list[Path]:
    """把命令列給的 target 解析成要處理的 .json 清單。

    輸入：一個 .json 檔的路徑，或一個含有多個 .json 的資料夾。
    輸出：排序後的 Path 清單；找不到任何檔案時直接中止程式。
    """
    target = Path(target)
    found = sorted(target.glob("*.json")) if target.is_dir() else [target]
    if not found:
        raise SystemExit(f"在 {target} 找不到任何 .json 標註檔")
    return found


def load_tooth(json_path: str | Path):
    """一次讀齊一顆牙需要的東西。

    輸入：labelme 的 .json 路徑。
    輸出：(標註資料, 灰階影像, 牙齒遮罩, 遮罩像素座標)。
    """
    data = load_annotation(json_path)
    mask = label_mask(data, LABEL_TOOTH)
    return data, load_image(data), mask, mask_points(mask)


def make_figure(rows: int, cols: int, size: tuple, out: str | None):
    """建立圖表並套用中文字型設定。

    out 非空時要切成無視窗後端，而**後端必須在 import pyplot 之前設定**，
    所以 matplotlib 的 import 得延後到這裡，不能放在檔案開頭。

    輸入：子圖列數、行數、figsize、輸出路徑（None 表示開視窗）。
    輸出：(pyplot 模組, fig, axes)；axes 一律是二維陣列。
    """
    import matplotlib

    if out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_cjk_font(plt)
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False)
    return plt, fig, axes


def save_or_show(plt, out: str | None, dpi: int = 120) -> None:
    """收尾：有給輸出路徑就存檔，否則開視窗。"""
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"已存檔 -> {out}")
    else:
        plt.show()


def load_annotation(json_path: str | Path) -> dict:
    """讀取一個 labelme 的 .json 標註檔。"""
    json_path = Path(json_path)
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    data["_json_path"] = json_path
    return data


def load_image(data: dict) -> np.ndarray:
    """讀取該標註所對應的灰階根尖片。"""
    img_path = Path(data["_json_path"]).parent / data["imagePath"]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"無法讀取影像：{img_path}")
    return img


def label_mask(data: dict, label: str) -> np.ndarray:
    """回傳含有指定標籤所有多邊形的二值遮罩（uint8, 0/1）。

    同一標籤的多個多邊形（例如琺瑯質的左右兩條）會合併成同一張遮罩。
    """
    h, w = data["imageHeight"], data["imageWidth"]
    mask = np.zeros((h, w), np.uint8)
    found = False
    for shape in data["shapes"]:
        if shape["label"] != label:
            continue
        cv2.fillPoly(mask, [np.array(shape["points"], dtype=np.int32)], 1)
        found = True
    if not found:
        raise ValueError(
            f"{data['imagePath']} 中沒有標籤為 '{label}'（{LABEL_NAMES.get(label, '?')}）的多邊形"
        )
    return mask


def setup_cjk_font(plt) -> None:
    """讓 matplotlib 能正確顯示中文，否則圖上的中文會變成空心方框。

    依序嘗試幾個 Windows 內建的中文字型，最後保留 matplotlib 的預設
    字型作為後備。另外關閉 unicode_minus，因為多數中文字型缺少
    U+2212（真正的減號）字符，開著會讓負號也變成方框。
    """
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",  # 微軟正黑體（繁中）
        "Microsoft YaHei",  # 微軟雅黑（簡中）
        "MingLiU",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def mask_points(mask: np.ndarray) -> np.ndarray:
    """取出遮罩中所有前景像素，回傳 (N, 2) 的 (x, y) 座標陣列。

    此處採用填滿後的遮罩像素、而非多邊形頂點，是關鍵的設計選擇：
    標註者在曲率大處點得密、在平直處點得疏，所以頂點是對形狀的
    **有偏取樣**；遮罩像素則是對面積的**均勻取樣**，才符合 PCA 對
    輸入分布的假設。
    """
    ys, xs = np.nonzero(mask)
    return np.column_stack([xs, ys]).astype(float)
