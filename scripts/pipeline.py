"""完整流程：一張根尖片進去，CRR 等指標出來，全程不需要人工標註。

    原始影像 -> 種子 -> 模板粗定位 -> 沿法線精修 -> 遮罩 -> 測量

前四步在 `find_seeds.py`、`align_template.py`、`refine_contour.py`，最後一步是
既有的 `measure.py`。這支程式只負責把它們串起來，本身不含演算法。

串接點是**遮罩**：`measure()` 本來就直接吃 (影像, 遮罩)，labelme 只用在它的
main() 讀檔而已。所以只要把精修後的輪廓光柵化成遮罩，整條路就通了——這也是
當初決定「一切只靠牙齒遮罩」的用意，換掉遮罩的來源不影響下游。

品質閘門
--------
配準的兩個輸出本身就能篩掉失敗的擺放，不必另外設計判準：

- **尺度**：實測失敗案例的尺度一律遠小於種子給的估計（<0.55），因為模板縮到
  某個局部亮區上去了
- **得分**：t 統計量偏低代表內外對比撐不起來

門檻目前是依 25 張測試片的觀察暫定的，尚未在有標註的資料上校準。

用法：
    py scripts/pipeline.py shape_prior_seg_test --only 13 41
    py scripts/pipeline.py shape_prior_seg_test --csv auto.csv -o pipeline.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.align_template import align_image  # noqa: E402
from scripts.labelme_io import make_figure, save_or_show  # noqa: E402
from scripts.measure import draw, measure  # noqa: E402
from scripts.refine_contour import refine  # noqa: E402
from scripts.shape_template import build_template  # noqa: E402

MIN_SCALE = 0.55  # 低於此值代表模板縮到局部亮區上
MIN_SCORE = 15.0  # t 統計量低於此值代表內外對比撐不起來


def contour_to_mask(contour: np.ndarray, shape: tuple) -> np.ndarray:
    """把輪廓光柵化成 uint8 遮罩，供 measure() 使用。"""
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(contour).astype(np.int32)], 1)
    return mask


def segment(img: np.ndarray, template, top: int) -> list[tuple]:
    """分割出牙齒，回傳 [(遮罩, 得分, 位移量), ...]，已過品質閘門。"""
    placements = align_image(img, template, 30.0, 170.0, 140.0)
    placements = [p for p in placements
                  if p.score >= MIN_SCORE and min(p.scale_x, p.scale_y) >= MIN_SCALE]
    placements = sorted(placements, key=lambda p: -p.score)[:top]

    results = []
    for placement in placements:
        refined, shift = refine(img, template.place(*placement.params))
        results.append((contour_to_mask(refined, img.shape), placement.score, shift))
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", help="含有 .jpg 的資料夾")
    parser.add_argument("--only", nargs="+", help="只處理這幾張")
    parser.add_argument("--template", default="labeled_PA/13.json", help="標準形狀的來源")
    parser.add_argument("--top", type=int, default=1, help="每張最多測量幾顆牙，預設 1")
    parser.add_argument("--csv", help="把結果寫成 CSV")
    parser.add_argument("-o", "--out", help="圖表存檔路徑，不給則開視窗")
    # 下游測量的參數，維持與 measure.py 相同的預設值
    parser.add_argument("--drop-apical", type=float, default=0.25)
    parser.add_argument("--cut-at", type=float, default=1.0 / 3.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--surface-fraction", type=float, default=0.10)
    parser.add_argument("--inner-fraction", type=float, default=0.03)
    parser.add_argument("--outer-fraction", type=float, default=0.15)
    parser.add_argument("--root-fraction", type=float, default=2.0 / 3.0)
    args = parser.parse_args()

    folder = Path(args.folder)
    files = sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if args.only:
        files = [f for f in files if f.stem in set(args.only)]
    if not files:
        raise SystemExit(f"在 {folder} 找不到影像")

    template = build_template(args.template)

    # 先全部算完再畫，因為欄數要等篩選過後才知道
    jobs, rows = [], []
    for path in files:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        found = segment(img, template, args.top)
        if not found:
            print(f"{path.stem}：沒有通過品質閘門的擺放")
            continue

        for index, (mask, score, shift) in enumerate(found):
            name = path.stem if len(found) == 1 else f"{path.stem}-{index + 1}"
            try:
                result = measure(name, img, mask, args.drop_apical, args.cut_at,
                                 args.threshold, args.surface_fraction,
                                 args.inner_fraction, args.outer_fraction,
                                 args.root_fraction)
            except (ValueError, IndexError) as exc:
                print(f"{name}：測量失敗 - {exc}")
                continue

            print(f"{name}　得分 {score:5.1f}　位移中位數 {np.median(shift):4.1f} px"
                  f"　CRR {result.crr:.3f}　ABLR {result.ablr:.3f}"
                  f"　MaxBLR {result.max_blr:.3f}　B-CRR {result.b_crr:.3f}")

            jobs.append((img, mask, result))
            rows.append({
                "tooth": name,
                "score": round(score, 1),
                "shift_median": round(float(np.median(shift)), 1),
                "CRR": round(result.crr, 4),
                "ABLR": round(result.ablr, 4),
                "MaxBLR": round(result.max_blr, 4),
                "B_CRR": round(result.b_crr, 4),
            })

    if not jobs:
        raise SystemExit("沒有任何影像測量成功")

    cols = min(5, len(jobs))
    rows_n = (len(jobs) + cols - 1) // cols
    plt, fig, axes = make_figure(rows_n, cols, (4.4 * cols, 8.0 * rows_n), args.out)
    for index, (img, mask, result) in enumerate(jobs):
        draw(axes[index // cols][index % cols], img, mask, result)
    for blank in range(len(jobs), rows_n * cols):
        axes[blank // cols][blank % cols].axis("off")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n已寫入 {len(rows)} 筆 -> {args.csv}")

    save_or_show(plt, args.out, dpi=120)


if __name__ == "__main__":
    main()
