"""把 Roboflow 匯出的 COCO 整理成可直接訓練的格式，並切出 fold。

Roboflow 的匯出有三處要修：

    area          填的是 bbox 面積而非多邊形面積（實測差 26~47%）。pycocotools
                  用 area 分 small/medium/large 算 AP，有些訓練腳本也用它濾小
                  物件，不修會讓評估數字失真。
    categories    多一個 id 0 的 dummy（supercategory "none"），實際標註全在
                  id 1。不清掉會多出一個永遠沒有樣本的類別。
    file_name     被改成 hash，原始檔名藏在 images[].extra.name。切 fold、對回
                  原圖都要用原始檔名，否則無法跟 PA 來源對應。

bbox 本身沒問題（實測與多邊形外接矩形最大差 0.01 px），不需要重算。

未標註的牙
----------
收錄準則是「只標輪廓清楚的牙」，被跳過的牙若留白會被當成背景，等於教模型
不要偵測那種牙。補標的 unclear 多邊形在此轉成 iscrowd=1——Detectron2 與
torchvision 的 COCO 評估都把 iscrowd 當 ignore 區，不計 FP 也不計 FN。
目前標註檔裡還沒有 unclear，這段會自動跳過，補標後不必改程式。

輸出分別放到 koi/images/（原圖）與 koi/annotations/（COCO 與 fold）。

用法：
    py koi/scripts/prep_coco.py ~/Downloads/CEJ/train
    py koi/scripts/prep_coco.py ~/Downloads/CEJ/train --folds 5
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "images"
ANN = ROOT / "annotations"
TOOTH = "tooth"
IGNORE_LABELS = {"unclear", "1p", "partial", "tooth_partial"}


def polygon_area(seg: list[float]) -> float:
    """鞋帶公式。COCO 的 segmentation 是 [x0,y0,x1,y1,...] 攤平的形式。"""
    x, y = seg[0::2], seg[1::2]
    n = len(x)
    return abs(sum(x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i] for i in range(n))) / 2


def clean(src_json: Path) -> dict:
    data = json.loads(src_json.read_text(encoding="utf-8"))

    # Roboflow 的 dummy category（id 0、supercategory "none"）沒有任何標註，
    # 只留真正被用到的那一個，並統一改名為 tooth——原本叫 CEJ 是專案名，
    # 但標的是整顆牙，留著只會在後面轉檔與寫論文時造成混淆。
    used = {a["category_id"] for a in data["annotations"]}
    if len(used) != 1:
        raise ValueError(f"預期只有一個類別，實際用到 {sorted(used)}")
    old_id = used.pop()

    images = []
    for im in data["images"]:
        images.append(
            {
                "id": im["id"],
                "file_name": im.get("extra", {}).get("name") or im["file_name"],
                "width": im["width"],
                "height": im["height"],
                "_src": im["file_name"],
            }
        )

    label_of = {c["id"]: c["name"] for c in data["categories"]}
    anns, n_ignore = [], 0
    for a in data["annotations"]:
        if len(a["segmentation"]) != 1:
            raise ValueError(f"標註 {a['id']} 不是單一多邊形，需要人工確認")
        ignore = label_of.get(a["category_id"], "").lower() in IGNORE_LABELS
        if a["category_id"] != old_id and not ignore:
            raise ValueError(f"標註 {a['id']} 的類別非預期")
        n_ignore += ignore
        anns.append(
            {
                "id": a["id"],
                "image_id": a["image_id"],
                "category_id": 1,
                "bbox": [round(v, 2) for v in a["bbox"]],
                "segmentation": a["segmentation"],
                "area": round(polygon_area(a["segmentation"][0]), 2),
                "iscrowd": 1 if ignore else 0,
            }
        )

    return {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": [{"id": 1, "name": TOOTH, "supercategory": TOOTH}],
        "images": images,
        "annotations": anns,
        "_n_ignore": n_ignore,
    }


def make_folds(images: list[dict], k: int, seed: int) -> list[list[str]]:
    """以「影像」為單位切 fold，並依影像尺寸分層。

    絕不能以牙齒為單位切：同一張片子的牙如果分散到 train 與 val，模型看到的
    是同一個病人、同一次曝光、同一片骨小樑紋理，等於資料洩漏。

    依尺寸分層是因為這批資料混了兩種解析度，不分層的話某個 fold 可能整組都
    是同一種尺寸，fold 間的分數會被解析度而非方法差異主導。
    """
    import random

    rng = random.Random(seed)
    by_size: dict[tuple[int, int], list[str]] = defaultdict(list)
    for im in images:
        by_size[(im["width"], im["height"])].append(im["file_name"])

    folds: list[list[str]] = [[] for _ in range(k)]
    cursor = 0
    for size in sorted(by_size, key=lambda s: -len(by_size[s])):
        names = sorted(by_size[size])
        rng.shuffle(names)
        for name in names:
            folds[cursor % k].append(name)
            cursor += 1
    return [sorted(f) for f in folds]


def subset(data: dict, names: set[str]) -> dict:
    imgs = [im for im in data["images"] if im["file_name"] in names]
    ids = {im["id"] for im in imgs}
    return {
        "info": data["info"],
        "licenses": data["licenses"],
        "categories": data["categories"],
        "images": [{k: v for k, v in im.items() if not k.startswith("_")} for im in imgs],
        "annotations": [a for a in data["annotations"] if a["image_id"] in ids],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="Roboflow 匯出的資料夾（含 _annotations.coco.json）")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = clean(args.src / "_annotations.coco.json")
    for d in (IMAGES, ANN):
        d.mkdir(parents=True, exist_ok=True)
    for im in data["images"]:
        shutil.copy2(args.src / im["_src"], IMAGES / im["file_name"])

    all_names = {im["file_name"] for im in data["images"]}
    (ANN / "instances_all.json").write_text(
        json.dumps(subset(data, all_names), ensure_ascii=False), "utf-8"
    )

    folds = make_folds(data["images"], args.folds, args.seed)
    (ANN / "splits.json").write_text(
        json.dumps({f"fold{i}": f for i, f in enumerate(folds)}, ensure_ascii=False, indent=2), "utf-8"
    )
    for i in range(args.folds):
        val = set(folds[i])
        train = {n for j, f in enumerate(folds) if j != i for n in f}
        (ANN / f"fold{i}_val.json").write_text(json.dumps(subset(data, val), ensure_ascii=False), "utf-8")
        (ANN / f"fold{i}_train.json").write_text(json.dumps(subset(data, train), ensure_ascii=False), "utf-8")

    per = Counter(a["image_id"] for a in data["annotations"] if not a["iscrowd"])
    print(f"影像 {len(data['images'])} 張，牙齒標註 {sum(per.values())} 個"
          f"（ignore 區 {data['_n_ignore']} 個）")
    print(f"每張 {min(per.values())}~{max(per.values())} 顆，平均 {sum(per.values()) / len(per):.2f}")
    print(f"{args.folds}-fold（以影像為單位、依尺寸分層）：{[len(f) for f in folds]}")
    print(f"輸出 → {IMAGES}/ 與 {ANN}/")


if __name__ == "__main__":
    main()
