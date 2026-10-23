"""重新標註後，沿用原本的切分，只換掉多邊形。

放在 koi/setA/final/scripts/ 執行。Table 1 的五條 pipeline 都不是自己切資料，而是
讀 final/annotations/ 裡同一組檔案：

    instances_all.json        訓練用全部 93 張（第二階段 crop 由它產生）
    fold{k}_train.json        第 k 折的訓練影像（含 30 張只當訓練的影像）
    fold{k}_val.json          第 k 折的驗證影像（63 張分成五份）
    holdout.json              測試集 18 張，影像在 final/holdout/

重標後若重新切分，五個模型雖然彼此仍然公平，但跟原本的實驗就不是同一組資料，
論文裡任何引用舊數字的地方都會對不上。所以本腳本**不重新切分**：每個檔案的影像
清單原封不動，只把每張影像的標註換成新匯出的版本。

新匯出可以是一個或多個 Roboflow 資料夾（含 _annotations.coco.json），例如
train/valid/test 三個子資料夾一起給；影像以原始檔名對應（Roboflow 的 file_name
是 hash，原始檔名在 images[].extra.name，與 prep_coco.py 的處理相同）。

檢查項目（任何一項不通過就不寫檔）：
    - 舊切分裡的每一張影像都要在新匯出中找得到
    - 影像尺寸必須與舊檔相同（Roboflow 若做了 resize/auto-orient，座標會對不上）
    - 每個標註必須是單一多邊形

用法（在 koi/setA/final 底下）：
    python3 scripts/relabel_keep_split.py ~/Downloads/relabel/train ~/Downloads/relabel/valid ~/Downloads/relabel/test
    python3 scripts/relabel_keep_split.py ... --write     # 確認無誤後才寫入（舊檔自動備份）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if ROOT.name != "final":
    sys.exit(f"⚠ 這支腳本必須放在 koi/setA/final/scripts/ 執行，目前的根目錄是 {ROOT}。")
ANN = ROOT / "annotations"
IGNORE_LABELS = {"unclear", "1p", "partial", "tooth_partial"}


def polygon_area(seg):
    x, y = seg[0::2], seg[1::2]
    n = len(x)
    return abs(sum(x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i] for i in range(n))) / 2


def load_export(path: Path) -> dict[str, dict]:
    """回傳 {原始檔名: {"width", "height", "anns": [...], "src": 匯出資料夾}}。"""
    js = path / "_annotations.coco.json" if path.is_dir() else path
    data = json.loads(js.read_text(encoding="utf-8"))
    label_of = {c["id"]: c["name"] for c in data["categories"]}
    tooth_ids = {a["category_id"] for a in data["annotations"]
                 if label_of.get(a["category_id"], "").lower() not in IGNORE_LABELS}
    if len(tooth_ids) > 1:
        raise SystemExit(f"⚠ {js}：牙齒類別不只一個 {sorted(label_of[i] for i in tooth_ids)}，請確認")
    out: dict[str, dict] = {}
    by_id = {}
    for im in data["images"]:
        name = im.get("extra", {}).get("name") or im["file_name"]
        if name in out:
            raise SystemExit(f"⚠ {js}：影像 {name} 出現兩次")
        out[name] = {"width": im["width"], "height": im["height"], "anns": [], "src": str(js.parent)}
        by_id[im["id"]] = name
    for a in data["annotations"]:
        if len(a["segmentation"]) != 1:
            raise SystemExit(f"⚠ {by_id[a['image_id']]} 的標註 {a['id']} 不是單一多邊形，請在 Roboflow 修正")
        ignore = label_of.get(a["category_id"], "").lower() in IGNORE_LABELS
        out[by_id[a["image_id"]]]["anns"].append({
            "bbox": [round(v, 2) for v in a["bbox"]],
            "segmentation": a["segmentation"],
            "area": round(polygon_area(a["segmentation"][0]), 2),
            "iscrowd": 1 if ignore else 0,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", nargs="+", type=Path, help="Roboflow 匯出資料夾（或 _annotations.coco.json）")
    ap.add_argument("--write", action="store_true", help="確認無誤後加上這個才會寫檔")
    args = ap.parse_args()

    new: dict[str, dict] = {}
    for p in args.exports:
        for name, v in load_export(p.expanduser()).items():
            if name in new:
                sys.exit(f"⚠ 影像 {name} 同時出現在 {new[name]['src']} 與 {v['src']}")
            new[name] = v
    print(f"新匯出：{len(new)} 張影像，{sum(len(v['anns']) for v in new.values())} 個標註")

    files = ["instances_all.json"] + [f"fold{k}_{s}.json" for k in range(5) for s in ("train", "val")] \
        + ["holdout.json"]
    old = {f: json.loads((ANN / f).read_text(encoding="utf-8")) for f in files}

    # 1. 檢查：舊切分的每一張都要找得到、尺寸要相同
    used, missing, size_bad = set(), [], []
    for f, d in old.items():
        for im in d["images"]:
            n = im["file_name"]
            used.add(n)
            if n not in new:
                missing.append((f, n))
            elif (new[n]["width"], new[n]["height"]) != (im["width"], im["height"]):
                size_bad.append((n, (im["width"], im["height"]), (new[n]["width"], new[n]["height"])))
    if missing or size_bad:
        for f, n in sorted(set(missing)):
            print(f"  ✗ 新匯出找不到 {n}（{f} 需要）")
        for n, a, b in sorted(set(size_bad)):
            print(f"  ✗ {n} 尺寸不同：舊 {a[0]}×{a[1]}，新 {b[0]}×{b[1]}（匯出時請關掉 resize / auto-orient）")
        sys.exit("⚠ 檢查未通過，沒有寫入任何檔案。")
    extra = sorted(set(new) - used)
    if extra:
        print(f"  ※ 新匯出多了 {len(extra)} 張不在原切分裡的影像，不會被使用：{', '.join(extra[:10])}"
              f"{' …' if len(extra) > 10 else ''}")

    # 2. 組新檔：影像清單與 image id 照舊；標註 id 依檔名排序統一編號，各檔一致
    ann_id, next_id = {}, 1
    for n in sorted(used):
        ann_id[n] = list(range(next_id, next_id + len(new[n]["anns"])))
        next_id += len(new[n]["anns"])
    out = {}
    print(f"\n  {'檔案':<22}{'影像':>6}{'牙（舊）':>10}{'牙（新）':>10}{'ignore':>8}")
    for f, d in old.items():
        anns, cat = [], d["categories"][0]["id"]
        for im in d["images"]:
            for i, a in zip(ann_id[im["file_name"]], new[im["file_name"]]["anns"]):
                anns.append({"id": i, "image_id": im["id"], "category_id": cat, **a})
        out[f] = {**{k: v for k, v in d.items() if k != "annotations"}, "annotations": anns}
        n_old = sum(1 for a in d["annotations"] if not a.get("iscrowd"))
        n_new = sum(1 for a in anns if not a["iscrowd"])
        n_ign = sum(1 for a in anns if a["iscrowd"])
        print(f"  {f:<22}{len(d['images']):>6}{n_old:>10}{n_new:>10}{n_ign:>8}")

    # 3. 切分沒有變的自我檢查：每折 train ∪ val = instances_all、val 兩兩不重疊、holdout 不在訓練裡
    all_tr = {im["file_name"] for im in old["instances_all.json"]["images"]}
    vals = [{im["file_name"] for im in old[f"fold{k}_val.json"]["images"]} for k in range(5)]
    hold = {im["file_name"] for im in old["holdout.json"]["images"]}
    warn = []
    for k in range(5):
        tr = {im["file_name"] for im in old[f"fold{k}_train.json"]["images"]}
        if tr | vals[k] != all_tr or tr & vals[k]:
            warn.append(f"fold{k} 的 train ∪ val 與 instances_all 不一致")
    if sum(map(len, vals)) != len(set().union(*vals)):
        warn.append("各折 val 有重疊")
    if hold & all_tr:
        warn.append("holdout 影像出現在 instances_all")
    for w in warn:
        print(f"  ⚠ {w}（這是舊切分本身的狀態，本腳本不會改動它；請確認是否原本就如此）")
    only_train = all_tr - set().union(*vals)
    print(f"\n  切分{'檢查有警告' if warn else '檢查通過'}：訓練 {len(all_tr)} 張（其中 {len(only_train)} 張只當訓練），"
          f"五折 val {[len(v) for v in vals]}，holdout {len(hold)} 張")

    if not args.write:
        print("\n（試跑，未寫檔。確認上表無誤後加上 --write）")
        return
    bak = ROOT / f"annotations_before_relabel_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(ANN, bak)
    for f, d in out.items():
        (ANN / f).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print(f"\n已寫入 {ANN}/，舊檔備份在 {bak}/")
    print("接著必須重建由標註衍生的檔案：python3 scripts/make_crops_obb.py　與　python3 scripts/make_yolo.py")


if __name__ == "__main__":
    main()
