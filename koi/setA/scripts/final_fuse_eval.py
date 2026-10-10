"""在 final/ 的 holdout 上評估融合法，輸出與 Table 1 同格式的逐顆 CSV。

放在 koi/setA/final/scripts/ 執行。能借 final 原程式的地方一律借用：
    Mask R-CNN 載入　 final 版 eval_holdout_all.maskrcnn()
    HRNet 推論　　　  final 版 eval_seg2_holdout.predict()（含它的 TTA）
    Mask R-CNN TTA　  final 版 tta.predict_tta()
    pad、配對、指標、後處理　final 版的 PAD、metrics.match、clean_mask、gt_mask

**先跑 --check。** 它用同一段程式碼、原本的權重，重算 Table 1 的 ① 與 ④，再逐顆牙
和既有的 CSV 比對。全部吻合，才代表這支腳本的推論路徑與 Table 1 完全相同，
新方法的數字才能放進同一張表；不吻合會列出哪幾顆、差多少。

若 final/checkpoints_obb_base/ 存在（用 final_jit_train.py --no-jitter 重訓的 ④），另外輸出：
    FUS_OBBbase       重訓的 ④（與擾動版同一支腳本、同一組參數）
    FUS_fuse_base     融合，但 HRNet 用不擾動的重訓 ④（消融：融合是否需要擾動）

新方法輸出四組（檔名都以 hold5_FUS_ 開頭，不覆蓋既有結果）：
    FUS_OBBjit        Mask R-CNN → OBB → 框擾動版 HRNet（單獨，消融用）
    FUS_fuse          clean_mask(0.5·P_HRNet + 0.5·P_MaskRCNN > 0.5)，無 TTA，與 Table 1 同規則
    FUS_MaskRCNN_tta  Mask R-CNN + 翻轉 TTA（給 TTA 版融合當公平對照）
    FUS_fuse_tta      融合，兩個模型都做翻轉 TTA
融合權重固定 0.5，未在任何資料上調整。

用法（在 koi/setA/final 底下）：
    python3 scripts/final_fuse_eval.py --check <①的hold5名稱> <④的hold5名稱>
    python3 scripts/final_fuse_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_holdout_all as E  # noqa: E402  final 版
from eval_seg2_holdout import HOLD, gt_mask, predict  # noqa: E402
from make_crops_obb import obb_of, warp_of  # noqa: E402
from metrics import FIELDS, match  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, ROOT  # noqa: E402
from train_seg2 import build_seg2, split_tag  # noqa: E402
from tta import predict_tta  # noqa: E402

PAD = getattr(E, "PAD", 0.2)
TAG = getattr(E, "TAG", "unet_tu-hrnet_w32")
EVAL = ROOT / "eval"


def load_jit(ckdir, fold):
    arch, enc = split_tag(TAG)
    m = build_seg2(arch, enc, pretrained=False)
    m.load_state_dict(torch.load(ROOT / ckdir / "seg2" / TAG / f"fold{fold}.pt",
                                 map_location="cpu", weights_only=False)["model"])
    return m.eval()


@torch.no_grad()
def mr_probs(fold, gray, thr, use_tta):
    model = E.maskrcnn(fold)
    if use_tta:
        prob, _, sc = predict_tta(model, gray, thr)
    else:
        t = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1)
        out = model([t])[0]
        keep = out["scores"].numpy() >= thr
        prob, sc = out["masks"].numpy()[keep, 0], out["scores"].numpy()[keep]
    return np.asarray(prob, np.float32).reshape(-1, *gray.shape), np.asarray(sc)


def hr_prob(seg, gray, box, use_tta):
    """斜框裁切 → final 的 predict() → 轉回原圖的機率圖。"""
    h, w = gray.shape
    M, cw, ch = warp_of(*box, PAD)
    prob = predict([seg], cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR), use_tta)
    small = cv2.resize(prob, (cw, ch), interpolation=cv2.INTER_LINEAR)
    return cv2.warpAffine(small, cv2.invertAffineTransform(M), (w, h), flags=cv2.INTER_LINEAR)


def run_variants(fold, seg, gray, thr, use_tta, names):
    """回傳 {名稱: (遮罩, 分數)}；names 決定要算哪些。"""
    hw = gray.shape
    prob, sc = mr_probs(fold, gray, thr, use_tta)
    m_bin, e_bin, f_bin, keep = [], [], [], []
    for i, p in enumerate(prob):
        cm = np.asarray(clean_mask(p > 0.5), bool)
        m_bin.append(cm)
        if not cm.any():
            continue
        pe = hr_prob(seg, gray, obb_of(cm.astype(np.uint8)), use_tta)
        e_bin.append(np.asarray(clean_mask(pe > 0.5), bool))
        f_bin.append(np.asarray(clean_mask(0.5 * pe + 0.5 * p > 0.5), bool))
        keep.append(i)

    def arr(x):
        return np.array(x, bool).reshape(-1, *hw)
    out = {"M": (arr(m_bin), sc), "E": (arr(e_bin), sc[keep]), "F": (arr(f_bin), sc[keep])}
    return {name: out[k] for k, name in names.items()}


def holdout():
    coco = json.loads((ANN / "holdout.json").read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    per: dict[int, list] = {}
    for a in coco["annotations"]:
        if not a.get("iscrowd"):
            per.setdefault(a["image_id"], []).append(a)
    return imgs, per


def evaluate(jobs, thr):
    """jobs: [(use_tta, {"M"/"E"/"F": 輸出名稱}, 權重載入函式)]。寫出 hold5_<名稱>_fold*.csv。"""
    imgs, per = holdout()
    print(f"holdout（{ANN / 'holdout.json'}）：{len(per)} 張、"
          f"{sum(map(len, per.values()))} 顆標註牙　門檻 {thr}　pad {PAD}", flush=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    t0, done, total = time.time(), 0, 5 * len(per)
    for fold in range(5):
        rows: dict[str, list] = {}
        segs = [loader(fold) for _, _, loader in jobs]
        for iid, anns in sorted(per.items()):
            im = imgs[iid]
            gray = cv2.imread(str(HOLD / im["file_name"]), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"  ⚠ 讀不到 {HOLD / im['file_name']}")
                continue
            h, w = gray.shape
            gt = np.stack([gt_mask(a, h, w) for a in anns])
            for (use_tta, names, _), seg in zip(jobs, segs):
                for name, (pred, sc) in run_variants(fold, seg, gray, thr, use_tta, names).items():
                    r = match(pred, sc, gt, im["file_name"])[0]
                    for x in r:
                        x["gt_idx"] = f"{x['gt_idx']}"
                    rows.setdefault(name, []).extend(r)
            done += 1
            el = time.time() - t0
            print(f"  fold {fold}　{done}/{total}　{im['file_name']}　已花 {el:.0f}s　"
                  f"預估剩 {el / done * (total - done):.0f}s", flush=True)
        for name, rs in rows.items():
            with (EVAL / f"hold5_{name}_fold{fold}.csv").open("w", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(fh, fieldnames=FIELDS)
                wr.writeheader()
                wr.writerows(rs)


def compare(mine, ref):
    """逐折、逐顆比對兩組 CSV；回傳 (比對筆數, 不吻合清單)。"""
    bad, n = [], 0
    for k in range(5):
        a = {(r["image"], r["gt_idx"], r["kind"]): r for r in
             csv.DictReader((EVAL / f"hold5_{mine}_fold{k}.csv").open(encoding="utf-8"))}
        b = {(r["image"], r["gt_idx"], r["kind"]): r for r in
             csv.DictReader((EVAL / f"hold5_{ref}_fold{k}.csv").open(encoding="utf-8"))}
        tp_a = {x for x in a if x[2] == "TP"}
        tp_b = {x for x in b if x[2] == "TP"}
        for x in tp_a ^ tp_b:
            bad.append((k, x, "只在一邊是 TP"))
        for x in tp_a & tp_b:
            n += 1
            for m in ("dice", "hd95", "assd"):
                d = abs(float(a[x][m]) - float(b[x][m]))
                if d > 1e-3 * max(1.0, abs(float(b[x][m]))):
                    bad.append((k, x, f"{m} {a[x][m]} vs {b[x][m]}"))
    return n, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", nargs=2, metavar=("REF_M", "REF_E"),
                    help="Table 1 中 ① 與 ④ 的 hold5_ 名稱；只做重現檢查")
    ap.add_argument("--ckpt-dir", default="checkpoints_obb_jit")
    ap.add_argument("--base-ckpt-dir", default="checkpoints_obb_base",
                    help="--no-jitter 重訓的 ④；目錄不存在就略過")
    ap.add_argument("--thr", type=float, default=0.35, help="偵測門檻，須與 Table 1 相同")
    args = ap.parse_args()

    if args.check:
        ref_m, ref_e = args.check
        evaluate([(False, {"M": "CHK_MaskRCNN", "E": "CHK_OBB"}, lambda f: E.seg2(f, True))],
                 args.thr)
        ok = True
        for mine, ref in (("CHK_MaskRCNN", ref_m), ("CHK_OBB", ref_e)):
            n, bad = compare(mine, ref)
            print(f"\n{mine} vs {ref}：比對 {n} 筆 TP，不吻合 {len(bad)}")
            for b in bad[:10]:
                print(f"    fold {b[0]}　{b[1][0]} #{b[1][1]}　{b[2]}")
            ok &= not bad
        print("\n" + ("✅ 完全吻合：推論路徑與 Table 1 相同，可以跑新方法。" if ok else
                      "❌ 不吻合：推論路徑與 Table 1 不同，新方法的數字不可放進同一張表。"
                      "請把上面的輸出貼回來。"))
        return

    jit = lambda f: load_jit(args.ckpt_dir, f)   # noqa: E731
    jobs = [(False, {"E": "FUS_OBBjit", "F": "FUS_fuse"}, jit),
            (True, {"M": "FUS_MaskRCNN_tta", "F": "FUS_fuse_tta"}, jit)]
    if (ROOT / args.base_ckpt_dir / "seg2" / TAG / "fold0.pt").exists():
        base = lambda f: load_jit(args.base_ckpt_dir, f)   # noqa: E731
        jobs.append((False, {"E": "FUS_OBBbase", "F": "FUS_fuse_base"}, base))
        print(f"一併評估重訓的 ④：{args.base_ckpt_dir}")
    evaluate(jobs, args.thr)
    print(f"\n完成 → {EVAL}/hold5_FUS_*_fold{{0..4}}.csv")


if __name__ == "__main__":
    main()
