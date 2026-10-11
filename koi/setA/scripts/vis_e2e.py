"""把 Mask R-CNN 與 Mask R-CNN→OBB→HRNet 的每一張結果畫出來，附逐顆牙指標，方便找爆掉的牙。

每張影像輸出一張圖（三格）：
    左　GT（綠）＋ Mask R-CNN（青）
    中　GT（綠）＋ 端到端（紅）＋ 斜框（黃）；FP 用洋紅
    右　端到端 HD95 最差那顆牙的放大圖
每顆牙旁標 `#編號 M:Dice/HD95  E:Dice/HD95`（M = Mask R-CNN，E = 端到端）。

另外輸出：
    index.html　 所有牙一列一列排好，預設依端到端 HD95 由差到好，點表頭可改排序，
                 點縮圖開大圖。
    teeth.csv　  同樣的逐顆牙指標。
圖檔名以名次開頭（001_、002_…），最爛的排最前面。

評估設定與 eval_e2e_obb.py 一致：門檻 0.35、pad 0.2、clean_mask、單模型、原圖座標。
--split oof：每張影像用「它所屬驗證折」的權重（五折 OOF）。
--split holdout：用 --fold 指定的那一折權重跑 holdout。holdout 只標了部分牙齒，
                 FP 不代表誤判（見 eval_holdout_all.py）。
輸出的圖含病患影像，只存在本機的 eval/ 底下（.gitignore 已排除 png）。

用法：
    python3 scripts/vis_e2e.py --split oof --ckpt-dir checkpoints_obb_jit
    python3 scripts/vis_e2e.py --split holdout --fold 0 --ckpt-dir checkpoints_obb_jit
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_seg2_holdout import HOLD, gt_mask  # noqa: E402
from make_crops_obb import obb_of, warp_of  # noqa: E402
from metrics import match  # noqa: E402
from postprocess import clean_mask  # noqa: E402
from train_maskrcnn import ANN, CKPT, ROOT, ToothDataset, build_model, collate  # noqa: E402
from train_seg2 import SIZE, build_seg2, split_tag  # noqa: E402

PAD = 0.2
GREEN, CYAN, RED, YELLOW, MAGENTA = (0, 200, 0), (255, 200, 0), (0, 0, 255), (0, 220, 255), (255, 0, 255)
METRICS = ("dice", "hd95", "assd", "iou")


# ---------------------------------------------------------------- 模型與推論

def load_mrcnn(fold):
    ck = torch.load(CKPT / "original" / f"maskrcnn_fold{fold}.pt", map_location="cpu",
                    weights_only=False)
    m = build_model(False, ck.get("mask_res", 28))
    m.load_state_dict(ck["model"])
    return m.eval()


def load_seg(ckdir, tag, fold, device):
    arch, enc = split_tag(tag)
    m = build_seg2(arch, enc, pretrained=False)
    m.load_state_dict(torch.load(ROOT / ckdir / "seg2" / tag / f"fold{fold}.pt",
                                 map_location="cpu", weights_only=False)["model"])
    return m.eval().to(device)


@torch.no_grad()
def run_image(mr, seg, gray, thr, device):
    """回傳 (Mask R-CNN 遮罩, 分數, 端到端遮罩, 端到端分數, 斜框)。"""
    hw = gray.shape
    t = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1)
    out = mr([t])[0]
    keep = out["scores"].numpy() >= thr
    mm = np.array([np.asarray(clean_mask(x), bool) for x in out["masks"].numpy()[keep, 0] > 0.5],
                  bool).reshape(-1, *hw)
    sc = out["scores"].numpy()[keep]

    ok = [i for i, m in enumerate(mm) if m.any()]
    boxes = [obb_of(mm[i].astype(np.uint8)) for i in ok]
    e2e = []
    if boxes:
        xs, geo = [], []
        for box in boxes:
            M, cw, ch = warp_of(*box, PAD)
            crop = cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR)
            xs.append(cv2.resize(crop, SIZE[::-1], interpolation=cv2.INTER_AREA))
            geo.append((M, cw, ch))
        x = torch.from_numpy(np.stack(xs)).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
        probs = torch.sigmoid(seg(x.to(device)))[:, 0].cpu().numpy()
        for p, (M, cw, ch) in zip(probs, geo):
            small = cv2.resize(p, (cw, ch), interpolation=cv2.INTER_LINEAR)
            back = cv2.warpAffine(small, cv2.invertAffineTransform(M), (hw[1], hw[0]),
                                  flags=cv2.INTER_LINEAR) > 0.5
            e2e.append(np.asarray(clean_mask(back), bool))
    e2e = np.array(e2e, bool).reshape(-1, *hw)
    return mm, sc, e2e, sc[ok], boxes


# ---------------------------------------------------------------- 資料

def iter_oof():
    for fold in range(5):
        ds = ToothDataset(ANN / f"fold{fold}_val.json", train=False, enhance="original")
        for imgs, targets in DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate):
            t = targets[0]
            gray = (imgs[0][0].numpy() * 255).astype(np.uint8)
            yield fold, t["_name"], gray, t["masks"].numpy().astype(bool)


def iter_holdout(fold):
    coco = json.loads((ANN / "holdout.json").read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    per: dict[int, list] = {}
    for a in coco["annotations"]:
        if not a.get("iscrowd"):
            per.setdefault(a["image_id"], []).append(a)
    for iid, anns in sorted(per.items()):
        im = imgs[iid]
        gray = cv2.imread(str(HOLD / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h, w = gray.shape
        yield fold, im["file_name"], gray, np.stack([gt_mask(a, h, w) for a in anns])


# ---------------------------------------------------------------- 繪圖

def outline(img, mask, color, th):
    cnts = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(img, cnts, -1, color, th, cv2.LINE_AA)


def label(img, text, xy, scale, th, color=(255, 255, 255)):
    x, y = int(xy[0]), int(xy[1])
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x = min(max(x, 0), img.shape[1] - tw - 2)
    y = min(max(y, tht + 2), img.shape[0] - 2)
    cv2.rectangle(img, (x - 2, y - tht - 3), (x + tw + 2, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def fmt(r):
    return "--" if r is None else f"{float(r['dice']):.3f}/{float(r['hd95']):.0f}"


def figure(gray, gt, mm, e2e, boxes, map_m, map_e, rows_m, rows_e, title):
    h, w = gray.shape
    th = max(1, round(max(h, w) / 500))
    sc_txt = max(0.4, max(h, w) / 1600)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    left, mid = base.copy(), base.copy()

    for g in gt:
        outline(left, g, GREEN, th)
        outline(mid, g, GREEN, th)
    inv_m = {gi: pi for pi, gi in map_m.items()}
    inv_e = {gi: pi for pi, gi in map_e.items()}
    for pi, m in enumerate(mm):
        outline(left, m, CYAN if pi in map_m else MAGENTA, th)
    for pi, m in enumerate(e2e):
        outline(mid, m, RED if pi in map_e else MAGENTA, th)
        if pi not in map_e:
            ys, xs = np.nonzero(m)
            if len(xs):
                label(mid, "FP", (xs.mean(), ys.mean()), sc_txt, th, MAGENTA)
    for box in boxes:
        M, cw, ch = warp_of(*box, PAD)
        corners = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float64)
        Mi = cv2.invertAffineTransform(M)
        pts = (corners @ Mi[:, :2].T + Mi[:, 2]).astype(np.int32)
        cv2.polylines(mid, [pts], True, YELLOW, max(1, th // 2), cv2.LINE_AA)

    by_m = {int(r["gt_idx"]): r for r in rows_m if r["kind"] == "TP"}
    by_e = {int(r["gt_idx"]): r for r in rows_e if r["kind"] == "TP"}
    for gi, g in enumerate(gt):
        ys, xs = np.nonzero(g)
        xy = (xs.min(), ys.mean())
        label(left, f"#{gi} M {fmt(by_m.get(gi))}", xy, sc_txt, th,
              (255, 255, 255) if gi in inv_m else MAGENTA)
        label(mid, f"#{gi} E {fmt(by_e.get(gi))}", xy, sc_txt, th,
              (255, 255, 255) if gi in inv_e else MAGENTA)

    # 右格：端到端 HD95 最差的那顆牙放大
    zoom = np.zeros_like(base)
    worst = max(by_e.items(), key=lambda kv: float(kv[1]["hd95"]), default=None)
    if worst is not None:
        gi = worst[0]
        ys, xs = np.nonzero(gt[gi] | e2e[inv_e[gi]])
        if gi in inv_m:
            ys2, xs2 = np.nonzero(mm[inv_m[gi]])
            ys, xs = np.r_[ys, ys2], np.r_[xs, xs2]
        p = int(0.15 * max(np.ptp(ys), np.ptp(xs))) + 5
        y0, y1 = max(ys.min() - p, 0), min(ys.max() + p, h)
        x0, x1 = max(xs.min() - p, 0), min(xs.max() + p, w)
        z = base[y0:y1, x0:x1].copy()
        zt = max(1, th // 2)
        outline(z, gt[gi][y0:y1, x0:x1], GREEN, zt)
        if gi in inv_m:
            outline(z, mm[inv_m[gi]][y0:y1, x0:x1], CYAN, zt)
        outline(z, e2e[inv_e[gi]][y0:y1, x0:x1], RED, zt)
        s = min(h / z.shape[0], w / z.shape[1])
        z = cv2.resize(z, (int(z.shape[1] * s), int(z.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
        zoom[:z.shape[0], :z.shape[1]] = z
        label(zoom, f"#{gi} worst E  M {fmt(by_m.get(gi))}  E {fmt(worst[1])}",
              (5, 5), sc_txt, th)

    fig = np.hstack([left, mid, zoom])
    bar = np.zeros((int(60 * sc_txt / 0.6) + 10, fig.shape[1], 3), np.uint8)
    label(bar, title, (8, bar.shape[0] - 12), sc_txt * 1.1, th)
    label(bar, "green GT | cyan MaskRCNN | red E2E | yellow OBB | magenta FP/FN   (Dice/HD95px)",
          (fig.shape[1] // 2, bar.shape[0] - 12), sc_txt * 0.9, th)
    fig = np.vstack([bar, fig])
    s = min(1.0, 2400 / fig.shape[1])
    return cv2.resize(fig, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 主程式

PAGE = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>E2E Tooth Review</title><style>
:root{{--bg:#fff;--fg:#111;--mut:#666;--line:#ddd;--bad:#fde2e1;--warn:#fff4d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--mut:#999;--line:#333;--bad:#4a1f1f;--warn:#45391a}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;margin:16px}}
table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right;white-space:nowrap}}
th{{cursor:pointer;position:sticky;top:0;background:var(--bg)}}
td.l,th.l{{text-align:left}} tr.bad{{background:var(--bad)}} tr.warn{{background:var(--warn)}}
img{{width:360px;display:block}} .mut{{color:var(--mut)}}
</style></head><body>
<h2>{title}</h2><p class="mut">{note}</p>
<table id="t"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>
<script>
document.querySelectorAll('#t th').forEach((th,i)=>th.onclick=()=>{{
 const tb=document.querySelector('#t tbody'),rows=[...tb.rows];
 const asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
 rows.sort((a,b)=>{{const x=a.cells[i].dataset.v??a.cells[i].innerText,y=b.cells[i].dataset.v??b.cells[i].innerText;
  const nx=parseFloat(x),ny=parseFloat(y);
  const c=(isNaN(nx)||isNaN(ny))?x.localeCompare(y):nx-ny;return asc?c:-c}});
 rows.forEach(r=>tb.appendChild(r));}});
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["oof", "holdout"], default="oof")
    ap.add_argument("--fold", type=int, default=0, help="--split holdout 時用哪一折的權重")
    ap.add_argument("--model", default="unet_tu-hrnet_w32")
    ap.add_argument("--ckpt-dir", default="checkpoints_obb",
                    help="第二階段權重目錄；框擾動版是 checkpoints_obb_jit")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu")
    out = ROOT / "eval" / "vis_e2e" / f"{args.split}_{args.ckpt_dir}"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()

    it = iter_oof() if args.split == "oof" else iter_holdout(args.fold)
    models: dict = {}
    teeth, figs = [], []
    for k, (fold, name, gray, gt) in enumerate(it):
        if fold not in models:
            models.clear()
            models[fold] = (load_mrcnn(fold), load_seg(args.ckpt_dir, args.model, fold, device))
        mr, seg = models[fold]
        mm, sc, e2e, sc_e, boxes = run_image(mr, seg, gray, args.thr, device)
        rows_m, map_m = match(mm, sc, gt, name)
        rows_e, map_e = match(e2e, sc_e, gt, name)

        by_m = {int(r["gt_idx"]): r for r in rows_m if r["kind"] == "TP"}
        by_e = {int(r["gt_idx"]): r for r in rows_e if r["kind"] == "TP"}
        img_rows = []
        for gi in range(len(gt)):
            r = {"fold": fold, "image": name, "tooth": gi}
            for tag, src in (("m", by_m), ("e", by_e)):
                for key in METRICS:
                    r[f"{tag}_{key}"] = float(src[gi][key]) if gi in src else np.nan
            img_rows.append(r)
        n_fp = sum(r["kind"] == "FP" for r in rows_e)
        n_fn = sum(np.isnan(r["e_hd95"]) for r in img_rows)
        worst = max([r["e_hd95"] for r in img_rows if not np.isnan(r["e_hd95"])], default=0.0)
        title = (f"fold {fold}  {name}  teeth {len(gt)}  E2E FN {n_fn}  FP {n_fp}  "
                 f"worst E2E HD95 {worst:.1f}px")
        fig = figure(gray, gt, mm, e2e, boxes, map_m, map_e, rows_m, rows_e, title)
        figs.append(((n_fn > 0, worst), name, fig, img_rows, n_fp))   # 有漏檢的排最前
        teeth += img_rows
        print(f"  {k + 1:3d}  fold {fold}  {name}  {len(gt)} 顆　漏檢 {n_fn}　"
              f"最差 E2E HD95 {worst:.1f}", flush=True)

    figs.sort(key=lambda f: f[0], reverse=True)
    body = []
    for rank, (_, name, fig, rows, n_fp) in enumerate(figs, 1):
        fn = f"{rank:03d}_{Path(name).stem}.png"
        cv2.imwrite(str(out / fn), fig)
        for r in rows:
            r["png"] = fn
            r["e2e_fp_in_image"] = n_fp

    teeth.sort(key=lambda r: -(np.inf if np.isnan(r["e_hd95"]) else r["e_hd95"]))
    with (out / "teeth.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(teeth[0]))
        wr.writeheader()
        wr.writerows(teeth)

    cols = [("影像", "l"), ("折", ""), ("牙#", "")] + \
        [(f"{p} {k}", "") for p in ("MRCNN", "E2E") for k in ("Dice", "HD95", "ASSD")] + \
        [("ΔDice E−M", ""), ("ΔHD95 E−M", ""), ("影像 FP", ""), ("圖", "l")]
    head = "".join(f'<th class="{c}">{html.escape(t)}</th>' for t, c in cols)
    hd = np.array([r["e_hd95"] for r in teeth if not np.isnan(r["e_hd95"])])
    bad, warn = (np.percentile(hd, 90), np.percentile(hd, 75)) if len(hd) else (np.inf, np.inf)

    def cell(v, nd):
        return '<td data-v="999999">FN</td>' if np.isnan(v) else f'<td>{v:.{nd}f}</td>'

    for r in teeth:
        e = r["e_hd95"]
        cls = "bad" if (np.isnan(e) or e >= bad) else "warn" if e >= warn else ""
        tds = [f'<td class="l">{html.escape(r["image"])}</td>', f'<td>{r["fold"]}</td>',
               f'<td>{r["tooth"]}</td>']
        for p in ("m", "e"):
            tds += [cell(r[f"{p}_dice"], 4), cell(r[f"{p}_hd95"], 1), cell(r[f"{p}_assd"], 2)]
        tds += [cell(r["e_dice"] - r["m_dice"], 4), cell(r["e_hd95"] - r["m_hd95"], 1),
                f'<td>{r["e2e_fp_in_image"]}</td>',
                f'<td class="l"><a href="{r["png"]}" target="_blank">'
                f'<img loading="lazy" src="{r["png"]}"></a></td>']
        body.append(f'<tr class="{cls}">{"".join(tds)}</tr>')

    def med(k):
        v = np.array([r[k] for r in teeth], float)
        return np.nanmedian(v), np.nanmean(v)

    note = (f"{len(figs)} 張影像、{len(teeth)} 顆牙　｜　"
            f"MRCNN Dice 中位 {med('m_dice')[0]:.4f} / HD95 中位 {med('m_hd95')[0]:.1f}　｜　"
            f"E2E Dice 中位 {med('e_dice')[0]:.4f} / HD95 中位 {med('e_hd95')[0]:.1f}　｜　"
            f"紅底 = E2E HD95 前 10% 差或漏檢，黃底 = 前 25%。點表頭排序，點縮圖開大圖。")
    (out / "index.html").write_text(PAGE.format(
        title=f"{args.split}　{args.ckpt_dir}　{args.model}", note=html.escape(note),
        head=head, body="".join(body)), encoding="utf-8")
    print(f"\n完成 → {out / 'index.html'}")
    print(f"  用瀏覽器打開：open '{out / 'index.html'}'")


if __name__ == "__main__":
    main()
