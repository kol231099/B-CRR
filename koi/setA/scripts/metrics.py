"""三條 pipeline 共用的評估邏輯。

比較的前提是所有方法走同一段程式碼。指標的細節——配對用哪個門檻、HD95 取
哪個百分位、邊界怎麼定義——只要各自實作就一定會有偏差，而那個偏差會跟方法
之間的真實差異混在一起，分不開。所以這裡集中一份，誰都不准自己重寫。

一律在**原圖座標**上計算。在 crop 內算 Dice 會系統性虛高，因為裁切已把大部分
背景移掉、分母變小；裁切組與不裁切組要能互比，就必須共用同一個座標系。
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def _surface(m: np.ndarray) -> np.ndarray:
    """物件的表面（邊界）像素：自身減去侵蝕一圈後的結果。"""
    er = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    return (m.astype(np.uint8) - er).astype(bool)


def surface_distances(pred: np.ndarray, gt: np.ndarray):
    """回傳 (預測表面到真實表面的距離, 真實表面到預測表面的距離)，單位像素。"""
    sp, sg = _surface(pred), _surface(gt)
    if not sp.any() or not sg.any():
        return np.array([]), np.array([])
    dp = distance_transform_edt(~sp)   # 到預測表面的距離場
    dg = distance_transform_edt(~sg)   # 到真實表面的距離場
    return dg[sp], dp[sg]


def assd(pred: np.ndarray, gt: np.ndarray) -> float:
    """平均對稱表面距離。與 HD95 互補：HD95 看最差的邊界，ASSD 看平均的邊界。

    文獻上與 Dice 併報是慣例（例如 TransUNet + nnU-Net 的牙齒分割研究報告
    Dice 0.9409、ASSD 0.5011）。
    """
    a, b = surface_distances(pred, gt)
    if not len(a) or not len(b):
        return float("nan")
    return float((a.sum() + b.sum()) / (len(a) + len(b)))


def nsd(pred: np.ndarray, gt: np.ndarray, tol: float = 3.0) -> float:
    """正規化表面 Dice：邊界落在容差 tol 像素內的比例。

    比 HD95 好解讀——「87% 的邊界誤差在 3 像素以內」是可以直接講給臨床聽的說法，
    而「HD95 = 11 px」不是。MICCAI 的分割挑戰賽近年普遍採用。
    """
    a, b = surface_distances(pred, gt)
    if not len(a) or not len(b):
        return float("nan")
    return float(((a <= tol).sum() + (b <= tol).sum()) / (len(a) + len(b)))


def boundary_iou(pred: np.ndarray, gt: np.ndarray, ratio: float = 0.02) -> float:
    """Boundary IoU（Cheng et al., CVPR 2021）：只在邊界帶上算 IoU。

    Dice 與 IoU 在大而完整的物件上會飽和——牙齒只要主體圈對就有 0.96，邊界差幾
    個像素幾乎不反映。Boundary IoU 把物件內部挖掉、只留邊界附近的帶狀區域再算
    IoU，因此對邊界品質敏感得多。這正是本任務需要的。

    帶寬取物件對角線的 ratio 倍（原論文建議 2%）。
    """
    if not pred.any() or not gt.any():
        return float("nan")
    ys, xs = np.nonzero(gt)
    diag = float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    d = max(1, int(round(diag * ratio)))
    k = np.ones((2 * d + 1, 2 * d + 1), np.uint8)
    bp = pred & ~cv2.erode(pred.astype(np.uint8), k, iterations=1).astype(bool)
    bg = gt & ~cv2.erode(gt.astype(np.uint8), k, iterations=1).astype(bool)
    union = np.logical_or(bp, bg).sum()
    return float(np.logical_and(bp, bg).sum() / union) if union else float("nan")


def rvd(pred: np.ndarray, gt: np.ndarray) -> float:
    """相對面積差 (|P| - |G|) / |G|。正值代表過度分割、負值代表分割不足。

    Dice 不區分方向——多圈 5% 和少圈 5% 分數一樣。RVD 看得出模型的系統性偏誤，
    例如 MedSAM 在本資料上的遮罩系統性內縮，這個指標會直接顯示為負值。
    """
    g = gt.sum()
    return float((pred.sum() - g) / g) if g else float("nan")


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """雙向的 95 百分位 Hausdorff 距離，單位為像素。

    取 95% 而非最大值，是為了不讓單一個離群像素主導整個數字。牙齒是大而完整
    的物件，Dice 很快會停在 0.96 上下，對邊界的細微偏移幾乎沒有反應；HD95 是
    這個任務真正能拉開差距的指標。
    """
    if not pred.any() or not gt.any():
        return float("nan")
    # 距離轉換算的是「到最近前景像素」的距離，所以要對補集做轉換
    dp = distance_transform_edt(~pred)
    dg = distance_transform_edt(~gt)
    ep = cv2.Canny(pred.astype(np.uint8) * 255, 100, 200) > 0
    eg = cv2.Canny(gt.astype(np.uint8) * 255, 100, 200) > 0
    if not ep.any() or not eg.any():
        return float("nan")
    return float(max(np.percentile(dg[ep], 95), np.percentile(dp[eg], 95)))


def match(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray,
          image: str, iou_thr: float = 0.5) -> tuple[list[dict], dict[int, int]]:
    """把預測與真實遮罩貪婪配對，回傳逐顆牙的紀錄與 pred→gt 的對應。

    漏檢與誤報必須跟 Dice 分開報：漏掉的牙根本不會進 Dice 的平均，所以一個
    漏了一半牙齒但把剩下的圈得很準的模型，Dice 會非常好看。

    每筆紀錄都帶 gt_idx（該影像內第幾顆真實牙齒）。三條 pipeline 測的是同一批
    57 顆牙，所以必須用配對檢定；沒有這個欄位就無法跨方法對齊到同一顆牙，只能
    比兩組的分布，統計檢定力會差很多。
    """
    rows: list[dict] = []
    matched: dict[int, int] = {}
    for pi, p in enumerate(pred):
        best, best_iou = -1, 0.0
        for gi, g in enumerate(gt):
            if gi in matched.values():
                continue
            union = np.logical_or(p, g).sum()
            iou = np.logical_and(p, g).sum() / union if union else 0.0
            if iou > best_iou:
                best, best_iou = gi, iou
        if best_iou >= iou_thr:
            matched[pi] = best
            g = gt[best]
            inter = np.logical_and(p, g).sum()
            rows.append({
                "image": image, "gt_idx": best, "kind": "TP", "score": round(float(scores[pi]), 4),
                "dice": round(2 * inter / (p.sum() + g.sum()), 4),
                "iou": round(best_iou, 4),
                "sens": round(inter / g.sum(), 4),                      # 像素層級召回
                "prec": round(inter / max(p.sum(), 1), 4),              # 像素層級精確
                "rvd": round(rvd(p, g), 4),
                "hd95": round(hd95(p, g), 2),
                "assd": round(assd(p, g), 3),
                "nsd3": round(nsd(p, g, 3.0), 4),
                "nsd5": round(nsd(p, g, 5.0), 4),
                "biou": round(boundary_iou(p, g), 4),
            })
        else:
            rows.append({"image": image, "gt_idx": "", "kind": "FP",
                         "score": round(float(scores[pi]), 4)})
    for gi in range(len(gt)):
        if gi not in matched.values():
            rows.append({"image": image, "gt_idx": gi, "kind": "FN", "score": ""})
    return rows, matched


FIELDS = ["image", "gt_idx", "kind", "score", "dice", "iou", "sens", "prec",
          "rvd", "hd95", "assd", "nsd3", "nsd5", "biou"]


def summarize(rows: list[dict], label: str) -> dict:
    tp = [r for r in rows if r["kind"] == "TP"]
    n_fp = sum(r["kind"] == "FP" for r in rows)
    n_fn = sum(r["kind"] == "FN" for r in rows)
    d = np.array([float(r["dice"]) for r in tp])
    h = np.array([float(r["hd95"]) for r in tp])
    out = {
        "label": label, "tp": len(tp), "fp": n_fp, "fn": n_fn,
        "dice_med": float(np.median(d)), "dice_mean": float(d.mean()), "dice_std": float(d.std()),
        "dice_min": float(d.min()),
        "hd95_med": float(np.nanmedian(h)), "hd95_mean": float(np.nanmean(h)), "hd95_max": float(np.nanmax(h)),
        "recall": len(tp) / (len(tp) + n_fn), "precision": len(tp) / (len(tp) + n_fp),
    }
    print(f"{label}　TP {out['tp']}　FP {out['fp']}　FN {out['fn']}")
    print(f"  Dice  中位 {out['dice_med']:.4f}　平均 {out['dice_mean']:.4f} ± {out['dice_std']:.4f}"
          f"　最低 {out['dice_min']:.4f}")
    print(f"  HD95  中位 {out['hd95_med']:.1f} px　平均 {out['hd95_mean']:.1f}　最高 {out['hd95_max']:.1f}")
    print(f"  recall {out['recall']:.3f}　precision {out['precision']:.3f}")
    return out
