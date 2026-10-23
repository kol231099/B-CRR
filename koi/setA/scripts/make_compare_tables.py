"""產生兩份左右對照表（PDF，論文格式：三線表、全黑、中文標楷體、英數 Times New Roman）。
  table_old_vs_new.pdf     原標註（n = 13）vs 重新標註（n = 10），皆為共同命中
  table_common_vs_per.pdf  重新標註：共同命中（n = 10）vs 各模型自身偵測（per-model）
用法：python3 make_tables.py（只需要 matplotlib）
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ── 字型：中文標楷體、英數 Times New Roman ──
KAI_FILES = ["/Applications/Microsoft Word.app/Contents/Resources/DFonts/kaiu.ttf",
             "/Library/Fonts/kaiu.ttf", os.path.expanduser("~/Library/Fonts/kaiu.ttf"),
             "C:/Windows/Fonts/kaiu.ttf", "/Library/Fonts/BiauKai.ttf", "/Library/Fonts/BiauKai.ttc"]
KAI_FILES += glob.glob("/System/Library/Fonts/Supplemental/*[Kk]ai*.tt*")
KAI_FILES += glob.glob("/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/BiauKai*.tt*")
KAI_FILES += glob.glob(os.path.expanduser("~/Library/Fonts/*[Kk]ai*.tt*"))
for f in KAI_FILES:
    if os.path.exists(f):
        try:
            font_manager.fontManager.addfont(f)
        except Exception:
            pass
have = {f.name for f in font_manager.fontManager.ttflist}
KAI = next((n for n in ("DFKai-SB", "BiauKai", "BiauKaiTC", "標楷體", "Kaiti TC", "STKaiti") if n in have), None)
TIMES = next((n for n in ("Times New Roman", "Times", "Liberation Serif") if n in have), "serif")
if KAI is None:
    print("⚠ 找不到標楷體，中文會用替代字型。請確認已安裝標楷體（Word 內附的 kaiu.ttf）。")
    KAI = next((n for n in ("WenQuanYi Zen Hei", "PingFang TC", "Heiti TC") if n in have), "sans-serif")
plt.rcParams.update({"font.family": [TIMES, KAI], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "text.color": "black", "axes.edgecolor": "black"})
print(f"字型：英數 {TIMES}、中文 {KAI}")

OURS = "Mask R-CNN +\nU-Net-HRNet-w32"
METRICS = ["DICE ↑", "IOU ↑", "HD95 (px) ↓", "ASSD (px) ↓", "Area ICC ↑"]

# 每列：(名稱, 額外欄位 (主, 副) 或 None, [(值, p 或 None) × 4], Area ICC)；p 為 None 者為本方法
OLD = [("Mask R-CNN", None, [("0.9742", "= 0.003"), ("0.9496", "= 0.003"), ("9.97", "= 0.110"), ("3.57", "= 0.006")], "0.9997"),
       (OURS, None, [("0.9770", None), ("0.9549", None), ("9.53", None), ("3.48", None)], "0.9998"),
       ("YOLOv8-seg", None, [("0.9562", "< 0.001"), ("0.9161", "< 0.001"), ("43.50", "< 0.001"), ("7.54", "< 0.001")], "0.9895"),
       ("YOLO11-seg", None, [("0.9546", "< 0.001"), ("0.9132", "< 0.001"), ("33.76", "< 0.001"), ("5.88", "< 0.001")], "0.9945"),
       ("YOLO26-seg", None, [("0.9650", "< 0.001"), ("0.9324", "< 0.001"), ("20.18", "< 0.001"), ("4.64", "< 0.001")], "0.9960")]
NEW = [("Mask R-CNN", None, [("0.9747", "= 0.193"), ("0.9507", "= 0.160"), ("7.90", "= 0.557"), ("3.05", "= 0.131")], "0.9997"),
       (OURS, None, [("0.9763", None), ("0.9537", None), ("7.45", None), ("2.80", None)], "0.9998"),
       ("YOLOv8-seg", None, [("0.9637", "= 0.008"), ("0.9300", "= 0.008"), ("15.56", "= 0.008"), ("3.87", "= 0.008")], "0.9979"),
       ("YOLO11-seg", None, [("0.9641", "= 0.008"), ("0.9307", "= 0.008"), ("15.56", "= 0.008"), ("3.81", "= 0.008")], "0.9964"),
       ("YOLO26-seg", None, [("0.9705", "= 0.039"), ("0.9427", "= 0.039"), ("11.77", "= 0.012"), ("3.34", "= 0.012")], "0.9987")]
PER = [("Mask R-CNN", ("25/26", "配對 25"), [("0.9737", "< 0.001"), ("0.9487", "< 0.001"), ("9.30", "< 0.001"), ("3.42", "< 0.001")], "0.9997"),
       (OURS, ("25/26", None), [("0.9762", None), ("0.9536", None), ("8.42", None), ("3.34", None)], "0.9998"),
       ("YOLOv8-seg", ("19/26", "配對 19"), [("0.9650", "< 0.001"), ("0.9323", "< 0.001"), ("21.04", "< 0.001"), ("4.58", "< 0.001")], "0.9972"),
       ("YOLO11-seg", ("15/26", "配對 15"), [("0.9623", "< 0.001"), ("0.9275", "< 0.001"), ("21.13", "< 0.001"), ("4.72", "< 0.001")], "0.9928"),
       ("YOLO26-seg", ("16/26", "配對 15"), [("0.9705", "= 0.010"), ("0.9427", "= 0.010"), ("9.96", "< 0.001"), ("3.20$^{\\dagger}$", "< 0.001")], "0.9975")]


def draw(ax, caption, rows, extra=None, notes=()):
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 0.975, caption, fontsize=12, va="top")
    cols = ["Pipeline"] + ([extra] if extra else []) + METRICS
    x0 = 0.24 if extra else 0.26
    step = (1 - x0) / (len(cols) - 1)
    xs = [0.0] + [x0 + step * (i + 0.5) for i in range(len(cols) - 1)]
    top, hh, rh = 0.885, 0.075, 0.12
    rule = lambda y, w: ax.plot([0, 1], [y, y], color="black", lw=w, solid_capstyle="butt")
    rule(top, 1.5)
    for j, c in enumerate(cols):
        ax.text(xs[j], top - hh / 2, c, fontsize=10.5, ha="left" if j == 0 else "center", va="center")
    y = top - hh
    rule(y, 0.75)
    for name, ex, vals, icc in rows:
        ours = vals[0][1] is None
        fw = "bold" if ours else "normal"
        cy = y - rh / 2
        ax.text(xs[0], cy, name, fontsize=10.5, fontweight=fw, va="center", linespacing=1.15)
        j = 1
        if extra:
            main, sub = ex
            if sub:
                ax.text(xs[j], cy + 0.02, main, fontsize=10.5, fontweight=fw, ha="center", va="center")
                ax.text(xs[j], cy - 0.025, f"（{sub}）", fontsize=8.5, ha="center", va="center")
            else:
                ax.text(xs[j], cy, main, fontsize=10.5, fontweight=fw, ha="center", va="center")
            j += 1
        for v, p in vals:
            if p is None:
                ax.text(xs[j], cy, v, fontsize=10.5, fontweight="bold", ha="center", va="center")
            else:
                ax.text(xs[j], cy + 0.02, v, fontsize=10.5, ha="center", va="center")
                ax.text(xs[j], cy - 0.025, rf"($p$ {p})", fontsize=8.5, ha="center", va="center")
            j += 1
        ax.text(xs[j], cy, icc, fontsize=10.5, fontweight=fw, ha="center", va="center")
        y -= rh
    rule(y, 1.5)
    for k, t in enumerate(notes):
        ax.text(0, y - 0.035 - k * 0.045, t, fontsize=9, va="top")


def page(fname, left, right, footer):
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 5.8))
    draw(axes[0], *left)
    draw(axes[1], *right)
    for k, t in enumerate(footer):
        fig.text(0.012, 0.075 - k * 0.032, t, fontsize=9)
    plt.subplots_adjust(left=0.012, right=0.988, top=0.97, bottom=0.10, wspace=0.06)
    fig.savefig(fname)
    fig.savefig(fname.replace(".pdf", ".png"), dpi=110)
    print("→", fname)


COMMON = "數值為五折逐折中位數之平均；僅計五種模型於五折中皆偵測到之牙齒。"
FOOT = ["本方法：Mask R-CNN + U-Net-HRNet-w32（粗體）。p 值：與本方法之雙尾配對 Wilcoxon 符號等級檢定，"
        "每顆牙齒取五折平均，並於各指標內以 Holm 法校正。",
        "↑ 表示數值越高越好；↓ 表示數值越低越好。"]

page("table_old_vs_new.pdf",
     ("(a) 原標註（共同命中，n = 13）", OLD, None, (COMMON,)),
     ("(b) 重新標註（共同命中，n = 10）", NEW, None, (COMMON,)),
     FOOT)
page("table_common_vs_per.pdf",
     ("(a) 重新標註：共同命中（各模型 n = 10）", NEW, None, (COMMON,)),
     ("(b) 重新標註：各模型自身偵測之牙齒（per-model）", PER, "Detected",
      ("數值為各模型於五折中皆偵測到之牙齒；p 值以本方法與該模型皆偵測到之牙齒配對（括號內為配對數）。",
       "† 於兩者皆偵測到之 15 顆牙齒上，本方法 ASSD 為 2.51 px，YOLO26-seg 為 3.03 px（本方法較佳 13/15 顆）。")),
     FOOT)
