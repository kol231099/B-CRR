#!/usr/bin/env python3
"""比對同一批影像在 CPU 與 GPU 上算出來的指標是否一致。

為什麼要有這支：換裝置不該改變任何結論。浮點在不同硬體上本來就會有
最後幾位的差異，問題是那點差異會不會把某顆牙從「量得出來」翻成「量不出來」，
或把風險等級翻過門檻。這支就是回答這件事的，跑過再上台。

    python3 server/check_device.py            # 預設比 cpu vs mps
    BCRR_ALT=cuda python3 server/check_device.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TESTSET = REPO / "koi" / "setA" / "testset"
KEYS = ("crr", "ablr", "bcrr", "conf")
# 報告書上變紅色的門檻，跟 vault.html 的 CEIL 一致
CEIL = {"crr": 1.00, "ablr": 0.33, "bcrr": 1.50}


def run(device: str, paths: list[Path]) -> dict:
    """在子行程裡跑，避免兩個裝置的模型同時佔記憶體。"""
    code = r'''
import json, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, %r); sys.path.insert(0, %r)
import app
app.MODELS["det"] = app.load_detector()
app.MODELS["segs"] = app.load_segmenters(app.FOLDS)
out = {}
for p in sys.argv[1:]:
    img, _ = app.read_image(open(p, "rb").read())
    out[p.split("/")[-1]] = app.analyse(img)
print("<<<" + json.dumps(out) + ">>>")
''' % (str(REPO), str(HERE))
    env = dict(os.environ, BCRR_DEVICE=device)
    r = subprocess.run([sys.executable, "-c", code] + [str(p) for p in paths],
                       capture_output=True, text=True, env=env, cwd=REPO)
    if "<<<" not in r.stdout:
        print(r.stderr[-2000:]); raise SystemExit(f"{device} 跑失敗")
    return json.loads(r.stdout.split("<<<")[1].split(">>>")[0])


def main() -> None:
    alt = os.environ.get("BCRR_ALT", "mps")
    paths = sorted(TESTSET.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"找不到影像：{TESTSET}")
    print(f"{len(paths)} 張，比較 cpu 與 {alt}\n")

    a, b = run("cpu", paths), run(alt, paths)

    n_teeth = flip = cross = 0
    worst = {k: 0.0 for k in KEYS}
    worst_at = {k: "" for k in KEYS}
    for name in a:
        ta, tb = a[name], b[name]
        if len(ta) != len(tb):
            print(f"  !! {name} 牙數不同：cpu {len(ta)} / {alt} {len(tb)}")
            flip += 1
            continue
        for i, (x, y) in enumerate(zip(ta, tb)):
            n_teeth += 1
            if bool(x["ok"]) != bool(y["ok"]):
                print(f"  !! {name} 第 {i+1} 顆：量得出來與否不同"
                      f"（cpu {x['ok']} / {alt} {y['ok']}）")
                flip += 1
                continue
            if not x["ok"]:
                continue
            for k in KEYS:
                if k in x and k in y:
                    d = abs(float(x[k]) - float(y[k]))
                    if d > worst[k]:
                        worst[k], worst_at[k] = d, f"{name} 第 {i+1} 顆"
                    # 真正要緊的不是差多少，是有沒有跨過會變紅的那條線，
                    # 或是四捨五入到報告書的兩位小數之後顯示不一樣
                    if k in CEIL and (float(x[k]) > CEIL[k]) != (float(y[k]) > CEIL[k]):
                        print(f"  !! {name} 第 {i+1} 顆 {k} 跨過門檻 {CEIL[k]}："
                              f"cpu {x[k]} / {alt} {y[k]}")
                        cross += 1
                    if k in CEIL and round(float(x[k]), 2) != round(float(y[k]), 2):
                        print(f"  ·  {name} 第 {i+1} 顆 {k} 顯示值不同："
                              f"{float(x[k]):.2f} / {float(y[k]):.2f}")

    print(f"\n共 {n_teeth} 顆牙")
    print(f"量得出來與否不同的：{flip} 顆"
          + ("  ← 有問題，不要換裝置" if flip else "  ← 沒有"))
    print(f"跨過紅色門檻的：    {cross} 顆"
          + ("  ← 有問題，不要換裝置" if cross else "  ← 沒有"))
    print("\n指標的最大差異：")
    for k in KEYS:
        print(f"  {k:5s} {worst[k]:.6f}   {worst_at[k]}")
    print("\n數值本身一定會有浮點差異，那不是問題；問題只有兩個："
          "\n結論會不會翻（量得出來與否）、顯示會不會變（跨門檻或兩位小數不同）。"
          "\n上面兩行都是 0 就可以放心用。")


if __name__ == "__main__":
    main()
