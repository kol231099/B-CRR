"""解析預訓練權重的路徑。

ultralytics 若拿到裸檔名（例如 "sam2.1_b.pt"）會在**當前工作目錄**尋找，找不到就
下載一份放在那裡——於是每次從不同目錄執行就多一份散落的權重檔。這裡統一解析到
koi/baselines/weights/，找不到才回退成裸檔名讓 ultralytics 自行下載。

放在 baselines/ 而非 setA/ 是因為這些是 SAM 2 與 YOLO 的官方權重，屬於對照組的
模型，不是 setA 這條主線的產物。
"""

from __future__ import annotations

from pathlib import Path

WEIGHTS = Path(__file__).resolve().parent.parent.parent / "baselines" / "weights"


def weight(name: str) -> str:
    """回傳權重的完整路徑；不存在時回傳原檔名，交給套件自行下載。"""
    p = WEIGHTS / name
    return str(p) if p.exists() else name
