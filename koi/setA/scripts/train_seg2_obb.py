"""用 OBB crop 訓練第二階段，其餘與 train_seg2.py 完全相同。

只改兩個模組變數：資料改讀 crops_obb/，權重改寫到 checkpoints_obb/，
避免覆蓋既有的 HBB 版權重。訓練超參、增強、fold 切分一律沿用，
這樣 OBB vs HBB 的差異才只來自裁切方式。

用法：
    py scripts/train_seg2_obb.py --arch unet --encoder tu-hrnet_w32 --fold 0
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_seg2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
train_seg2.CROPS = ROOT / "crops_obb"
train_seg2.CKPT = ROOT / "checkpoints_obb"

if __name__ == "__main__":
    train_seg2.main()
