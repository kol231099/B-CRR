#!/usr/bin/env bash
# 在自己的電腦上把推論服務跑起來，網頁會自動找到它。
#
#   bash server/run-local.sh
#
# 這支同時把網站也端出來，所以直接開 http://127.0.0.1:8900/vault.html 最保險——
# 網站與 API 同源，不會踩到瀏覽器的跨來源／私有網路限制。
# 開線上的 pcastandardllm.online 也可以，前端會自己找過來（Safari 沒問題，
# Chrome 要看它的 Private Network Access 放不放行）。
# 要停就按 Ctrl-C。
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=${PORT:-8900}
FOLDS=${BCRR_FOLDS:-5}
SITE=${BCRR_SITE:-$HOME/Downloads/bcrr-site}

# 本機這支預設不要登入。服務綁在 127.0.0.1，只有這台機器進得來；
# 上台演示時還要連 Google 登入，是多一個會出事的環節。
# 想在本機測登入流程：  BCRR_OPEN=0 PORT=8900 bash server/run-local.sh
OPEN=${BCRR_OPEN:-1}

# 保險庫的資料目錄。伺服器上是 /var/lib/bcrr（root 寫得進去），本機不行，
# 沒有這一行服務會在啟動時就 PermissionError 掛掉。
DATA=${BCRR_DATA:-$HOME/.bcrr/data}
mkdir -p "$DATA" && chmod 700 "$DATA"

# 有 Metal GPU 就用，實測快三倍多，量測結果與 CPU 一致
# （驗證方式：python3 server/check_device.py）。
# 現場想強制用 CPU：  BCRR_DEVICE=cpu bash server/run-local.sh
DEVICE=${BCRR_DEVICE:-$(python3 -c "
import warnings; warnings.filterwarnings('ignore')
try:
    import torch
    print('mps' if torch.backends.mps.is_available() else
          'cuda' if torch.cuda.is_available() else 'cpu')
except Exception:
    print('cpu')" 2>/dev/null || echo cpu)}

miss=$(python3 - <<'PY'
import importlib.util
need = {"torch":"torch","torchvision":"torchvision","cv2":"opencv-python",
        "numpy":"numpy","scipy":"scipy","segmentation_models_pytorch":"segmentation-models-pytorch",
        "timm":"timm","fastapi":"fastapi","uvicorn":"uvicorn","multipart":"python-multipart"}
print(" ".join(p for m, p in need.items()
                if importlib.util.find_spec(m) is None))
PY
)
if [ -n "$miss" ]; then
  echo "缺少套件：$miss"
  echo "裝一下：  python3 -m pip install $miss"
  exit 1
fi

# 只看 LISTEN：用戶端留下的 socket（瀏覽器關掉之後的 CLOSED/TIME_WAIT）
# 也會出現在 -i tcp:PORT 裡，拿那個判斷會誤以為埠被佔用
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "127.0.0.1:$PORT 已經有東西在跑了。"
  echo "要換一個：  PORT=8901 bash server/run-local.sh"
  exit 1
fi

case "$DEVICE" in
  mps)  echo "用 Metal GPU 跑（比 CPU 快約三倍）" ;;
  cuda) echo "用 NVIDIA GPU 跑" ;;
  *)    echo "用 CPU 跑（這台沒有可用的 GPU）" ;;
esac
echo "載入模型中（${FOLDS} 折，約十幾秒）…"
if [ -d "$SITE" ]; then
  echo
  echo "  好了之後開這個網址：  http://127.0.0.1:${PORT}/vault.html"
  echo
else
  echo "（找不到網站目錄 $SITE，只提供 API）"
fi
[ "$OPEN" = 1 ] && echo "（本機模式：不需要登入）"
echo "保持這個視窗開著，網頁才能用。停止按 Ctrl-C。"
echo
BCRR_FOLDS="$FOLDS" BCRR_SITE="$SITE" BCRR_DEVICE="$DEVICE" BCRR_OPEN="$OPEN" \
BCRR_DATA="$DATA" \
exec python3 -m uvicorn server.app:app \
  --host 127.0.0.1 --port "$PORT" --log-level info
