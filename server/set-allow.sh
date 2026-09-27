#!/bin/bash
# 設定誰能進保險庫，順便確保資料加密金鑰存在。
#
#   bash server/set-allow.sh                      開放任何 Google 帳號
#   bash server/set-allow.sh a@x.com b@y.com      只有這幾個人（第一個是管理者）
#   ALLOW="me@x.com:admin,*" bash server/set-allow.sh    完全自己指定
#
# 不碰 Client Secret，所以不用再輸入一次。
set -euo pipefail

# 部署目標。公開版不寫死實際主機——
#   BCRR_HOST=root@你的主機 bash server/deploy.sh
BCRR_HOST=${BCRR_HOST:?請設定 BCRR_HOST，例如 root@1.2.3.4}
ADMIN=${ADMIN:?請設定 ADMIN，例如 you@example.com}

if [ -n "${ALLOW:-}" ]; then
  :
elif [ $# -gt 0 ]; then
  # 第一個給 admin，其餘一般身分
  ALLOW="$1:admin"; shift
  for e in "$@"; do ALLOW="$ALLOW,$e"; done
else
  ALLOW="$ADMIN:admin,*"       # * = 任何 Google 帳號都放行
fi

echo "伺服器  $BCRR_HOST"
echo "白名單  $ALLOW"
case "$ALLOW" in
  *'*'*) echo "        （含 * ：任何 Google 帳號都進得來）" ;;
esac
echo

# 金鑰在伺服器上產生，不經過這台機器、不進 shell 歷史。
# 已經有就絕對不動——換掉等於把所有存過的影像變成亂數，救不回來。
ssh "$BCRR_HOST" "
  set -e
  if grep -q '^BCRR_DATA_KEY=' /etc/bcrr.env; then
    echo '加密金鑰：沿用既有的'
  else
    printf 'BCRR_DATA_KEY=%s\n' \"\$(openssl rand -hex 32)\" >> /etc/bcrr.env
    echo '加密金鑰：這次新產生的'
  fi
  sed -i 's|^BCRR_ALLOW=.*|BCRR_ALLOW=${ALLOW}|' /etc/bcrr.env
  grep -q '^BCRR_ALLOW=' /etc/bcrr.env || printf 'BCRR_ALLOW=%s\n' '${ALLOW}' >> /etc/bcrr.env
  chmod 600 /etc/bcrr.env
  systemctl restart bcrr-api
"

echo
echo "現在的設定（敏感欄位只顯示長度）："
ssh "$BCRR_HOST" "awk -F= '
  /SECRET|DATA_KEY/ { print \"  \" \$1 \" = \" length(\$2) \" 字元\"; next }
  NF { print \"  \" \$0 }' /etc/bcrr.env"

echo
echo "等服務起來…"
for i in $(seq 1 40); do
  if curl -sf --max-time 4 https://pcastandardllm.online/api/health >/dev/null; then
    echo "  好了：$(curl -s --max-time 5 https://pcastandardllm.online/api/health)"
    exit 0
  fi
  sleep 5
done
echo "  起不來，看：ssh $BCRR_HOST 'journalctl -u bcrr-api -n 40'"
exit 1
