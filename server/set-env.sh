#!/bin/bash
# 把 Google OAuth 憑證與白名單寫進伺服器的 /etc/bcrr.env（權限 600）。
#
#   bash server/set-env.sh
#
# 密鑰用互動方式輸入，不落在指令列、不落在 shell 歷史、不落在任何檔案。
# 故意帶 shebang：這樣不管你的互動 shell 是 zsh、bash 還是 tcsh，
# 裡面都是 bash 在跑，read 的行為才一致。
set -euo pipefail

# 變數名不用 HOST：tcsh 內建的 $HOST 是本機名稱，會把整支腳本導到自己的電腦去。
# 部署目標。公開版不寫死實際主機——
#   BCRR_HOST=root@你的主機 bash server/deploy.sh
BCRR_HOST=${BCRR_HOST:?請設定 BCRR_HOST，例如 root@1.2.3.4}
CLIENT_ID=${CLIENT_ID:?請設定 CLIENT_ID（Google Cloud 主控台取得）}
BASE_URL=${BASE_URL:-https://your-domain.example}
# 白名單：只有這些信箱進得了保險庫。冒號後面是角色，不寫就是 doctor。
# 要加人就改這行，或執行前先 export ALLOW="a@x.com:admin,b@y.com"
ALLOW=${ALLOW:-you@example.com:admin}

# 資料加密金鑰。已經有的話一定要沿用——換掉等於把所有存過的影像變成亂數，
# 救不回來。所以先去伺服器上撈，撈不到才產生新的。
KEY=$(ssh "$BCRR_HOST" 'grep -m1 "^BCRR_DATA_KEY=" /etc/bcrr.env 2>/dev/null | cut -d= -f2' || true)
if [ -n "$KEY" ]; then
  KEY_NOTE="沿用伺服器上既有的（保險庫裡的影像才解得開）"
else
  KEY=$(python3 -c 'import os; print(os.urandom(32).hex())')
  KEY_NOTE="這次新產生的"
fi

printf '伺服器      %s\n' "$BCRR_HOST"
printf '加密金鑰    %s\n' "$KEY_NOTE"
printf '用戶端 ID   %s\n' "$CLIENT_ID"
printf '網站網址    %s\n' "$BASE_URL"
printf '白名單      %s\n\n' "$ALLOW"

printf '貼上 Client Secret 然後按 Enter（不會顯示）： '
read -rs SECRET
echo
[ -n "$SECRET" ] || { echo "沒有輸入任何東西，取消。"; exit 1; }

printf 'BCRR_GOOGLE_CLIENT_ID=%s\nBCRR_GOOGLE_CLIENT_SECRET=%s\nBCRR_BASE_URL=%s\nBCRR_ALLOW=%s\nBCRR_DATA_KEY=%s\n' \
  "$CLIENT_ID" "$SECRET" "$BASE_URL" "$ALLOW" "$KEY" \
  | ssh "$BCRR_HOST" 'cat > /etc/bcrr.env && chmod 600 /etc/bcrr.env'
unset SECRET

echo
echo "寫好了，檢查："
ssh "$BCRR_HOST" 'ls -l /etc/bcrr.env
  awk -F= "/SECRET|DATA_KEY/ {print \"  \" \$1 \" = \" length(\$2) \" 字元（不顯示內容）\"}
           !/SECRET|DATA_KEY/ && NF {print \"  \" \$0}" /etc/bcrr.env'
