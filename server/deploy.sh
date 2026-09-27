#!/usr/bin/env bash
# 把推論服務部署到 pcastandardllm.online。
#
# 只送跑得起來的東西：scripts/、koi/setA/scripts/*.py、兩階段的權重，
# 其餘 baselines 與訓練資料都留在本機（整個 repo 有好幾 GB）。
#
#   bash server/deploy.sh            完整部署
#   bash server/deploy.sh code       只更新程式（改 app.py 之後用這個，很快）
set -euo pipefail

# 變數名不用 HOST：tcsh 內建的 $HOST 是本機名稱，會把整支腳本導到自己的電腦去。
# 部署目標。公開版不寫死實際主機——
#   BCRR_HOST=root@你的主機 bash server/deploy.sh
BCRR_HOST=${BCRR_HOST:?請設定 BCRR_HOST，例如 root@1.2.3.4}
DEST=/opt/bcrr/repo
REPO=$(cd "$(dirname "$0")/.." && pwd)
MODE=${1:-full}
SITE=${SITE:-$HOME/Downloads/bcrr-site}

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

say "檢查機器 $BCRR_HOST"
ssh "$BCRR_HOST" 'free -m | awk "/^Mem:/ {print \"RAM \" \$2 \" MB\"}"; df -h / | tail -1 | awk "{print \"DISK 剩 \" \$4}"'
ssh "$BCRR_HOST" 'M=$(free -m | awk "/^Mem:/ {print \$2}"); [ "$M" -ge 3500 ] || {
  echo; echo "!! RAM 只有 ${M} MB。常駐 1.5 GB、單張推論高水位約 4 GB，這台會 OOM。"
  echo "   先在 DigitalOcean 把 droplet 調到 4 GB 再跑這支。"; exit 1; }'

say "確保有 swap（吸收推論尾端的高水位，沒有的話 4 GB 會很緊）"
ssh "$BCRR_HOST" '[ -f /swapfile ] || {
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap -q /swapfile && swapon /swapfile
  grep -q "^/swapfile" /etc/fstab || echo "/swapfile none swap sw 0 0" >> /etc/fstab
  sysctl -q -w vm.swappiness=10; }
free -m | awk "/^Swap:/ {print \"swap \" \$2 \" MB\"}"'

say "建立目錄"
# rsync 不會自己建多層目錄，權重的兩個路徑要先開好
ssh "$BCRR_HOST" "mkdir -p $DEST/koi/setA/scripts $DEST/server \
  $DEST/koi/setA/checkpoints/original \
  $DEST/koi/setA/checkpoints_obb/seg2/unet_tu-hrnet_w32"

say "送程式"
rsync -az --delete "$REPO/scripts/"            "$BCRR_HOST:$DEST/scripts/"
rsync -az --include='*.py' --exclude='*'       "$REPO/koi/setA/scripts/" "$BCRR_HOST:$DEST/koi/setA/scripts/"
rsync -az "$REPO/server/app.py" "$REPO/server/auth.py" "$REPO/server/oauth.py" \
  "$REPO/server/vault.py" \
  "$REPO/server/requirements.txt" "$BCRR_HOST:$DEST/server/"
# 網站與 API 同源端出去，瀏覽器的跨來源與私有網路限制才不會擋
rsync -az "$SITE/" "$BCRR_HOST:/var/www/html/"
ssh "$BCRR_HOST" "touch $DEST/server/__init__.py"

if [ "$MODE" = full ]; then
  say "送權重（866 MB，第一次會很久）"
  rsync -az --stats \
    "$REPO/koi/setA/checkpoints/original/maskrcnn_final.pt" \
    "$BCRR_HOST:$DEST/koi/setA/checkpoints/original/"
  rsync -az --stats \
    "$REPO/koi/setA/checkpoints_obb/seg2/unet_tu-hrnet_w32/" \
    "$BCRR_HOST:$DEST/koi/setA/checkpoints_obb/seg2/unet_tu-hrnet_w32/"

  say "建 venv"
  ssh "$BCRR_HOST" "apt-get update -qq && apt-get install -y -qq python3-venv python3-dev >/dev/null
    python3 -m venv /opt/bcrr/venv
    /opt/bcrr/venv/bin/pip install -q --upgrade pip"
fi

# 兩種模式都要跑：已經裝好的套件 pip 會直接略過，很快；
# 少跑這一步的話，改了 requirements.txt 也不會生效。
say "裝套件"
ssh "$BCRR_HOST" "/opt/bcrr/venv/bin/pip install -q -r $DEST/server/requirements.txt"

say "裝 systemd 服務"
scp -q "$REPO/server/bcrr-api.service" "$BCRR_HOST:/etc/systemd/system/"
ssh "$BCRR_HOST" 'systemctl daemon-reload && systemctl enable -q bcrr-api && systemctl restart bcrr-api'

say "接上 nginx"
# 舊機器上這行可能還沒有：沒有它，服務會以為自己是 http，轉址網址對不上 Google
ssh "$BCRR_HOST" 'grep -q "X-Forwarded-Proto" /etc/nginx/sites-enabled/default || {
  sed -i "s|proxy_set_header Host \$host;|proxy_set_header Host \$host;\n\t\tproxy_set_header X-Forwarded-Proto \$scheme;|" /etc/nginx/sites-enabled/default
  nginx -t && systemctl reload nginx; }'
ssh "$BCRR_HOST" 'grep -q "location /api/" /etc/nginx/sites-enabled/default || {
  sed -i "0,/^\s*server\s*{/{s|^\(\s*server\s*{\)|\1\n\n\tlocation /api/ {\n\t\tproxy_pass http://127.0.0.1:8900;\n\t\tproxy_http_version 1.1;\n\t\tproxy_set_header Host \$host;\n\t\tproxy_set_header X-Forwarded-Proto \$scheme;\n\t\tproxy_read_timeout 300s;\n\t\tproxy_send_timeout 300s;\n\t\tclient_max_body_size 24m;\n\t}\n|}" /etc/nginx/sites-enabled/default
  nginx -t && systemctl reload nginx; }'

# 上面兩個 sed 只在缺的時候才跑，但設定檔可能是上一輪改好卻沒重讀的。
# 無條件驗一次再 reload，這動作本來就很便宜。
ssh "$BCRR_HOST" 'nginx -t && systemctl reload nginx'

say "等模型載入"
ssh "$BCRR_HOST" 'for i in $(seq 1 60); do
    curl -sf --max-time 3 http://127.0.0.1:8900/api/health && { echo; exit 0; }
    sleep 5
  done
  echo "起不來，看 journalctl -u bcrr-api -n 50"; exit 1'

say "從外面打一次"
# 用網域不要用 IP：certbot 把 port 80 的預設 server 設成一律回 404
curl -s --max-time 20 "https://pcastandardllm.online/api/health"; echo
