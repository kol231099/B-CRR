#!/bin/bash
# 換掉保險庫的資料加密金鑰，並把所有既有影像重新加密。
#
#   bash server/rotate-key.sh
#
# 為什麼要有這個：金鑰放久了、或懷疑外洩過（備份被拷走、有人離職），
# 就該換。沒有這支腳本的話，換金鑰等於把所有存過的影像變成亂數。
#
# 流程與安全性：
#   1. 停服務——避免換到一半有人在寫新資料
#   2. 新鑰在伺服器上產生，不經過你的電腦、不進 shell 歷史
#   3. 舊鑰移到 BCRR_DATA_KEY_OLD，解得開舊檔；新鑰負責寫
#   4. 逐張解開再加密，寫暫存檔再 rename（原子操作，斷電不會留半個檔）
#   5. 全部成功才把舊鑰拿掉；有任何一張失敗就保留舊鑰，補跑即可
set -euo pipefail

# 部署目標。公開版不寫死實際主機——
#   BCRR_HOST=root@你的主機 bash server/deploy.sh
BCRR_HOST=${BCRR_HOST:?請設定 BCRR_HOST，例如 root@1.2.3.4}
echo "伺服器  $BCRR_HOST"
echo

ssh "$BCRR_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
ENV=/etc/bcrr.env

OLD=$(grep -m1 "^BCRR_DATA_KEY=" $ENV | cut -d= -f2) || true
if [ -z "${OLD:-}" ]; then echo "還沒有金鑰，不用輪替。先跑 set-allow.sh"; exit 1; fi

echo "停服務"
systemctl stop bcrr-api

NEW=$(openssl rand -hex 32)
# 舊鑰疊在既有的退役清單前面：上一次沒換完的也還讀得到
PREV=$(grep -m1 "^BCRR_DATA_KEY_OLD=" $ENV | cut -d= -f2 || true)
KEEP="$OLD${PREV:+,$PREV}"

sed -i "/^BCRR_DATA_KEY_OLD=/d" $ENV
sed -i "s|^BCRR_DATA_KEY=.*|BCRR_DATA_KEY=$NEW|" $ENV
printf 'BCRR_DATA_KEY_OLD=%s\n' "$KEEP" >> $ENV
chmod 600 $ENV
echo "新金鑰已寫入（舊鑰暫時保留在 BCRR_DATA_KEY_OLD）"
echo

set -a; . $ENV; set +a
cd /opt/bcrr/repo
/opt/bcrr/venv/bin/python3 - <<'PY'
import sys, json
sys.path.insert(0, "/opt/bcrr/repo/server")
import auth, vault
auth.init_db(); vault.init_db()
r = vault.rotate(progress=lambda m: print("  " + m, flush=True))
print(json.dumps(r, ensure_ascii=False))
sys.exit(0 if r["failed"] == 0 else 2)
PY
RC=$?

if [ $RC -eq 0 ]; then
  sed -i "/^BCRR_DATA_KEY_OLD=/d" $ENV
  echo "全部換完，舊鑰已退役（從此解不開任何東西）"
else
  echo "有檔案沒換成功，舊鑰先留著。修完再跑一次這支腳本即可。"
fi

systemctl start bcrr-api
exit $RC
REMOTE

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
