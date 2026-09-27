"""每個帳號自己的保險庫：影像與分析結果存在伺服器上。

影像落地加密。用的是 AES-256-GCM，金鑰放在 /etc/bcrr.env（權限 600），
不跟資料放同一個地方——磁碟快照、備份檔、被拿走的整顆 volume，沒有那把
金鑰都是一堆亂數。GCM 附帶完整性驗證，檔案被改過解不開，不會默默讀出
壞掉的影像。

這擋得住什麼、擋不住什麼，寫清楚比較好：
  擋得住   磁碟快照外流、備份檔外流、雲端商的儲存層被讀
  擋不住   伺服器本身被入侵（程式跑著就握有金鑰）
沒有金鑰時整個保險庫回 503 而不是改存明文——醫療影像不該因為有人忘了
設環境變數就躺在磁碟上。

檔名不用信箱，用信箱的雜湊。目錄列表本身不該洩漏有哪些人用過。
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path

import auth

DATA = Path(os.environ.get("BCRR_DATA", "/var/lib/bcrr"))
SLOTS = 49
MAX_ITEMS = int(os.environ.get("BCRR_MAX_ITEMS", "200"))
MAX_BYTES = int(os.environ.get("BCRR_MAX_MB", "200")) * 1024 * 1024
OK_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
NAME_MAX = 512          # 檔名是密文，base64 之後比原本長很多


class Locked(Exception):
    """沒有設金鑰。呼叫端要回 503，不要退回明文。"""


def _parse(raw: str) -> bytes | None:
    raw = raw.strip()
    if len(raw) != 64:
        return None
    try:
        return bytes.fromhex(raw)
    except ValueError:
        return None


def fingerprint(key: bytes) -> str:
    """金鑰的短指紋。存在每一筆資料旁邊，換鑰之後才知道哪筆該用哪把解。
    存的是雜湊不是金鑰本身，資料庫外流也推不回金鑰。"""
    return hashlib.sha256(key).hexdigest()[:8]


def _key() -> bytes:
    """現用的金鑰：新資料一律用這把加密。"""
    k = _parse(os.environ.get("BCRR_DATA_KEY", ""))
    if k is None:
        raise Locked("伺服器沒有設定資料加密金鑰（BCRR_DATA_KEY，64 個十六進位字元）")
    return k


def _keys() -> dict[str, bytes]:
    """所有解得開資料的金鑰（現用的 + 還沒退役的舊鑰）。

    輪替不是一瞬間的事：換鑰要把每一張圖重新加密，中間可能被中斷。
    舊鑰留在 BCRR_DATA_KEY_OLD 裡，換到一半停掉也還讀得到，補跑就好。
    全部換完再把那行拿掉。
    """
    out = {}
    for raw in os.environ.get("BCRR_DATA_KEY_OLD", "").split(","):
        k = _parse(raw)
        if k:
            out[fingerprint(k)] = k
    cur = _key()
    out[fingerprint(cur)] = cur
    return out


def ready() -> bool:
    try:
        _key()
        return True
    except Locked:
        return False


def init_db() -> None:
    with auth.connect() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS vault_items (
                id     TEXT PRIMARY KEY,
                owner  TEXT NOT NULL,
                slot   INTEGER NOT NULL,
                name   TEXT NOT NULL,
                bytes  INTEGER NOT NULL,
                mime   TEXT NOT NULL,
                at     INTEGER NOT NULL,
                keyid  TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_items_owner ON vault_items(owner, slot);
            -- 使用者的資料金鑰，用兩把鑰匙各鎖一份。伺服器存的是「鎖住的盒子」，
            -- 兩份都打不開：通關密語與救援碼都只在使用者的瀏覽器裡出現過。
            CREATE TABLE IF NOT EXISTS vault_keys (
                owner     TEXT PRIMARY KEY,
                salt      TEXT NOT NULL,   -- PBKDF2 的 salt，不是秘密
                wrap_pass TEXT NOT NULL,   -- 用通關密語衍生的金鑰包起來的資料金鑰
                wrap_rec  TEXT NOT NULL,   -- 用救援碼包起來的同一把資料金鑰
                at        INTEGER NOT NULL,
                changed   INTEGER
            );
            CREATE TABLE IF NOT EXISTS vault_slots (
                owner   TEXT NOT NULL,
                slot    INTEGER NOT NULL,
                done    INTEGER NOT NULL DEFAULT 0,
                combo   TEXT,
                results TEXT,
                at      INTEGER NOT NULL,
                PRIMARY KEY (owner, slot)
            );
        """)
    # 舊資料庫補欄位。keyid 是 NULL 代表「換鑰功能之前存的」，
    # 讀的時候把已知的金鑰逐一試過去——GCM 會驗證，錯的鑰一定解不開。
    with auth.connect() as con:
        have = {r["name"] for r in con.execute("PRAGMA table_info(vault_items)")}
        if "keyid" not in have:
            con.execute("ALTER TABLE vault_items ADD COLUMN keyid TEXT")

    DATA.mkdir(parents=True, exist_ok=True)
    os.chmod(DATA, 0o700)


def _dir(owner: str) -> Path:
    # 目錄名用雜湊：光看檔案系統不該知道有誰用過這個系統
    d = DATA / hashlib.sha256(owner.encode()).hexdigest()[:32]
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d


def _path(owner: str, item_id: str) -> Path:
    return _dir(owner) / f"{item_id}.bin"


def usage(owner: str) -> dict:
    with auth.connect() as con:
        r = con.execute("SELECT count(*) n, coalesce(sum(bytes),0) b "
                        "FROM vault_items WHERE owner=?", (owner,)).fetchone()
    return {"items": r["n"], "bytes": r["b"],
            "max_items": MAX_ITEMS, "max_bytes": MAX_BYTES}


def put(owner: str, slot: int, name: str, mime: str, raw: bytes) -> dict:
    """存一張影像。回傳這一筆的中繼資料（不含內容）。

    raw 是瀏覽器加密過才送上來的密文，name 也是。伺服器從頭到尾沒看過
    明文，所以這裡不能、也不該檢查「這是不是一張合法的影像」——那件事
    在使用者的瀏覽器裡做完了。mime 只是使用者宣告的原始格式，拿來讓
    前端解密之後知道要怎麼顯示，本身不是秘密。

    落地前還會再用伺服器的金鑰加密一次。對「伺服器管理者偷看」沒有幫助
    （他有那把鑰），但對「整顆磁碟被複製走」多一道；而且成本幾乎是零。
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not 0 <= slot < SLOTS:
        raise ValueError("保險箱編號超出範圍")
    if mime not in OK_MIME:
        raise ValueError(f"不支援的影像格式：{mime}")
    if len(name) > NAME_MAX:
        raise ValueError("檔名過長")
    u = usage(owner)
    if u["items"] >= MAX_ITEMS:
        raise ValueError(f"最多只能存 {MAX_ITEMS} 張影像，請先刪掉一些")
    if u["bytes"] + len(raw) > MAX_BYTES:
        raise ValueError(f"空間已滿（上限 {MAX_BYTES // 1024 // 1024} MB），請先刪掉一些")

    item_id = secrets.token_hex(16)
    key = _key()
    nonce = os.urandom(12)
    # 把 id 綁進驗證資料：檔案被換成另一筆的內容也會解不開
    blob = nonce + AESGCM(key).encrypt(nonce, raw, item_id.encode())

    p = _path(owner, item_id)
    p.write_bytes(blob)
    os.chmod(p, 0o600)

    now = int(time.time())
    with auth.connect() as con:
        con.execute("INSERT INTO vault_items(id, owner, slot, name, bytes, mime, at, keyid) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (item_id, owner, slot, name[:NAME_MAX], len(raw), mime, now,
                     fingerprint(key)))
        con.execute("INSERT INTO vault_slots(owner, slot, at) VALUES(?,?,?) "
                    "ON CONFLICT(owner, slot) DO UPDATE SET at=excluded.at",
                    (owner, slot, now))
    return {"id": item_id, "name": name[:NAME_MAX], "size": len(raw), "mime": mime}


def read(owner: str, item_id: str) -> tuple[bytes, str]:
    """取回影像。owner 一定要帶，不然就是任何人都能猜 id 拿別人的東西。"""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    with auth.connect() as con:
        r = con.execute("SELECT mime, keyid FROM vault_items WHERE id=? AND owner=?",
                        (item_id, owner)).fetchone()
    if r is None:
        raise KeyError(item_id)
    blob = _path(owner, item_id).read_bytes()
    keys = _keys()
    order = ([keys[r["keyid"]]] if r["keyid"] in keys else list(keys.values()))
    for key in order:
        try:
            return (AESGCM(key).decrypt(blob[:12], blob[12:], item_id.encode()),
                    r["mime"])
        except Exception:                                          # noqa: BLE001
            continue
    raise Locked("沒有任何一把金鑰解得開這個檔案")


def drop_item(owner: str, item_id: str) -> bool:
    with auth.connect() as con:
        n = con.execute("DELETE FROM vault_items WHERE id=? AND owner=?",
                        (item_id, owner)).rowcount
    if n:
        _path(owner, item_id).unlink(missing_ok=True)
    return bool(n)


def clear_slot(owner: str, slot: int) -> int:
    with auth.connect() as con:
        ids = [r["id"] for r in con.execute(
            "SELECT id FROM vault_items WHERE owner=? AND slot=?", (owner, slot))]
        con.execute("DELETE FROM vault_items WHERE owner=? AND slot=?", (owner, slot))
        con.execute("DELETE FROM vault_slots WHERE owner=? AND slot=?", (owner, slot))
    for i in ids:
        _path(owner, i).unlink(missing_ok=True)
    return len(ids)


def save_slot(owner: str, slot: int, done: bool | None = None,
              combo: list | None = None, results: list | None = None) -> None:
    if not 0 <= slot < SLOTS:
        raise ValueError("保險箱編號超出範圍")
    now = int(time.time())
    with auth.connect() as con:
        con.execute("INSERT INTO vault_slots(owner, slot, at) VALUES(?,?,?) "
                    "ON CONFLICT(owner, slot) DO NOTHING", (owner, slot, now))
        if done is not None:
            con.execute("UPDATE vault_slots SET done=?, at=? WHERE owner=? AND slot=?",
                        (int(done), now, owner, slot))
        if combo is not None:
            con.execute("UPDATE vault_slots SET combo=? WHERE owner=? AND slot=?",
                        (json.dumps(combo), owner, slot))
        if results is not None:
            con.execute("UPDATE vault_slots SET results=? WHERE owner=? AND slot=?",
                        (json.dumps(results), owner, slot))


def load(owner: str) -> dict:
    """整個保險庫的狀態，前端進頁面時拿這一份把畫面長回來。"""
    slots = [{"slot": i, "done": False, "combo": None,
              "items": [], "results": None} for i in range(SLOTS)]
    with auth.connect() as con:
        for r in con.execute("SELECT * FROM vault_slots WHERE owner=?", (owner,)):
            s = slots[r["slot"]]
            s["done"] = bool(r["done"])
            s["combo"] = json.loads(r["combo"]) if r["combo"] else None
            s["results"] = json.loads(r["results"]) if r["results"] else None
        for r in con.execute("SELECT id, slot, name, bytes, mime FROM vault_items "
                             "WHERE owner=? ORDER BY at", (owner,)):
            slots[r["slot"]]["items"].append(
                {"id": r["id"], "name": r["name"], "size": r["bytes"], "mime": r["mime"]})
    return {"slots": slots, "usage": usage(owner)}


# ── 保存期限 ──────────────────────────────────────────────────────────────

RETAIN_DAYS = int(os.environ.get("BCRR_RETAIN_DAYS", "30"))


def sweep(now: float | None = None) -> int:
    """刪掉超過保存期限的影像。

    資料最小化：留著用不到的醫療影像，只是讓外洩的時候損失更大。期限到了
    就刪，不通知、不進回收桶——「還留著但看不到」不是刪除。
    箱子的狀態（結果、密碼盤）不跟著刪：那裡面沒有影像，留著使用者回頭
    還看得到自己做過什麼。
    """
    if RETAIN_DAYS <= 0:
        return 0
    cut = int((now or time.time()) - RETAIN_DAYS * 86400)
    with auth.connect() as con:
        rows = list(con.execute(
            "SELECT id, owner FROM vault_items WHERE at < ?", (cut,)))
        if rows:
            con.execute("DELETE FROM vault_items WHERE at < ?", (cut,))
    for r in rows:
        _path(r["owner"], r["id"]).unlink(missing_ok=True)
    return len(rows)


def expires_at(owner: str) -> dict[str, int]:
    """每一箱最早那張的到期時間，前端要提醒使用者就靠這個。"""
    if RETAIN_DAYS <= 0:
        return {}
    with auth.connect() as con:
        return {str(r["slot"]): r["at"] + RETAIN_DAYS * 86400
                for r in con.execute(
                    "SELECT slot, min(at) at FROM vault_items "
                    "WHERE owner=? GROUP BY slot", (owner,))}


# ── 金鑰輪替 ──────────────────────────────────────────────────────────────

def rotate(progress=None) -> dict:
    """把所有影像改用現在這把金鑰重新加密。

    做法是「解開再加密」，不是換個標籤——舊鑰從此對這些檔案無效。
    寫檔先寫暫存再 rename（同一個檔案系統上 rename 是原子操作），
    中途斷電不會留下半個檔案。

    可以重複執行：已經是現用金鑰的跳過。所以做到一半掛掉，補跑就好，
    前提是舊鑰還留在 BCRR_DATA_KEY_OLD 裡。
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    cur = _key()
    cur_fp = fingerprint(cur)
    keys = _keys()
    done = skipped = failed = 0

    with auth.connect() as con:
        rows = list(con.execute("SELECT id, owner, keyid FROM vault_items"))

    for r in rows:
        if r["keyid"] == cur_fp:
            skipped += 1
            continue
        p = _path(r["owner"], r["id"])
        if not p.exists():
            failed += 1
            continue
        blob = p.read_bytes()
        raw = None
        order = ([keys[r["keyid"]]] if r["keyid"] in keys else list(keys.values()))
        for k in order:
            try:
                raw = AESGCM(k).decrypt(blob[:12], blob[12:], r["id"].encode())
                break
            except Exception:                                      # noqa: BLE001
                continue
        if raw is None:
            failed += 1
            if progress:
                progress(f"解不開 {r['id'][:8]}（舊鑰沒帶到？）")
            continue

        nonce = os.urandom(12)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(nonce + AESGCM(cur).encrypt(nonce, raw, r["id"].encode()))
        os.chmod(tmp, 0o600)
        tmp.replace(p)
        with auth.connect() as con:
            con.execute("UPDATE vault_items SET keyid=? WHERE id=?", (cur_fp, r["id"]))
        done += 1
        if progress and done % 20 == 0:
            progress(f"已換 {done} 張")

    return {"total": len(rows), "rotated": done, "already": skipped, "failed": failed,
            "keyid": cur_fp}


# ── 使用者的資料金鑰（伺服器只保管鎖住的盒子）────────────────────────────
#
# 這裡面沒有一個欄位能拿來解密。salt 本來就不是秘密；wrap_pass 與 wrap_rec
# 是同一把資料金鑰被兩把不同的鑰匙鎖起來的結果，而那兩把鑰匙從來沒有離開
# 過使用者的瀏覽器。伺服器拿這三樣東西推不出任何東西，這就是重點。
#
# 為什麼要多一層「資料金鑰」而不是直接用密語加密影像：換密語的時候只要把
# 小盒子重鎖一次，幾百張影像完全不用動。

def key_get(owner: str) -> dict | None:
    with auth.connect() as con:
        r = con.execute("SELECT salt, wrap_pass, wrap_rec, at, changed "
                        "FROM vault_keys WHERE owner=?", (owner,)).fetchone()
    return dict(r) if r else None


def key_set(owner: str, salt: str, wrap_pass: str, wrap_rec: str) -> None:
    """第一次設定。已經有了就不覆蓋——覆蓋等於把使用者鎖在門外。"""
    if key_get(owner):
        raise ValueError("這個帳號已經設定過通關密語了")
    with auth.connect() as con:
        con.execute("INSERT INTO vault_keys(owner, salt, wrap_pass, wrap_rec, at) "
                    "VALUES(?,?,?,?,?)",
                    (owner, salt, wrap_pass, wrap_rec, int(time.time())))


def key_rewrap(owner: str, salt: str, wrap_pass: str) -> None:
    """換密語：只換密語那份包裝，救援碼那份不動，影像一張都不用重新加密。"""
    with auth.connect() as con:
        n = con.execute("UPDATE vault_keys SET salt=?, wrap_pass=?, changed=? "
                        "WHERE owner=?",
                        (salt, wrap_pass, int(time.time()), owner)).rowcount
    if not n:
        raise ValueError("這個帳號還沒設定過通關密語")
