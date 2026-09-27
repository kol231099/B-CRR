"""帳號、密碼與連線階段。

不依賴任何外部套件：雜湊用標準庫的 hashlib.scrypt（記憶體困難型 KDF），
簽章 cookie 用 itsdangerous（純 Python、無編譯相依）。理由是這套系統要能在
診所的電腦上離線安裝，相依愈少愈好。

刻意沒有做的事，以及原因：
  * 沒有註冊頁面——帳號由管理者用 server/users.py 建立。醫療系統不該讓
    任何人自己開帳號。
  * 沒有「忘記密碼」寄信——離線環境沒有郵件伺服器，改由管理者重設。
  * 不存 JWT，改用簽章 cookie。要讓某個帳號立刻失效時，停用帳號即可，
    不必等 token 過期。
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import time
from pathlib import Path

import hmac

from itsdangerous import BadSignature, SignatureExpired, TimestampSigner

DB_PATH = Path(os.environ.get("BCRR_DB",
                              Path(__file__).resolve().parent / "bcrr.sqlite3"))
COOKIE = "bcrr_session"
IDLE_SECONDS = int(os.environ.get("BCRR_IDLE", "900"))      # 閒置 15 分鐘登出
ROLES = ("admin", "doctor", "assistant")

# scrypt 參數：n 愈大愈慢也愈難暴力破解。2**14 在一般筆電約 0.1 秒，
# 對使用者無感，對攻擊者是每次嘗試都要付的成本。
_N, _R, _P, _DKLEN = 2 ** 14, 8, 1, 32


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init_db() -> None:
    with connect() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                username    TEXT PRIMARY KEY,
                pw_hash     TEXT NOT NULL,
                salt        TEXT NOT NULL,
                role        TEXT NOT NULL DEFAULT 'doctor',
                disabled    INTEGER NOT NULL DEFAULT 0,
                nda_at      INTEGER,
                created_at  INTEGER NOT NULL,
                last_login  INTEGER
            );
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                at       INTEGER NOT NULL,
                username TEXT,
                action   TEXT NOT NULL,
                detail   TEXT
            );
        """)
    # 後加的欄位用 ALTER 補，舊資料庫升級不必重建
    with connect() as con:
        have = {r["name"] for r in con.execute("PRAGMA table_info(users)")}
        if "auth_kind" not in have:
            con.execute("ALTER TABLE users ADD COLUMN auth_kind TEXT "
                        "NOT NULL DEFAULT 'password'")
        if "display" not in have:
            con.execute("ALTER TABLE users ADD COLUMN display TEXT")

    # 檔案權限收緊：同一台機器的其他使用者不該讀得到雜湊
    try:
        os.chmod(DB_PATH, 0o600)
    except OSError:
        pass


def _secret() -> bytes:
    """簽章金鑰存在資料庫裡，服務重啟後大家不用重新登入。"""
    with connect() as con:
        row = con.execute("SELECT value FROM meta WHERE key='secret'").fetchone()
        if row:
            return bytes.fromhex(row["value"])
        key = secrets.token_bytes(32)
        con.execute("INSERT INTO meta(key, value) VALUES('secret', ?)", (key.hex(),))
        return key


def secret() -> bytes:
    return _secret()


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    h = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt),
                       n=_N, r=_R, p=_P, dklen=_DKLEN)
    return h.hex(), salt


def verify_password(password: str, pw_hash: str, salt: str) -> bool:
    calc, _ = hash_password(password, salt)
    return secrets.compare_digest(calc, pw_hash)      # 定時比較，避免時序側通道


def log(username: str | None, action: str, detail: str = "") -> None:
    """稽核：誰、什麼時候、做了什麼。不記錄影像內容與檔名以外的資訊。"""
    with connect() as con:
        con.execute("INSERT INTO audit(at, username, action, detail) VALUES(?,?,?,?)",
                    (int(time.time()), username, action, detail[:200]))


def authenticate(username: str, password: str) -> sqlite3.Row | None:
    with connect() as con:
        u = con.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if not u or u["disabled"]:
        return None
    if not verify_password(password, u["pw_hash"], u["salt"]):
        return None
    with connect() as con:
        con.execute("UPDATE users SET last_login=? WHERE username=?",
                    (int(time.time()), username))
    return u


def _signer() -> TimestampSigner:
    """itsdangerous 預設是 HMAC-SHA1。HMAC-SHA1 至今沒有實用攻擊，但沒有理由
    留著一個要向別人解釋的東西——換成 SHA-256 只是換一個參數。"""
    return TimestampSigner(_secret(), digest_method=hashlib.sha256)


def issue(username: str) -> str:
    return _signer().sign(username.encode()).decode()


def read(token: str | None) -> str | None:
    """驗簽並檢查閒置時間。過期或被竄改一律當成未登入。"""
    if not token:
        return None
    try:
        raw = _signer().unsign(token, max_age=IDLE_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
    name = raw.decode()
    with connect() as con:
        u = con.execute("SELECT disabled FROM users WHERE username=?", (name,)).fetchone()
    return None if (not u or u["disabled"]) else name


def get_user(username: str) -> sqlite3.Row | None:
    with connect() as con:
        return con.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()


def accept_nda(username: str) -> None:
    with connect() as con:
        con.execute("UPDATE users SET nda_at=? WHERE username=?",
                    (int(time.time()), username))


# ── Google 帳號 ────────────────────────────────────────────────────────────
#
# OAuth 只回答「你是誰」，不回答「你可不可以進來」。任何人都能拿 Google 帳號
# 走完登入流程，所以白名單是必要的，不是多此一舉。名單放環境變數而不是資料庫，
# 是因為加一個評審或口試委員時，改一行設定再重啟比開資料庫快，也留在部署紀錄裡。

def allowlist() -> dict[str, str]:
    """BCRR_ALLOW="a@x.com:admin, b@y.com" → {email: role}，沒寫角色的給 doctor。

    單獨一個 "*" 代表「任何 Google 帳號都放行」，可以跟指定的信箱並存：
        BCRR_ALLOW="me@x.com:admin,*"   我是管理者，其他人以一般身分進來
    開放之後擋不住的是運算資源，不是資料——伺服器一張影像都不存，
    每個人只看得到自己上傳的東西。所以開放的前提是流量限制要在（見 app.py）。
    """
    out: dict[str, str] = {}
    for item in os.environ.get("BCRR_ALLOW", "").split(","):
        item = item.strip()
        if not item:
            continue
        email, _, role = item.partition(":")
        role = role.strip() or "doctor"
        out[email.strip().lower()] = role if role in ROLES else "doctor"
    return out


def google_login(email: str, display: str) -> sqlite3.Row | None:
    """名單內就登入（第一次順手建帳號），名單外一律擋掉。"""
    allow = allowlist()
    if email in allow:
        role = allow[email]
    elif "*" in allow:
        role = "doctor"
    else:
        log(email, "denied", "不在白名單")
        return None

    now = int(time.time())
    with connect() as con:
        u = con.execute("SELECT * FROM users WHERE username=?", (email,)).fetchone()
        if u is None:
            con.execute(
                "INSERT INTO users(username, pw_hash, salt, role, created_at, "
                "last_login, auth_kind, display) VALUES(?,?,?,?,?,?,?,?)",
                (email, "", "", role, now, now, "google", display))
        elif u["disabled"]:
            # 停用是管理者的明確決定，白名單不該把它蓋回去
            log(email, "denied", "帳號已停用")
            return None
        else:
            con.execute("UPDATE users SET last_login=?, display=?, role=? "
                        "WHERE username=?", (now, display, role, email))

    log(email, "login", "google")
    return get_user(email)
