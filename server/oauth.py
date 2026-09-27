"""用 Google 帳號登入。

只做 OpenID Connect 的授權碼流程，沒有用任何 OAuth 套件——標準庫的
urllib 就夠了，理由跟 auth.py 一樣：這套要能在診所的機器上離線安裝，
每多一個相依就多一個裝不起來的理由。

兩個安全細節值得寫下來：

  * state 與 PKCE 都做了。state 擋的是 CSRF（有人把他自己的授權碼餵給
    你的瀏覽器，讓你以他的身分登入）；PKCE 擋的是授權碼在轉址途中被
    攔走（瀏覽器歷程、Referer、代理伺服器的日誌都看得到網址）。兩者
    的隨機值都放在簽章過的短效 cookie 裡，伺服器本身不存狀態。

  * id_token 沒有驗簽。這不是偷懶：token 是我們自己從 Google 的 token
    端點、走 TLS、用 client_secret 取回來的，不是從瀏覽器轉址帶進來的，
    OpenID Connect 規格（3.1.3.7）明文說這種情況可以不驗簽。仍然要檢查
    iss / aud / exp，否則拿別的專案的 token 也能進來。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
import urllib.parse
import urllib.request

from itsdangerous import BadSignature, SignatureExpired, TimestampSigner

import auth

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
ISSUERS = ("https://accounts.google.com", "accounts.google.com")
SCOPES = "openid email profile"

CLIENT_ID = os.environ.get("BCRR_GOOGLE_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("BCRR_GOOGLE_CLIENT_SECRET", "")

FLOW_COOKIE = "bcrr_flow"
FLOW_SECONDS = 600          # 十分鐘內沒走完就重來


def enabled() -> bool:
    return bool(CLIENT_ID and CLIENT_SECRET)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def start(redirect_uri: str, secret: bytes, next_url: str = "/vault.html") -> tuple[str, str]:
    """回傳（要把瀏覽器送去的網址, 要種進 cookie 的流程票根）。"""
    nonce = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(48)
    challenge = _b64(hashlib.sha256(verifier.encode()).digest())

    # 先 base64 再簽：JSON 裡的空白、逗號、大括號都不是合法的 cookie 字元，
    # 直接簽會被加上引號跳脫，換個瀏覽器就可能對不回來。
    payload = _b64(json.dumps({"n": nonce, "v": verifier, "r": redirect_uri,
                               "next": next_url}, separators=(",", ":")).encode())
    ticket = auth._signer().sign(payload.encode()).decode()

    query = urllib.parse.urlencode({
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "state": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        # 每次都問要用哪個帳號：共用電腦上最怕的就是默默沿用上一個人的登入
        "prompt": "select_account",
    })
    return f"{AUTH_URL}?{query}", ticket


def finish(code: str, state: str, ticket: str | None, secret: bytes) -> dict:
    """驗完 state、換 token、拆出身分。任何一關不過就丟 ValueError。"""
    # 三種失敗分開報，否則下次出事還是分不出是哪一種。
    # 前綴的代碼是給我們看的，使用者看得懂後面那句就夠。
    if not ticket:
        raise ValueError("NOCOOKIE｜瀏覽器沒有把登入票根送回來，請再按一次登入")
    try:
        raw = auth._signer().unsign(ticket, max_age=FLOW_SECONDS)
    except SignatureExpired:
        raise ValueError("EXPIRED｜登入流程超過十分鐘，請再按一次登入") from None
    except BadSignature:
        raise ValueError("BADSIG｜登入票根對不上，請再按一次登入") from None

    flow = json.loads(_unb64(raw.decode()))
    if not secrets.compare_digest(flow["n"], state or ""):
        raise ValueError("STATE｜登入流程校驗失敗，請再按一次登入")

    body = urllib.parse.urlencode({
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": flow["r"],
        "grant_type": "authorization_code",
        "code_verifier": flow["v"],
    }).encode()
    req = urllib.request.Request(TOKEN_URL, data=body,
                                 headers={"Content-Type":
                                          "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            tok = json.loads(resp.read())
    except Exception as exc:                                       # noqa: BLE001
        raise ValueError(f"向 Google 換取憑證失敗：{exc}") from exc

    claims = json.loads(_unb64(tok["id_token"].split(".")[1]))

    if claims.get("iss") not in ISSUERS:
        raise ValueError("憑證來源不正確")
    if claims.get("aud") != CLIENT_ID:
        raise ValueError("憑證不是發給這個網站的")
    if float(claims.get("exp", 0)) < time.time():
        raise ValueError("憑證已過期")
    if not claims.get("email_verified"):
        raise ValueError("這個 Google 帳號的信箱尚未驗證")

    return {"email": claims["email"].lower(),
            "name": claims.get("name") or claims["email"].split("@")[0],
            "next": flow.get("next", "/vault.html")}
