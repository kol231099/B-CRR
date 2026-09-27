"""B-CRR 推論 API：收一張根尖片，回傳每顆牙的幾何與三個指標。

回的是**座標**不是圖片——旋轉框、遮罩輪廓、主軸、七個特徵點、各高度線，
前端拿去用 SVG 疊在原圖上畫，線條才會銳利，也才跟 Demo 那頁同一套畫法。

重點取捨都寫在對應的函式裡：
  * 偵測門檻提前到 roi_heads，輸出完全一樣但少算九成遮罩（見 load_detector）
  * 長邊超過 MAX_SIDE 先縮，比值本身與尺度無關（見 read_image）
  * 有些牙量不出來是正常的，逐顆回報而不是整張失敗（見 analyse）

環境變數：
  BCRR_REPO    專案根目錄，預設是這支的上一層
  BCRR_FOLDS   第二階段要集成幾折，1–5，預設 5
  BCRR_MAXSIDE 影像長邊上限，預設 2400
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from urllib.parse import quote

from starlette.concurrency import run_in_threadpool
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     Response, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

REPO = Path(os.environ.get("BCRR_REPO", Path(__file__).resolve().parent.parent))
FOLDS = max(1, min(5, int(os.environ.get("BCRR_FOLDS", "5"))))
MAX_SIDE = int(os.environ.get("BCRR_MAXSIDE", "2400"))
SITE = os.environ.get("BCRR_SITE", "")          # 有給就順便把網站端出來
MAX_BYTES = 24 * 1024 * 1024
# 預設要登入才進得了保險庫。本機開發想略過就設 BCRR_OPEN=1——
# 刻意做成「明講才放行」，忘了設定的後果是進不去，不是全世界都進得去。
OPEN = os.environ.get("BCRR_OPEN", "") == "1"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "koi" / "setA" / "scripts"))
# 伺服器上是從 repo 根目錄以 server.app:app 啟動的，本機是在 server/ 裡面跑，
# 兩種情況下 auth/oauth 都要找得到
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 這幾支都不牽到 matplotlib，伺服器上不用裝繪圖套件
from make_crops_obb import obb_of, warp_of                 # noqa: E402
from postprocess import clean_mask                         # noqa: E402
from train_maskrcnn import CKPT, ROOT, build_model         # noqa: E402
from train_seg2 import SIZE, build_seg2, split_tag               # noqa: E402

from scripts.find_axis_raw import fit_axis, mask_points     # noqa: E402
from scripts.find_cej import fit_side, prepare_side         # noqa: E402
from scripts.measure import mask_points_of, measure         # noqa: E402

import auth                                                # noqa: E402
import oauth                                               # noqa: E402
import vault                                               # noqa: E402

SEG_TAG = "unet_tu-hrnet_w32"
DET_THR, PAD = 0.35, 0.2
DEFAULTS = dict(drop_apical=0.25, cut_at=1 / 3, threshold=0.0,
                surface_fraction=0.10, inner_fraction=0.03,
                outer_fraction=0.15, root_fraction=2 / 3)

def pick_device() -> "torch.device":
    """BCRR_DEVICE 指定要用哪個裝置，預設 cpu。

    伺服器沒有顯卡，維持 cpu；比賽現場在 MacBook 上跑就設 mps，實測快三倍多，
    而且量測數字跟 cpu 一致（見 server/check_device.py）。指定的裝置不存在時
    退回 cpu 並留一行警告——現場沒有比「因為打錯字所以整個跑不起來」更糟的事。
    """
    want = os.environ.get("BCRR_DEVICE", "cpu").strip().lower()
    if want in ("", "cpu"):
        return torch.device("cpu")
    if want == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        log.warning("這台機器沒有 Metal GPU，改用 cpu")
    elif want == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        log.warning("找不到 CUDA 裝置，改用 cpu")
    else:
        log.warning("BCRR_DEVICE=%s 不認得，改用 cpu", want)
    return torch.device("cpu")


log = logging.getLogger("bcrr")
torch.set_num_threads(int(os.environ.get("BCRR_THREADS", "0")) or torch.get_num_threads())

DEVICE = pick_device()
MODELS: dict = {}
LOCK = threading.Lock()          # 一次只跑一張：峰值記憶體是這台機器的瓶頸

# ── 流量限制 ──────────────────────────────────────────────────────────────
#
# 一張圖在這台機器上要二十幾秒，而且 LOCK 讓它們一張一張跑。沒有這一段的話，
# 十個人同時點，第十個要等四分鐘，服務等於癱瘓——開放任何人登入之後這不是
# 假設性的風險。兩個限制管的是不同的事：
#   佇列   保護機器：排隊的人超過上限就直接回絕，不要讓人乾等
#   配額   保護大家：一個帳號不該把整台機器包走
QUEUE_MAX = int(os.environ.get("BCRR_QUEUE", "4"))
PER_HOUR = int(os.environ.get("BCRR_PER_HOUR", "30"))

_gate = threading.Lock()
_waiting = 0
_recent: dict[str, list[float]] = {}


def take_slot(user: str) -> None:
    global _waiting
    now = time.time()
    with _gate:
        # 順手清掉不再有人用的紀錄，不然這個 dict 會隨著造訪人數一直長大
        if len(_recent) > 512:
            for k in [k for k, v in _recent.items() if not v or now - v[-1] > 3600]:
                del _recent[k]
        hits = [t for t in _recent.get(user, []) if now - t < 3600]
        if len(hits) >= PER_HOUR:
            raise HTTPException(429, f"一小時最多分析 {PER_HOUR} 張，請稍後再試")
        if _waiting >= QUEUE_MAX:
            raise HTTPException(503, "現在排隊的人太多，請過一分鐘再試")
        hits.append(now)
        _recent[user] = hits
        _waiting += 1


def release_slot() -> None:
    global _waiting
    with _gate:
        _waiting = max(0, _waiting - 1)


def load_detector():
    ck = torch.load(CKPT / "original" / "maskrcnn_final.pt",
                    map_location="cpu", weights_only=False)
    m = build_model(False, ck.get("mask_res", 28))
    m.load_state_dict(ck["model"])
    # torchvision 預設 score_thresh=0.05、detections_per_img=100，mask head 會對
    # 這一百個候選各算一張全圖大小的遮罩。我們本來就只留 >= DET_THR 的，把門檻
    # 提前等於同樣的輸出、少算九成——實測峰值從 3.2 GB 降到 2.4 GB，框完全一致。
    m.roi_heads.score_thresh = DET_THR
    m.roi_heads.detections_per_img = 12
    m.eval()
    return m.to(DEVICE)


def load_segmenters(n: int):
    """n 折集成。測試集不在任何一折的訓練資料裡，集成是合法的。"""
    arch, enc = split_tag(SEG_TAG)
    out = []
    for f in range(n):
        m = build_seg2(arch, enc, pretrained=False)
        m.load_state_dict(torch.load(
            ROOT / "checkpoints_obb" / "seg2" / SEG_TAG / f"fold{f}.pt",
            map_location="cpu", weights_only=False)["model"])
        m.eval()
        out.append(m.to(DEVICE))
    return out


@torch.no_grad()
def detect_obb(det, gray: np.ndarray) -> list[tuple]:
    """回傳 [((cx, cy, 短邊, 長邊, 角度), 信心度), ...]，依信心度排序。

    信心度是 Mask R-CNN 對「這塊是牙齒」的分數，之前沒有帶出來。它不是對量測
    結果的信心，但它是這條管線唯一真的有的信心值，報告書上那一圈就用它。
    """
    t = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1).to(DEVICE)
    out = det([t])[0]
    scores = out["scores"].cpu().numpy()
    keep = scores >= DET_THR
    masks = out["masks"].cpu().numpy()[keep, 0] > 0.5
    found = []
    for m, sc in zip(masks, scores[keep]):
        m8 = clean_mask(m).astype(np.uint8)
        if m8.any():
            found.append((obb_of(m8), float(sc)))
    return found


@torch.no_grad()
def seg_prob(segs, crop: np.ndarray) -> np.ndarray:
    """把 crop 丟進 n 折模型，取平均機率圖。

    這是 eval_seg2_holdout.predict 的同義寫法，唯一差別是張量會送到 DEVICE。
    沒有直接改那支，因為訓練與評估腳本都在用它，不該為了伺服器動研究端的程式。
    它的 TTA 分支這裡沒搬過來——app.py 一律不做 TTA。
    """
    base = cv2.resize(crop, SIZE[::-1], interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(np.ascontiguousarray(base)).float().div(255)
    t = t.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
    return np.mean([torch.sigmoid(m(t))[0, 0].cpu().numpy() for m in segs], axis=0)


@torch.no_grad()
def segment_in_obb(segs, gray: np.ndarray, box: tuple) -> np.ndarray:
    """把 OBB 轉正成 crop，送進 HRNet-w32，再把遮罩貼回原圖座標。"""
    h, w = gray.shape
    M, cw, ch = warp_of(*box, PAD)
    crop = cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR)
    prob = seg_prob(segs, crop)
    prob = cv2.resize(prob, (cw, ch), interpolation=cv2.INTER_LINEAR)
    back = cv2.warpAffine(prob, cv2.invertAffineTransform(M), (w, h),
                          flags=cv2.INTER_LINEAR) > 0.5
    return clean_mask(back).astype(np.uint8)


def contour_of(mask: np.ndarray, eps: float = 1.2) -> list:
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    c = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), eps, True)
    return [[round(float(p[0][0]), 1), round(float(p[0][1]), 1)] for p in c]


def geometry(img: np.ndarray, mask: np.ndarray, box) -> dict:
    """一顆牙的全部幾何，座標一律是（縮放後的）原圖像素。

    與 koi/setA/scripts/export_demo.py 同一份輸出格式，前端兩邊共用同一套畫法。
    """
    r = measure("api", img, mask, **DEFAULTS)
    frame, slope = r.axis.frame, r.axis.cd_slope
    xs, ys = frame.to_frame(mask_points_of(mask))
    half = float(np.abs(xs).max()) * 1.30

    def xy(p):
        return [round(float(p[0]), 1), round(float(p[1]), 1)]

    def level(key):
        v = r.levels[key]
        ends = frame.to_image(np.array([-half, half]),
                              np.array([v - half * slope, v + half * slope]))
        return {"p": xy(frame.point_at(v)), "line": [xy(ends[0]), xy(ends[1])]}

    quad = cv2.boxPoints(((box[0], box[1]), (box[2], box[3]), box[4]))
    return {
        "obb": [xy(p) for p in quad],
        "poly": contour_of(mask),
        "axis": [xy(frame.point_at(float(ys.min()))),
                 xy(frame.point_at(float(ys.max())))],
        "pts": {k: xy(v) for k, v in r.landmarks.items()},
        "lv": {k: level(k) for k in ("H", "I", "J", "K", "L", "R", "Q", "S")},
        "crr": round(r.crr, 4),
        "ablr": round(r.ablr, 4),
        "bcrr": round(r.b_crr, 4),
        "max_blr": round(r.max_blr, 4),
    }


def side_snr(y: np.ndarray, w: np.ndarray) -> float | None:
    """單側的鉸鏈訊雜比：牙冠外擴高出牙根基準線幾個雜訊標準差。

    分母用 fit_side 算出來的 noise（牙根段殘差的標準差），不是牙齒長度。
    絕對像素除以牙長等於拿絕對值比絕對值：一顆外擴幅度不大、但基準線非常
    平整的牙，CEJ 其實定得很準，用長度當分母卻會被判低分。
    """
    try:
        f = fit_side(y, w)
        noise, flare = f.noise, float(f.deviation.max())
    except Exception:                                              # noqa: BLE001
        # 這一側連鉸鏈都解不出來（量不出指標的牙就是卡在這）。仍然給個數字，
        # 用牙根半段自己擬一條基準線，算法與 fit_side 第一輪相同。
        root = y <= y.min() + 0.5 * np.ptp(y)
        if root.sum() < 5:
            return None
        design = np.column_stack([np.ones(int(root.sum())), y[root]])
        (a, b), *_ = np.linalg.lstsq(design, w[root], rcond=None)
        dev = w - (a + b * y)
        noise, flare = float(np.std(dev[root])), float(dev.max())
    if noise <= 0:
        return None
    return flare / noise


def confidence(mask: np.ndarray) -> float | None:
    """這顆牙的量測可信度。

    CEJ 要 C、D 兩側都定得出來才有 CD 線，所以取**較弱那一側**的訊雜比。
    再過一條飽和曲線壓到 0–1；r0 = 8 是由測試集 60 顆牙的中位數（11.4）
    反推的，量得出的牙落在 19–100（中位數 76、四分位 59–86）。
    """
    try:
        pts = mask_points(mask)
        frame = fit_axis(pts)
        weak = None
        for side in (-1, +1):
            y, w = prepare_side(pts, frame, side, DEFAULTS["drop_apical"])
            snr = side_snr(y, w)
            if snr is None:
                return None
            weak = snr if weak is None else min(weak, snr)
        return round(float(1 - np.exp(-weak / 8.0)), 4)
    except Exception:                                              # noqa: BLE001
        return None


def read_image(raw: bytes) -> tuple[np.ndarray, float]:
    """解成灰階，長邊超過 MAX_SIDE 就縮，回傳 (影像, 相對原檔的縮放比)。

    縮圖不影響三個指標——它們都是同一張圖上的長度比值，尺度會約掉。縮的是
    第二階段的 warp 與量測成本；第一階段的 torchvision 本來就會自己縮到
    min_size=800 / max_size=1333。
    """
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(415, "這個檔案解不出影像")
    h, w = img.shape
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        return img, s
    return img, 1.0


def analyse(img: np.ndarray) -> list[dict]:
    """逐顆處理。有些牙的 CEJ 鉸鏈無解、量不出指標，這是正常情況，
    回報成那一顆的 ok=false，不要讓整張圖失敗。"""
    det, segs = MODELS["det"], MODELS["segs"]
    teeth = []
    for box, score in sorted(detect_obb(det, img), key=lambda bs: bs[0][0]):
        mask = segment_in_obb(segs, img, box)
        if not mask.any():
            continue
        quad = cv2.boxPoints(((box[0], box[1]), (box[2], box[3]), box[4]))
        try:
            t = geometry(img, mask, box)
            t["ok"] = True
        except Exception as exc:                                   # noqa: BLE001
            t = {"ok": False,
                 "why": str(exc) or exc.__class__.__name__,
                 "obb": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in quad],
                 "poly": contour_of(mask)}
        t["score"] = round(score, 4)
        t["conf"] = confidence(mask)
        teeth.append(t)
    return teeth


app = FastAPI(title="B-CRR", docs_url=None, redoc_url=None)
# 白名單而不是 "*"。目前沒開 allow_credentials，所以就算放寬也偷不走登入狀態
# （瀏覽器不讓跨來源請求帶 cookie），但沒有理由讓任何網站都能借用這台的算力。
ORIGINS = [o.strip() for o in os.environ.get(
    "BCRR_ORIGINS",
    "https://pcastandardllm.online,https://www.pcastandardllm.online,"
    "http://127.0.0.1:8900,http://localhost:8900").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS,
                   allow_methods=["POST", "GET", "PUT", "DELETE", "OPTIONS"],
                   allow_headers=["*"])

# ── 登入 ──────────────────────────────────────────────────────────────────

def base_url(request: Request) -> str:
    """還原對外網址。

    服務跑在反向代理後面，自己看到的是 http://127.0.0.1:8900，但 Google 的
    轉址網址必須跟主控台上註冊的那條一字不差，所以要靠轉發標頭還原。偽造
    這些標頭沒有好處：網址對不上註冊清單，Google 那關就先擋掉了。
    """
    fixed = os.environ.get("BCRR_BASE_URL", "").rstrip("/")
    if fixed:
        return fixed
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = (request.headers.get("x-forwarded-host")
            or request.headers.get("host") or request.url.netloc)
    return f"{proto}://{host}"


def _cookie_kw(request: Request) -> dict:
    # SameSite=Lax：從 Google 轉址回來是頂層 GET，這個等級過得去；
    # Strict 會讓剛登入完的那一跳讀不到 cookie，變成登入不會成功。
    return dict(httponly=True, samesite="lax",
                secure=base_url(request).startswith("https"), path="/")


def who(request: Request) -> str | None:
    return auth.read(request.cookies.get(auth.COOKIE))


def require(request: Request) -> str:
    """API 的守門。沒登入回 401、沒簽保密協定回 403，前端據此把人帶去對的頁。"""
    if OPEN:
        return "open"
    name = who(request)
    if not name:
        raise HTTPException(401, "請先登入")
    u = auth.get_user(name)
    if u is None or not u["nda_at"]:
        raise HTTPException(403, "請先閱讀並同意保密協定")
    return name


@app.get("/api/auth/me")
def auth_me(request: Request):
    if OPEN:
        return {"login": True, "open": True, "email": "", "name": "本機模式",
                "role": "admin", "nda": True}
    name = who(request)
    if not name:
        return JSONResponse({"login": False, "enabled": oauth.enabled()},
                            status_code=401)
    u = auth.get_user(name)
    return {"login": True, "open": False, "email": u["username"],
            "name": u["display"] or u["username"], "role": u["role"],
            "nda": bool(u["nda_at"])}


@app.get("/api/auth/login")
def auth_login(request: Request, next: str = "/vault.html"):
    if not oauth.enabled():
        raise HTTPException(503, "這台伺服器尚未設定 Google 登入")
    # 只准導回站內。少了這一行，別人就能拿我們的網址把人騙去釣魚站。
    if not next.startswith("/") or next.startswith("//"):
        next = "/vault.html"
    url, ticket = oauth.start(base_url(request) + "/api/auth/callback",
                              auth.secret(), next)
    resp = RedirectResponse(url, status_code=303)
    resp.set_cookie(oauth.FLOW_COOKIE, ticket, max_age=oauth.FLOW_SECONDS,
                    **_cookie_kw(request))
    return resp


@app.get("/api/auth/callback")
def auth_callback(request: Request, code: str = "", state: str = "",
                  error: str = ""):
    def back(msg: str):
        return RedirectResponse("/login.html?err=" + quote(msg), status_code=303)

    if error:
        return back(error)
    try:
        ident = oauth.finish(code, state, request.cookies.get(oauth.FLOW_COOKIE),
                             auth.secret())
    except (ValueError, KeyError) as exc:
        # 失敗也要留紀錄，不然只能靠使用者回報畫面上寫什麼
        msg = str(exc) or "登入失敗"
        code, _, human = msg.partition("｜")
        auth.log(None, "login-fail", code if human else msg[:40])
        return back(human or msg)

    u = auth.google_login(ident["email"], ident["name"])
    if u is None:
        # 訊息刻意含糊：不告訴對方「這個信箱存在但被停用」之類的線索
        return back("這個帳號沒有使用權限")

    resp = RedirectResponse(ident["next"] if u["nda_at"] else "/nda.html",
                            status_code=303)
    resp.set_cookie(auth.COOKIE, auth.issue(u["username"]),
                    max_age=auth.IDLE_SECONDS, **_cookie_kw(request))
    resp.delete_cookie(oauth.FLOW_COOKIE, path="/")
    return resp


@app.post("/api/auth/logout")
def auth_logout(request: Request):
    name = who(request)
    if name:
        auth.log(name, "logout")
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(auth.COOKIE, path="/")
    return resp


@app.post("/api/auth/nda")
def auth_nda(request: Request):
    name = who(request)
    if not name:
        raise HTTPException(401, "請先登入")
    auth.accept_nda(name)
    auth.log(name, "nda", "同意保密協定")
    return {"ok": True}


@app.middleware("http")
async def gate(request, call_next):
    """保險庫要登入才進得去；首頁與 Demo 是公開展示，不擋。

    順便做滑動式閒置逾時：cookie 的有效期是從簽發那刻算的，每次有動作就
    重簽一次，使用者才不會做到一半被踢出去；真的離開 15 分鐘就會失效。
    """
    name = None if OPEN else who(request)
    if not OPEN and request.url.path in ("/vault.html", "/vault"):
        if not name:
            return RedirectResponse("/login.html", status_code=303)
        u = auth.get_user(name)
        if u is None or not u["nda_at"]:
            return RedirectResponse("/nda.html", status_code=303)

    resp = await call_next(request)
    if name:
        resp.set_cookie(auth.COOKIE, auth.issue(name),
                        max_age=auth.IDLE_SECONDS, **_cookie_kw(request))
    return resp



@app.middleware("http")
async def no_cache_html(request, call_next):
    """網頁本身一律不給快取。

    改完 vault.html 之後瀏覽器還拿舊的那份，會看到「明明改了卻沒生效」，
    很難跟真的 bug 分辨。圖片與影片有 ?v= 印記，不受影響。
    """
    resp = await call_next(request)
    path = request.url.path
    if path.endswith((".html", "/")) or path == "":
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@app.middleware("http")
async def allow_private_network(request, call_next):
    """讓公開網域的頁面也能打到這支跑在 loopback 的服務。

    Chromium 的 Private Network Access 會擋掉「公開來源 → 私有位址」的請求
    （net::ERR_BLOCKED_BY_CLIENT），除非目標在 preflight 明講允許。Safari 目前
    沒有這條限制。這個標頭只是把選擇權交出去，對同源存取沒有影響。
    """
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    return resp


async def sweeper() -> None:
    while True:
        try:
            n = await run_in_threadpool(vault.sweep)
            if n:
                log.info("清掉 %d 張超過 %d 天的影像", n, vault.RETAIN_DAYS)
        except Exception as exc:                                   # noqa: BLE001
            log.warning("過期清理失敗：%s", exc)
        await asyncio.sleep(6 * 3600)


@app.on_event("startup")
async def warm() -> None:
    auth.init_db()
    vault.init_db()
    if vault.ready() and vault.RETAIN_DAYS > 0:
        asyncio.create_task(sweeper())
    if not OPEN and not oauth.enabled():
        log.warning("尚未設定 Google 登入，目前沒有人進得了保險庫")
    t0 = time.time()
    MODELS["det"] = load_detector()
    MODELS["segs"] = load_segmenters(FOLDS)
    log.info("模型載入完成 %.1fs（第二階段 %d 折）", time.time() - t0, FOLDS)


@app.get("/api/health")
def health() -> dict:
    return {"ok": bool(MODELS), "folds": FOLDS, "max_side": MAX_SIDE,
            "threads": torch.get_num_threads()}


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...),
                      user: str = Depends(require)) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "沒有收到檔案")
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"檔案超過 {MAX_BYTES // 1024 // 1024} MB")

    img, scale = read_image(raw)
    if not OPEN:                       # 本機模式沒有別人要排隊
        take_slot(user)
    t0 = time.time()
    # 推論是同步的 CPU 工作，直接寫在 async 函式裡會把事件迴圈整個佔住——
    # 那二十幾秒內連首頁和健康檢查都不會有回應。丟到執行緒池，等待的人
    # 各自佔一條執行緒排隊，其他請求照常走。
    # LOCK 仍然在：這台機器的瓶頸是峰值記憶體，同時跑兩張會 OOM。
    def run():
        with LOCK:
            return analyse(img)

    try:
        teeth = await run_in_threadpool(run)
    finally:
        if not OPEN:
            release_slot()
    ms = round((time.time() - t0) * 1000)
    log.info("%s  %dx%d  %d 顆  %d ms", file.filename, img.shape[1], img.shape[0],
             len(teeth), ms)
    return {"name": file.filename, "size": [img.shape[1], img.shape[0]],
            "scale": round(scale, 4), "teeth": teeth, "ms": ms, "folds": FOLDS}




# ── 保險庫：每個帳號一份，存在伺服器上 ──────────────────────────────────

def _vault_ready() -> None:
    if not vault.ready():
        # 沒金鑰就整個停掉，不要改存明文。錯誤訊息不提金鑰長什麼樣子。
        raise HTTPException(503, "保險庫暫時無法使用，請聯絡管理者")


@app.get("/api/vault")
def vault_load(user: str = Depends(require)):
    _vault_ready()
    out = vault.load(user)
    out["retain_days"] = vault.RETAIN_DAYS
    out["expires"] = vault.expires_at(user)
    return out


# ── 通關密語：伺服器只保管鎖住的盒子，打不開 ────────────────────────────

@app.get("/api/vault/key")
def key_get(user: str = Depends(require)):
    _vault_ready()
    k = vault.key_get(user)
    return {"exists": bool(k), **(k or {})}


@app.post("/api/vault/key")
async def key_set(request: Request, user: str = Depends(require)):
    _vault_ready()
    b = await request.json()
    try:
        vault.key_set(user, b["salt"], b["wrap_pass"], b["wrap_rec"])
    except KeyError:
        raise HTTPException(400, "缺少必要欄位") from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    auth.log(user, "key-set", "設定通關密語")
    return {"ok": True}


@app.put("/api/vault/key")
async def key_rewrap(request: Request, user: str = Depends(require)):
    """換密語，或用救援碼重設。兩種都是「把資料金鑰換一把鎖重新鎖上」，
    伺服器無從分辨也不需要分辨——它本來就打不開那個盒子。"""
    _vault_ready()
    b = await request.json()
    try:
        vault.key_rewrap(user, b["salt"], b["wrap_pass"])
    except KeyError:
        raise HTTPException(400, "缺少必要欄位") from None
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None
    auth.log(user, "key-change", "更換通關密語")
    return {"ok": True}


@app.post("/api/vault/{slot}/items")
async def vault_add(slot: int,
                    files: list[UploadFile] = File(...),
                    names: list[str] = Form(...),
                    mimes: list[str] = Form(...),
                    user: str = Depends(require)):
    """收的是瀏覽器加密好的密文。

    真正的檔名在 names 裡，也是密文——瀏覽器送上來的 filename 一律是 "blob"，
    所以存取紀錄、反向代理的日誌、這支程式的變數裡，都不會出現病患的檔名。
    """
    _vault_ready()
    if not (len(files) == len(names) == len(mimes)):
        raise HTTPException(400, "欄位數量對不上")
    out = []
    for f, name, mime in zip(files, names, mimes):
        raw = await f.read()
        if not raw:
            continue
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, f"影像超過 {MAX_BYTES // 1024 // 1024} MB")
        try:
            out.append(vault.put(user, slot, name, mime, raw))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
    auth.log(user, "upload", f"箱 {slot}，{len(out)} 張")
    return {"items": out, "usage": vault.usage(user)}


@app.get("/api/vault/item/{item_id}")
def vault_item(item_id: str, user: str = Depends(require)):
    _vault_ready()
    try:
        raw, mime = vault.read(user, item_id)
    except KeyError:
        raise HTTPException(404, "找不到這張影像") from None
    # 端出去的是密文，型別一律 octet-stream；瀏覽器解完才知道那是什麼。
    # 病患影像不進任何快取層——no-store 連瀏覽器的磁碟快取都不留。
    del mime
    return Response(raw, media_type="application/octet-stream",
                    headers={"Cache-Control": "no-store, private"})


@app.delete("/api/vault/item/{item_id}")
def vault_drop(item_id: str, user: str = Depends(require)):
    _vault_ready()
    if not vault.drop_item(user, item_id):
        raise HTTPException(404, "找不到這張影像")
    auth.log(user, "delete", f"影像 {item_id[:8]}")
    return {"ok": True, "usage": vault.usage(user)}


@app.put("/api/vault/{slot}")
async def vault_save(slot: int, request: Request, user: str = Depends(require)):
    _vault_ready()
    body = await request.json()
    try:
        vault.save_slot(user, slot, body.get("done"), body.get("combo"),
                        body.get("results"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"ok": True}


@app.delete("/api/vault/{slot}")
def vault_clear(slot: int, user: str = Depends(require)):
    _vault_ready()
    n = vault.clear_slot(user, slot)
    auth.log(user, "clear", f"箱 {slot}，刪了 {n} 張")
    return {"ok": True, "removed": n, "usage": vault.usage(user)}




# 放最後：掛在 "/" 的靜態站會吃掉所有沒被上面路由接走的網址。
# 這樣網站與 API 同源，瀏覽器的跨來源／私有網路限制通通繞開了。
if SITE and Path(SITE).is_dir():
    app.mount("/", StaticFiles(directory=SITE, html=True), name="site")
    log.info("同時提供網站：%s", SITE)
