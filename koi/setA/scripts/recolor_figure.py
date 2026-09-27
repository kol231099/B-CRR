"""把投影片上那張殘差圖搬到深色底，但不動資料、也不換牙齒。

coolwarm 的中心是純白，直接貼在深色頁面上等於把「殘差＝0」畫成全圖最亮的
地方；整張反相又會讓 0 變成一片黑洞。所以分兩塊處理：

    圖面（殘差場與色條）  保留色相，只把接近白的中心色推到中灰
    其餘（白底、刻度、字）灰階反相：白底變深、黑字變亮

用法：
    py recolor_resid.py <裁好的圖> <輸出> [寬度]
"""
import sys, cv2, numpy as np

BG  = np.array((10, 12, 15), np.float32)      # 頁面深底
INK = 236.0                                   # 反相後的字色
MID = np.array((190, 180, 169), np.float32)   # #A9B4BE：殘差 0 的新中心色


def _plate(im):
    """回傳（圖面遮罩, 殘差場的 y 範圍, 色條左緣）。"""
    h, w = im.shape[:2]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mx = im.max(2).astype(np.int16); mn = im.min(2).astype(np.int16)
    col = cv2.morphologyEx(((mx - mn) >= 12).astype(np.uint8),
                           cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # 色條：細長的彩色元件。中間那條白（殘差 0）會把它切成上下兩段，
    # 取聯集外框一起填，0 的位置才不會留一條黑帶。刻度文字不在彩色元件裡，
    # 所以這個外框很窄，不會把標籤一起吃掉。
    n, lab, st, _ = cv2.connectedComponentsWithStats(col, 8)
    bars = [st[i, :4] for i in range(1, n)
            if st[i, cv2.CC_STAT_AREA] > 1500
            and st[i, cv2.CC_STAT_HEIGHT] > 3 * st[i, cv2.CC_STAT_WIDTH]]
    plate = np.zeros((h, w), np.uint8)
    bx0 = w
    if bars:
        bx0 = min(b[0] for b in bars); bx1 = max(b[0] + b[2] for b in bars)
        by0 = min(b[1] for b in bars); by1 = max(b[1] + b[3] for b in bars)
        plate[by0:by1, bx0:bx1] = 1

    # 殘差場：用「非純白」抓，中心接近白的像素才會一起收進來；
    # 取最大的那個橫向元件，投影片上其他殘留的方塊就不會被選到。
    m = cv2.morphologyEx((g < 250).astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    cand = [i for i in range(1, n)
            if st[i, cv2.CC_STAT_AREA] > 4000
            and st[i, cv2.CC_STAT_WIDTH] > st[i, cv2.CC_STAT_HEIGHT]]
    if not cand:
        return plate.astype(bool), (0, h), bx0
    i = max(cand, key=lambda k: st[k, cv2.CC_STAT_AREA])
    field = (lab == i).astype(np.uint8)
    cnts = cv2.findContours(field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(plate, cnts, -1, 1, cv2.FILLED)
    y0 = st[i, cv2.CC_STAT_TOP]
    return plate.astype(bool), (y0, y0 + st[i, cv2.CC_STAT_HEIGHT]), bx0


def recolor(im):
    h, w = im.shape[:2]
    plate, (fy0, fy1), bx0 = _plate(im)

    mx = im.max(2).astype(np.int16); mn = im.min(2).astype(np.int16)
    wgt = np.clip((mx - mn).astype(np.float32) / 45.0, 0, 1)[..., None]
    inside = im.astype(np.float32) * wgt + MID[None, None, :] * (1 - wgt)

    v = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inv = (255.0 - v) / 255.0
    outside = BG[None, None, :] * (1 - inv[..., None]) + INK * inv[..., None]

    out = np.where(plate[..., None], inside, outside).astype(np.uint8)

    # 色條比殘差場高，要完整收進來就得往上下多切一點，順手會帶到投影片上
    # 別的東西。色條那一欄以左、殘差場上下緣以外，整片抹成底色。
    left = np.zeros(w, bool); left[:bx0] = True
    out[:fy0][:, left] = BG.astype(np.uint8)
    out[fy1:][:, left] = BG.astype(np.uint8)
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 900
    out = recolor(cv2.imread(src))
    h, w = out.shape[:2]
    out = cv2.resize(out, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, out, [cv2.IMWRITE_WEBP_QUALITY, 92] if dst.endswith(".webp") else [])
    print(dst, out.shape)
