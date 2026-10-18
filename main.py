"""屏幕上的影子碰撞原型。

相机拍到的亮墙上，较暗的区域被收成低分辨率碰撞格子，小人在上面跑和跳。
画面和碰撞来自同一帧，所以先不需要投影仪，也不做相机到墙面的对齐。

运行:
    python main.py
    python main.py --demo
    python main.py http://手机IP:8080/video
"""

import ctypes
import math
import sys
import threading
import time

import cv2
import numpy as np

from physics import Player, WorldGeom
from shadow_map import ShadowExtractor, crop_to_grid, grid_size, prepare_canvas

DEFAULT_URL = "http://172.20.10.3:8080/video"
WINDOW = "Shadow Collision"

HELP = """
影子碰撞原型
  相机对着被灯照亮的墙，物体放在灯和墙之间。墙上的暗部会变成青色碰撞块。
  A/D 或方向键移动，空格 / W / 上方向键跳跃。
  先把物体拿开，按 R 记录空墙，再把物体放回去。这样比只调暗部阈值稳。
  C 清除参考。G 显示或隐藏碰撞块。M 显示或隐藏右上角蒙版。Q 或 Esc 退出。
  滑条 dark thresh：没按 R 时，比这个值更暗的像素算影子。
  滑条 ref delta：按 R 之后，比空墙暗过这个差值才算影子。
  滑条 cell fill%：一个格子里要有多少比例的影子，才变成实体。
  滑条 smooth%：越大越稳，影子变化越慢。
  屏幕最底边的橙色细条是安全地面，阈值没调好时人不会掉出画面。
  IP Webcam 里尽量锁定曝光。自动曝光会让整面墙突然变成或失去碰撞。
""".strip()


def parse_args(argv):
    url = DEFAULT_URL
    demo = False
    self_test = False
    for arg in argv[1:]:
        if arg == "--demo":
            demo = True
        elif arg == "--self-test":
            self_test = True
        elif arg.startswith("http://") or arg.startswith("https://"):
            url = arg
        else:
            print("无法识别的参数: %s" % arg)
            print("用法: python main.py [--demo|--self-test|http://ip:8080/video]")
            raise SystemExit(2)
    return url, demo, self_test


class Keys:
    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.prev = {}

    def down(self, vk):
        return bool(self.user32.GetAsyncKeyState(vk) & 0x8000)

    def pressed(self, vk):
        now = self.down(vk)
        was = self.prev.get(vk, False)
        self.prev[vk] = now
        return now and not was


def window_focused(title):
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return title in buf.value


def screen_size():
    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass
    width = int(user32.GetSystemMetrics(0) or 1280)
    height = int(user32.GetSystemMetrics(1) or 720)
    return max(width, 640), max(height, 480)


class DemoSource:
    """没有摄像头时用合成影子，方便先确认跳跃和碰撞。"""

    def __init__(self):
        self.w = 960
        self.h = 540
        self.t = 0.0
        self.healthy = True

    def read(self, dt):
        self.t += dt
        w, h = self.w, self.h
        img = np.full((h, w, 3), (230, 226, 220), np.uint8)

        def block(x0, y0, x1, y1):
            cv2.rectangle(
                img,
                (int(x0 * w), int(y0 * h)),
                (int(x1 * w), int(y1 * h)),
                (16, 16, 16),
                -1,
            )

        block(0.0, 0.86, 1.0, 1.0)
        block(0.06, 0.66, 0.30, 0.76)
        cx = 0.50 + 0.20 * math.sin(self.t * 0.55)
        block(cx - 0.11, 0.50, cx + 0.11, 0.58)
        block(0.72, 0.46, 0.86, 0.86)
        return img

    def close(self):
        self.healthy = False


class CameraSource:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.alive = True
        self.fails = 0
        self.healthy = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.alive:
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                self.fails += 1
                self.healthy = self.fails < 20
                time.sleep(0.02)
                continue
            self.fails = 0
            self.healthy = True
            with self.lock:
                self.frame = frame

    def read(self, dt):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def close(self):
        self.alive = False
        try:
            self.cap.release()
        except Exception:
            pass


def open_camera(url, timeout=4.0):
    result = {}

    def worker():
        cap = cv2.VideoCapture()
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout * 1000))
        except Exception:
            pass
        opened = cap.open(url, cv2.CAP_FFMPEG)
        if opened and cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            result["cap"] = cap
        else:
            cap.release()
            result["cap"] = None

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout + 1.0)
    return result.get("cap")


def load_font(size):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return None


def draw_hud(frame, lines, font):
    if not lines:
        return
    if font is None:
        y = 24
        for line in lines:
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
        return
    from PIL import Image, ImageDraw

    width = frame.shape[1]
    line_h = font.size + 8
    pad = 8
    text_w = 0
    for line in lines:
        box = font.getbbox(line)
        text_w = max(text_w, box[2] - box[0])
    panel_w = min(width, text_w + pad * 2)
    panel_h = min(frame.shape[0], line_h * len(lines) + pad)
    panel = frame[0:panel_h, 0:panel_w]
    frame[0:panel_h, 0:panel_w] = (panel.astype(np.float32) * 0.38).astype(np.uint8)
    pil = Image.fromarray(cv2.cvtColor(frame[0:panel_h, 0:panel_w], cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for index, line in enumerate(lines):
        draw.text((pad, 4 + index * line_h), line, font=font, fill=(255, 255, 255))
    frame[0:panel_h, 0:panel_w] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def paint_collision(frame, solid, top_frac, bot_frac):
    """按每格影子的上下沿涂碰撞，而不是把整格都涂满。"""
    height, width = frame.shape[:2]
    rows, cols = solid.shape
    bh = max(1, height // rows)
    bw = max(1, width // cols)
    usable_h = rows * bh
    usable_w = cols * bw
    cache = paint_collision.cache
    key = (usable_h, usable_w, rows, cols, bh, bw)
    if cache.get("key") != key:
        tiled = np.zeros((rows, bh, cols, bw), np.uint8)
        tint = np.empty((usable_h, usable_w, 3), np.uint8)
        tint[:] = (190, 220, 70)
        edge_color = np.empty_like(tint)
        edge_color[:] = (255, 255, 90)
        cache.update(key=key, tiled=tiled, tint=tint, edge_color=edge_color, kernel=np.ones((3, 3), np.uint8))
    top = np.zeros_like(solid, dtype=np.float32) if top_frac is None else top_frac
    bot = np.ones_like(solid, dtype=np.float32) if bot_frac is None else bot_frac
    top_px = np.floor(np.nan_to_num(top, nan=0.0) * bh + 1e-4).astype(np.int32)
    bot_px = np.ceil(np.nan_to_num(bot, nan=1.0) * bh - 1e-4).astype(np.int32)
    local = np.arange(bh, dtype=np.int32)[:, None, None]
    vis = (local >= top_px[None]) & (local < bot_px[None]) & solid[None]
    tiled = cache["tiled"]
    tiled[:] = (np.transpose(vis, (1, 0, 2)).astype(np.uint8) * 255)[:, :, :, None]
    mask = tiled.reshape(usable_h, usable_w)
    if not np.any(mask):
        return
    view = frame[:usable_h, :usable_w]
    blended = cv2.addWeighted(view, 0.55, cache["tint"], 0.45, 0)
    cv2.copyTo(blended, mask, view)
    edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, cache["kernel"])
    cv2.copyTo(cache["edge_color"], edges, view)


paint_collision.cache = {}


def compose(frame, solid, player, extractor, lines, font, show_grid, show_mask):
    if show_grid and solid is not None and np.any(solid):
        paint_collision(frame, solid, extractor.top_frac, extractor.bot_frac)

    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height - 5), (width, height - 1), (40, 170, 255), -1)

    x = int(round(player.x))
    y = int(round(player.y))
    w = max(2, int(round(player.w)))
    h = max(2, int(round(player.h)))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (245, 245, 245), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 20, 20), 2)
    look = 0
    if player.vx > 10:
        look = max(1, w // 12)
    elif player.vx < -10:
        look = -max(1, w // 12)
    eye_r = max(2, w // 10)
    eye_y = y + int(h * 0.34)
    cv2.circle(frame, (x + w // 3 + look, eye_y), eye_r, (25, 25, 25), -1)
    cv2.circle(frame, (x + 2 * w // 3 + look, eye_y), eye_r, (25, 25, 25), -1)

    if show_mask and extractor.mask is not None:
        inset_w = max(120, width // 6)
        inset_h = max(68, int(inset_w * extractor.mask.shape[0] / extractor.mask.shape[1]))
        inset = cv2.resize(extractor.mask, (inset_w, inset_h), interpolation=cv2.INTER_NEAREST)
        inset_bgr = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
        x0 = width - inset_w - 10
        y0 = 10
        if y0 + inset_h < height and x0 > 10:
            frame[y0:y0 + inset_h, x0:x0 + inset_w] = inset_bgr
            cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + inset_w, y0 + inset_h), (255, 255, 255), 1)

    draw_hud(frame, lines, font)
    return frame


def hud_lines(extractor, solid, fps, camera_ok):
    mode = "参考差分" if extractor.use_reference else "绝对阈值"
    cols = extractor.cols
    rows = extractor.rows
    lines = [
        "碰撞 %d×%d  %s  %.0ffps  暗部%d  差分%d" % (
            cols, rows, mode, fps, int(extractor.threshold), int(extractor.delta)
        ),
        "A/D移动  空格跳  R记录空墙  C清除  G格子  M蒙版  Q退出",
    ]
    if not camera_ok:
        lines.append("摄像头读帧失败，画面可能已卡住")
    if solid is not None and solid.size:
        ratio = float(solid.mean())
        if ratio > 0.65:
            lines.append("实体太多：降低 dark thresh，或拿开物体后按 R")
        elif ratio < 0.01:
            lines.append("几乎没有实体：提高 dark thresh，并确认灯照在墙上")
    return lines


def run_game(source, max_w, max_h):
    first = None
    started = time.perf_counter()
    while first is None and time.perf_counter() - started < 4.0:
        first = source.read(0.016)
        if first is None:
            time.sleep(0.02)
    if first is None:
        return False

    canvas = prepare_canvas(first, max_w, max_h)
    cols, rows = grid_size(canvas.shape[1], canvas.shape[0])
    canvas = crop_to_grid(canvas, cols, rows)
    height, width = canvas.shape[:2]
    extractor = ShadowExtractor(cols, rows)
    geom = WorldGeom(width, height, cols, rows)
    player = Player(geom)
    player.spawn()
    font = load_font(20 if width >= 1000 else 16)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, width, height)
    cv2.createTrackbar("dark thresh", WINDOW, 120, 255, lambda _v: None)
    cv2.createTrackbar("ref delta", WINDOW, 26, 80, lambda _v: None)
    cv2.createTrackbar("cell fill%", WINDOW, 30, 80, lambda _v: None)
    cv2.createTrackbar("smooth%", WINDOW, 45, 100, lambda _v: None)

    keys = Keys()
    show_grid = True
    show_mask = True
    fps = 0.0
    previous = time.perf_counter()

    vk_left, vk_right, vk_up = 0x25, 0x27, 0x26
    vk_space, vk_a, vk_d, vk_w = 0x20, 0x41, 0x44, 0x57
    vk_r, vk_c, vk_g, vk_m, vk_q, vk_esc = 0x52, 0x43, 0x47, 0x4D, 0x51, 0x1B

    while True:
        now = time.perf_counter()
        dt = now - previous
        previous = now

        raw = source.read(dt)
        if raw is None:
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break
            continue

        frame = prepare_canvas(raw, max_w, max_h)
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        frame = crop_to_grid(frame, cols, rows)

        extractor.threshold = cv2.getTrackbarPos("dark thresh", WINDOW)
        extractor.delta = max(1, cv2.getTrackbarPos("ref delta", WINDOW))
        extractor.fill_ratio = max(5, cv2.getTrackbarPos("cell fill%", WINDOW)) / 100.0
        smooth_pct = cv2.getTrackbarPos("smooth%", WINDOW)
        extractor.new_weight = 1.0 - 0.85 * (smooth_pct / 100.0)

        focused = window_focused(WINDOW)
        move = 0
        if focused and (keys.down(vk_left) or keys.down(vk_a)):
            move -= 1
        if focused and (keys.down(vk_right) or keys.down(vk_d)):
            move += 1
        jump_keys = (vk_space, vk_w, vk_up)
        edges = [keys.pressed(vk) for vk in jump_keys]
        held = [keys.down(vk) for vk in jump_keys]
        jump_pressed = focused and any(edges)
        jump_held = focused and any(held)

        if focused and keys.pressed(vk_r):
            extractor.set_reference(frame)
            print("已记录空墙参考。把物体放回灯光前，影子会出现在碰撞图里。")
        if focused and keys.pressed(vk_c):
            extractor.clear_reference()
            print("已清除参考，改回绝对阈值。")
        if focused and keys.pressed(vk_g):
            show_grid = not show_grid
        if focused and keys.pressed(vk_m):
            show_mask = not show_mask

        solid = extractor.update(frame)
        player.update(dt, solid, extractor.top_frac, extractor.bot_frac, move, jump_pressed, jump_held)
        instant = 1.0 / dt if dt > 1e-4 else 0.0
        fps = instant if fps == 0.0 else fps * 0.9 + instant * 0.1
        lines = hud_lines(extractor, solid, fps, source.healthy)
        compose(frame, solid, player, extractor, lines, font, show_grid, show_mask)
        cv2.imshow(WINDOW, frame)

        elapsed_ms = (time.perf_counter() - now) * 1000.0
        key = cv2.waitKey(max(1, int(round(16.0 - elapsed_ms)))) & 0xFF
        quit_key = key in (ord("q"), 27) or (focused and (keys.pressed(vk_q) or keys.pressed(vk_esc)))
        visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE)
        if quit_key or visible < 1:
            break

    cv2.destroyAllWindows()
    return True


def run_self_test():
    wall = np.full((360, 640, 3), 210, np.uint8)
    current = wall.copy()
    cv2.rectangle(current, (180, 140), (460, 210), (15, 15, 15), -1)
    frame = prepare_canvas(current, 1280, 800)
    cols, rows = grid_size(frame.shape[1], frame.shape[0])
    extractor = ShadowExtractor(cols, rows)
    extractor.new_weight = 1.0
    extractor.set_reference(prepare_canvas(wall, 1280, 800))
    solid = extractor.update(frame)
    if solid[0, 0] or solid[0, -1]:
        print("参考差分把亮墙也收成了实体")
        return 1
    center = solid[solid.shape[0] // 2, solid.shape[1] // 2]
    if not center:
        print("参考差分没有留下中间的暗块")
        return 1

    demo = DemoSource()
    demo_frame = prepare_canvas(demo.read(0.0), 1280, 800)
    height, width = demo_frame.shape[:2]
    cols, rows = grid_size(width, height)
    extractor = ShadowExtractor(cols, rows)
    extractor.threshold = 120
    extractor.new_weight = 1.0
    solid = extractor.update(demo_frame)
    ratio = float(solid.mean())
    if not (0.04 < ratio < 0.5):
        print("合成场景实体比例异常: %.3f" % ratio)
        return 1
    if float(solid[-2].mean()) < 0.6 or float(solid[1].mean()) > 0.05:
        print("合成场景的地面或空白区域不对")
        return 1

    geom = WorldGeom(width, height, cols, rows)
    player = Player(geom)
    player.spawn()
    for _ in range(150):
        player.update(1.0 / 60.0, solid, extractor.top_frac, extractor.bot_frac, 0, False, False)
    feet = player.y + player.h
    visual_top = 0.66 * geom.height
    below = min(geom.rows - 1, int((feet + 1.0) / geom.cell_h))
    col = min(geom.cols - 1, max(0, int((player.x + player.w * 0.5) / geom.cell_w)))
    if not player.on_ground or not solid[below, col] or feet > geom.height * 0.8:
        print("小人没有落在左侧平台上", player.x, player.y, player.on_ground)
        return 1
    if abs(feet - visual_top) > 8.0:
        print("脚没有贴住影子上沿", round(feet, 2), round(visual_top, 2))
        return 1
    landed = player.y
    player.update(1.0 / 60.0, solid, extractor.top_frac, extractor.bot_frac, 0, True, True)
    for _ in range(18):
        player.update(1.0 / 60.0, solid, extractor.top_frac, extractor.bot_frac, 0, False, True)
    if player.y > landed - geom.cell_h:
        print("起跳高度不够", player.y, landed)
        return 1

    # 回到平台上，确认向右走不会穿透实体。
    player.spawn()
    for _ in range(150):
        player.update(1.0 / 60.0, solid, extractor.top_frac, extractor.bot_frac, 0, False, False)
    x_before = player.x
    for _ in range(40):
        player.update(1.0 / 60.0, solid, extractor.top_frac, extractor.bot_frac, 1, False, False)
    if player.x <= x_before:
        print("向右没有移动")
        return 1
    if player._hits():
        print("移动后卡进碰撞体")
        return 1

    print("self-test ok", demo_frame.shape)
    return 0


def main():
    url, demo, self_test = parse_args(sys.argv)
    if self_test:
        raise SystemExit(run_self_test())

    print(HELP)
    sw, sh = screen_size()
    max_w = max(640, sw - 60)
    max_h = max(480, sh - 120)

    if demo:
        print("使用合成影子。加上摄像头地址即可切换到手机画面。")
        source = DemoSource()
    else:
        print("正在连接 %s ..." % url)
        cap = open_camera(url)
        if cap is None:
            print("没有连上手机摄像头，先用合成影子。手机开好后重新运行，或执行:")
            print("  python main.py %s" % url)
            source = DemoSource()
        else:
            print("已连上摄像头。把画面对准被照亮的墙。")
            source = CameraSource(cap)

    try:
        ok = run_game(source, max_w, max_h)
        if not ok:
            print("没有拿到第一帧，改用合成影子。")
            source.close()
            source = DemoSource()
            run_game(source, max_w, max_h)
    finally:
        source.close()


if __name__ == "__main__":
    main()
