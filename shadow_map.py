"""把相机画面里的暗部收成一张低分辨率碰撞格子。

对应原作里 TouchDesigner 做的那一步：分离阴影，再压成一张粗糙但稳定的碰撞图。
没有投影仪时，碰撞图和显示用的是同一帧，所以格子会和屏幕上的影子对齐。
"""

import cv2
import numpy as np


def prepare_canvas(frame, max_w=1280, max_h=720):
    """把相机帧缩放到适合窗口的尺寸，保持长宽比。"""
    height, width = frame.shape[:2]
    scale = min(max_w / float(width), max_h / float(height))
    if width * scale < 960:
        scale = min(960.0 / width, max_w / float(width), max_h / float(height))
    out_w = max(160, int(round(width * scale)) // 2 * 2)
    out_h = max(120, int(round(height * scale)) // 2 * 2)
    if out_w == width and out_h == height:
        return frame
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (out_w, out_h), interpolation=interp)


def grid_size(width, height, cell=16):
    """格子大约 cell 像素见方，并且宽高都能被格子数整除。"""
    cols = max(16, int(width) // int(cell))
    rows = max(12, int(height) // int(cell))
    return cols, rows


def crop_to_grid(frame, cols, rows):
    """裁掉不足一格的边，让物理坐标和影子边沿用同一套格子。"""
    height, width = frame.shape[:2]
    out_w = (width // cols) * cols
    out_h = (height // rows) * rows
    if out_w == width and out_h == height:
        return frame
    return frame[:out_h, :out_w]


class ShadowExtractor:
    """从一帧画面得到 bool 碰撞图，True 表示小人撞得上。"""

    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.threshold = 120
        self.delta = 26
        self.fill_ratio = 0.30
        self.new_weight = 0.62
        self.reference = None
        self.use_reference = False
        self.occupancy = None
        self.top_frac = None
        self.bot_frac = None
        self.mask = None
        self.solid = None

    def set_reference(self, frame_bgr):
        """记下没有物体时的亮墙。之后比这张图更暗的地方才算影子。"""
        self.reference = self._blur_gray(frame_bgr)
        self.use_reference = True
        self.occupancy = None
        self.top_frac = None
        self.bot_frac = None

    def clear_reference(self):
        self.reference = None
        self.use_reference = False
        self.occupancy = None
        self.top_frac = None
        self.bot_frac = None

    def update(self, frame_bgr):
        gray = self._blur_gray(frame_bgr)
        mask = self._shadow_mask(gray)
        mask = self._cleanup(mask)
        density = self._cell_density(mask)
        observed = (density >= self.fill_ratio).astype(np.float32)
        top_frac, bot_frac = self._edge_fractions(mask)
        weight = float(np.clip(self.new_weight, 0.05, 1.0))
        if self.occupancy is None or self.occupancy.shape != observed.shape:
            self.occupancy = observed
            self.top_frac = np.where(np.isfinite(top_frac), top_frac, 0).astype(np.float32)
            self.bot_frac = np.where(np.isfinite(bot_frac), bot_frac, 1).astype(np.float32)
        else:
            self.occupancy = (1.0 - weight) * self.occupancy + weight * observed
            valid = np.isfinite(top_frac)
            self.top_frac = np.where(valid, (1.0 - weight) * self.top_frac + weight * top_frac, self.top_frac)
            self.bot_frac = np.where(valid, (1.0 - weight) * self.bot_frac + weight * bot_frac, self.bot_frac)
        solid = self.occupancy >= 0.5
        self.mask = mask
        self.solid = solid
        return solid

    def _blur_gray(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _shadow_mask(self, gray):
        if self.use_reference and self.reference is not None:
            if self.reference.shape != gray.shape:
                self.reference = cv2.resize(
                    self.reference, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA
                )
            adjusted = self._match_bright_wall(gray)
            darker = cv2.subtract(self.reference, adjusted)
            _, mask = cv2.threshold(darker, int(self.delta), 255, cv2.THRESH_BINARY)
            return mask
        _, mask = cv2.threshold(gray, int(self.threshold), 255, cv2.THRESH_BINARY_INV)
        return mask

    def _match_bright_wall(self, gray):
        """用亮部（大约第 90 百分位）对齐曝光，避免自动曝光把整面墙判成影子。"""
        ref_level = float(np.percentile(self.reference, 90))
        cur_level = float(np.percentile(gray, 90))
        shift = ref_level - cur_level
        if abs(shift) < 1.0:
            return gray
        return np.clip(gray.astype(np.float32) + shift, 0, 255).astype(np.uint8)

    def _cleanup(self, mask):
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = 180
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < min_area:
                mask[labels == index] = 0
        return mask

    def _cell_density(self, mask):
        small = cv2.resize(mask, (self.cols, self.rows), interpolation=cv2.INTER_AREA)
        return small.astype(np.float32) / 255.0

    def _edge_fractions(self, mask):
        """每个格子里影子的上沿和下沿，单位是格子高度的比例。

        只把整格标成实心时，脚会站在格子上沿，悬在真实影子上方。
        """
        height, width = mask.shape
        rows, cols = self.rows, self.cols
        bh = max(1, height // rows)
        bw = max(1, width // cols)
        cropped = mask[: rows * bh, : cols * bw] > 0
        vis = cropped.reshape(rows, bh, cols, bw).any(axis=3)
        has = vis.any(axis=1)
        first = vis.argmax(axis=1).astype(np.float32)
        last = (bh - 1 - vis[:, ::-1, :].argmax(axis=1)).astype(np.float32)
        top_frac = np.where(has, first / bh, np.nan).astype(np.float32)
        bot_frac = np.where(has, (last + 1.0) / bh, np.nan).astype(np.float32)
        return top_frac, bot_frac
