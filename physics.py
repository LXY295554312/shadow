"""小人在碰撞格子上的平台跳跃。

格子是整面墙的低分辨率地图，不做一个影子一个刚体。
这样影子每帧变形时，不用重建一堆多边形。
"""

import math


class WorldGeom:
    def __init__(self, width, height, cols, rows):
        self.width = int(width)
        self.height = int(height)
        self.cols = int(cols)
        self.rows = int(rows)
        self.cell_w = self.width / float(self.cols)
        self.cell_h = self.height / float(self.rows)


class Player:
    def __init__(self, geom):
        self.geom = geom
        cell = geom.cell_h
        self.w = geom.cell_w * 1.7
        self.h = geom.cell_h * 2.6
        self.gravity = 52.0 * cell
        self.jump_speed = 34.0 * cell
        self.move_speed = 16.0 * geom.cell_w
        self.max_fall = 30.0 * cell
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.coyote = 0.0
        self.jump_buffer = 0.0
        self.grid = None
        self.top_frac = None
        self.bot_frac = None

    def spawn(self):
        geom = self.geom
        self.x = geom.width * 0.18 - self.w * 0.5
        self.y = geom.height * 0.08
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.coyote = 0.0
        self.jump_buffer = 0.0

    def update(self, dt, grid, top_frac, bot_frac, move, jump_pressed, jump_held):
        dt = min(max(float(dt), 0.0), 0.05)
        self.grid = grid
        self.top_frac = top_frac
        self.bot_frac = bot_frac
        geom = self.geom
        self.jump_buffer = max(0.0, self.jump_buffer - dt)
        if jump_pressed:
            self.jump_buffer = 0.12
        if self.on_ground:
            self.coyote = 0.12
        else:
            self.coyote = max(0.0, self.coyote - dt)

        if self.jump_buffer > 0.0 and self.coyote > 0.0:
            self.vy = -self.jump_speed
            self.jump_buffer = 0.0
            self.coyote = 0.0
            self.on_ground = False
        elif self.vy < 0.0 and not jump_held:
            self.vy = max(self.vy, -self.jump_speed * 0.4)

        self.vx = float(move) * self.move_speed
        self._unstick(grid, geom)

        remaining = dt
        while remaining > 1e-6:
            step = min(1.0 / 120.0, remaining)
            self._integrate(step, grid, geom)
            remaining -= step

    def _integrate(self, dt, grid, geom):
        self.vy = min(self.vy + self.gravity * dt, self.max_fall)
        self.x += self.vx * dt
        self._resolve_x(grid, geom)
        self.y += self.vy * dt
        self.on_ground = False
        self._resolve_y(grid, geom)

    def _unstick(self, grid, geom):
        """影子长进身体里时，把小人抬到实体上方，避免卡在碰撞体内部。"""
        if not self._hits():
            return
        origin = self.y
        limit = geom.height * 0.55
        dy = 2.0
        while dy <= limit:
            self.y = origin - dy
            if self.y < 0.0:
                break
            if not self._hits():
                return
            dy += 2.0
        self.y = origin

    def _resolve_x(self, grid, geom):
        if self.x < 0.0:
            self.x = 0.0
            self.vx = 0.0
        if self.x + self.w > geom.width:
            self.x = geom.width - self.w
            self.vx = 0.0
        hits = self._hits()
        if not hits:
            return
        if self.vy >= 0.0 and self._try_step_up(grid, geom):
            return
        if self.vx > 0.0:
            face = min(col * geom.cell_w for col, _row in hits)
            self.x = face - self.w - 0.01
            self.vx = 0.0
        elif self.vx < 0.0:
            face = max((col + 1) * geom.cell_w for col, _row in hits)
            self.x = face + 0.01
            self.vx = 0.0
        else:
            left = min(col * geom.cell_w for col, _row in hits)
            right = max((col + 1) * geom.cell_w for col, _row in hits)
            if (self.x + self.w - left) < (right - self.x):
                self.x = left - self.w - 0.01
            else:
                self.x = right + 0.01

    def _try_step_up(self, grid, geom):
        """矮于大约一格的影子边缘可以直接走上去，不用每次都跳。"""
        origin = self.y
        limit = geom.cell_h * 1.15
        dy = 2.0
        while dy <= limit:
            self.y = origin - dy
            if self.y >= 0.0 and not self._hits():
                return True
            dy += 2.0
        self.y = origin
        return False

    def _resolve_y(self, grid, geom):
        if self.y < 0.0:
            self.y = 0.0
            if self.vy < 0.0:
                self.vy = 0.0
        hits = self._hits()
        if hits:
            if self.vy >= 0.0:
                face = min(self._span(col, row)[0] for col, row in hits)
                self.y = face - self.h - 0.01
                self.vy = 0.0
                self.on_ground = True
            else:
                face = max(self._span(col, row)[1] for col, row in hits)
                self.y = face + 0.01
                self.vy = 0.0
        floor_y = geom.height - 1.0
        if self.y + self.h > floor_y:
            self.y = floor_y - self.h
            if self.vy > 0.0:
                self.vy = 0.0
            self.on_ground = True

    def _span(self, col, row):
        """这一格里真正有影子的垂直范围，单位是像素。"""
        geom = self.geom
        top_frac = 0.0
        bot_frac = 1.0
        if self.top_frac is not None:
            top_frac = float(self.top_frac[row, col])
            bot_frac = float(self.bot_frac[row, col])
            if not math.isfinite(top_frac):
                top_frac = 0.0
            if not math.isfinite(bot_frac):
                bot_frac = 1.0
        return (row + top_frac) * geom.cell_h, (row + bot_frac) * geom.cell_h

    def _hits(self):
        geom = self.geom
        grid = self.grid
        eps = 0.01
        c0 = math.floor((self.x + eps) / geom.cell_w)
        c1 = math.floor((self.x + self.w - eps) / geom.cell_w)
        r0 = math.floor((self.y + eps) / geom.cell_h)
        r1 = math.floor((self.y + self.h - eps) / geom.cell_h)
        hits = []
        for row in range(r0, r1 + 1):
            if row < 0 or row >= geom.rows:
                continue
            for col in range(c0, c1 + 1):
                if col < 0 or col >= geom.cols or not grid[row, col]:
                    continue
                solid_top, solid_bot = self._span(col, row)
                if self.y + self.h > solid_top + eps and self.y < solid_bot - eps:
                    hits.append((col, row))
        return hits
