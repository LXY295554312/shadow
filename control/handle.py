"""
盖世小鸡 / Xbox 兼容手柄 Demo（Pygame）

用法:
    python control/handle.py

操作（Xbox 布局）:
    左摇杆 / 方向键  — 左右移动
    A 键            — 跳跃 / 二连跳
    X 键            — 冲刺
    B 键 / ESC      — 退出

蓝牙连上后若系统识别为 Xbox 控制器，一般可直接使用。
"""

from __future__ import annotations

import sys

import pygame


# ---------------------------------------------------------------------------
# Xbox / XInput 常见按键编号（盖世小鸡 Xbox 模式通常一致）
# ---------------------------------------------------------------------------
BTN_A = 0
BTN_B = 1
BTN_X = 2
BTN_Y = 3
BTN_LB = 4
BTN_RB = 5
BTN_BACK = 6
BTN_START = 7
BTN_LS = 8  # 左摇杆按下
BTN_RS = 9  # 右摇杆按下

AXIS_LX = 0  # 左摇杆 X：左负右正
AXIS_LY = 1  # 左摇杆 Y：上负下正
AXIS_RX = 2  # 右摇杆 X（部分驱动为 3）
AXIS_RY = 3  # 右摇杆 Y（部分驱动为 4）
# 扳机在 Windows/XInput 上常为 axis 4/5，这里只做移动跳跃，暂不依赖

DEADZONE = 0.25  # 摇杆死区，避免漂移


# ---------------------------------------------------------------------------
# 手柄封装
# ---------------------------------------------------------------------------
class Gamepad:
    """读取单个手柄输入：move_x、jump / dash 边沿触发。"""

    def __init__(self, index: int = 0) -> None:
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "未检测到手柄。请确认蓝牙已连接，且系统能识别为 Xbox 控制器。"
            )
        self.joy = pygame.joystick.Joystick(index)
        self.joy.init()
        self._prev_a = False
        self._prev_x = False
        print(
            f"已连接: {self.joy.get_name()} | "
            f"按钮 {self.joy.get_numbuttons()} | "
            f"轴 {self.joy.get_numaxes()} | "
            f"方向帽 {self.joy.get_numhats()}"
        )

    @staticmethod
    def _apply_deadzone(value: float, deadzone: float = DEADZONE) -> float:
        if abs(value) < deadzone:
            return 0.0
        # 死区外重新映射到 [0, 1]，手感更线性
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - deadzone) / (1.0 - deadzone)

    def poll(self) -> tuple[float, bool, bool]:
        """
        返回 (move_x, jump_pressed, dash_pressed)。
        jump / dash 仅在 A / X 刚按下的那一帧为 True（边沿触发）。
        """
        move_x = 0.0

        if self.joy.get_numaxes() > AXIS_LX:
            move_x = self._apply_deadzone(self.joy.get_axis(AXIS_LX))

        # 方向键（hat）也可左右移动，优先/叠加摇杆
        if self.joy.get_numhats() > 0:
            hx, _hy = self.joy.get_hat(0)
            if hx != 0:
                move_x = float(hx)

        a_down = self.joy.get_numbuttons() > BTN_A and self.joy.get_button(BTN_A)
        jump = a_down and not self._prev_a
        self._prev_a = a_down

        x_down = self.joy.get_numbuttons() > BTN_X and self.joy.get_button(BTN_X)
        dash = x_down and not self._prev_x
        self._prev_x = x_down

        return move_x, jump, dash

    def quit_pressed(self) -> bool:
        if self.joy.get_numbuttons() <= BTN_B:
            return False
        return bool(self.joy.get_button(BTN_B))

    def debug_lines(self) -> list[str]:
        """调试用：当前各轴/按钮原始值。"""
        axes = [
            f"axis{i}={self.joy.get_axis(i):+.2f}"
            for i in range(self.joy.get_numaxes())
        ]
        btns = [
            str(i)
            for i in range(self.joy.get_numbuttons())
            if self.joy.get_button(i)
        ]
        hats = [
            f"hat{i}={self.joy.get_hat(i)}"
            for i in range(self.joy.get_numhats())
        ]
        return [
            " | ".join(axes) if axes else "无轴",
            "按下按钮: " + (",".join(btns) if btns else "无"),
            " | ".join(hats) if hats else "无方向帽",
        ]


# ---------------------------------------------------------------------------
# 简易角色（验证移动 / 跳跃）
# ---------------------------------------------------------------------------
class Player:
    WIDTH, HEIGHT = 40, 60
    SPEED = 320.0  # px/s
    JUMP_V = -520.0
    GRAVITY = 1400.0
    MAX_JUMPS = 2  # 地面一跳 + 空中一跳
    DASH_SPEED = 900.0
    DASH_DURATION = 0.14  # 秒
    DASH_COOLDOWN = 0.45  # 秒

    def __init__(self, x: float, ground_y: float) -> None:
        self.x = x
        self.y = ground_y - self.HEIGHT
        self.vx = 0.0
        self.vy = 0.0
        self.ground_y = ground_y
        self.on_ground = True
        self.jumps_left = self.MAX_JUMPS
        self.facing = 1  # 1 右 / -1 左
        self.dash_timer = 0.0
        self.dash_cd = 0.0

    @property
    def dashing(self) -> bool:
        return self.dash_timer > 0.0

    def update(self, dt: float, move_x: float, jump: bool, dash: bool) -> None:
        if move_x != 0.0:
            self.facing = 1 if move_x > 0 else -1

        if self.dash_cd > 0.0:
            self.dash_cd = max(0.0, self.dash_cd - dt)

        if dash and self.dash_cd <= 0.0 and not self.dashing:
            self.dash_timer = self.DASH_DURATION
            self.dash_cd = self.DASH_COOLDOWN
            # 冲刺瞬间压低竖直速度，轨迹更平
            self.vy = 0.0

        if self.dashing:
            self.dash_timer = max(0.0, self.dash_timer - dt)
            self.vx = self.facing * self.DASH_SPEED
            self.vy = 0.0  # 冲刺期间忽略重力位移感
        else:
            self.vx = move_x * self.SPEED
            if jump and self.jumps_left > 0:
                self.vy = self.JUMP_V
                self.jumps_left -= 1
                self.on_ground = False
            self.vy += self.GRAVITY * dt

        self.x += self.vx * dt
        self.y += self.vy * dt

        # 地面碰撞
        if self.y + self.HEIGHT >= self.ground_y:
            self.y = self.ground_y - self.HEIGHT
            self.vy = 0.0
            self.on_ground = True
            self.jumps_left = self.MAX_JUMPS

    def clamp(self, screen_w: int) -> None:
        self.x = max(0.0, min(self.x, screen_w - self.WIDTH))

    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), self.WIDTH, self.HEIGHT)


# ---------------------------------------------------------------------------
# Demo 主循环
# ---------------------------------------------------------------------------
def run_demo() -> None:
    pygame.init()
    pygame.joystick.init()

    W, H = 800, 480
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("手柄 Demo — 移动 / A 跳 / X 冲刺 / B 退出")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("microsoftyahei,consolas,arial", 18)

    try:
        pad = Gamepad(0)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        # 无手柄时仍开窗口提示，方便排查
        screen.fill((30, 30, 40))
        tip = font.render(str(e), True, (255, 180, 80))
        screen.blit(tip, (40, H // 2))
        pygame.display.flip()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN):
                    waiting = False
            clock.tick(30)
        pygame.quit()
        return

    ground_y = H - 60
    player = Player(W / 2 - Player.WIDTH / 2, ground_y)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            # 热插拔：重新枚举
            elif event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
                print("手柄连接状态变化，请重启 Demo 以重新绑定。")

        move_x, jump, dash = pad.poll()
        if pad.quit_pressed():
            running = False

        player.update(dt, move_x, jump, dash)
        player.clamp(W)

        # --- 绘制 ---
        screen.fill((24, 28, 36))
        pygame.draw.rect(screen, (60, 70, 80), (0, ground_y, W, H - ground_y))
        color = (255, 210, 80) if player.dashing else (90, 200, 120)
        pygame.draw.rect(screen, color, player.rect())

        lines = [
            f"FPS {clock.get_fps():.0f}  move_x={move_x:+.2f}  "
            f"on_ground={player.on_ground}  jumps_left={player.jumps_left}  "
            f"dash_cd={player.dash_cd:.2f}",
            *pad.debug_lines(),
            "左摇杆/方向键移动 | A 跳跃/二连跳 | X 冲刺 | B / ESC 退出",
        ]
        for i, text in enumerate(lines):
            surf = font.render(text, True, (220, 220, 230))
            screen.blit(surf, (12, 10 + i * 22))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run_demo()
