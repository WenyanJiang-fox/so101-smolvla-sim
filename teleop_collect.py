"""交互式遥操作采集：鼠标 + 键盘直接驱动 6 个舵机，边操作边录 LeRobot v3 数据集。

    python teleop_collect.py                     # 正常采集
    python teleop_collect.py --selftest          # 自检：每个舵机各扫一遍，存一条后退出

控制方式（一个舵机一个通道，谁也不抢谁）：

    舵机              通道                类型      范围
    shoulder_pan      鼠标左右            相对      20–160°
    shoulder_lift     鼠标上下            相对      20–120°
    elbow_flex        W / S               速率 60°/s  20–130°
    wrist_flex        Q / E               速率 60°/s  20–160°
    wrist_roll        A / D               速率 90°/s  0–180°
    gripper           滚轮上 / 下          步进 10°/格  0–180°
    gripper           左键按住 / 右键按住   速率 90°/s  0–180°
    按住 Shift        所有速度 x0.25（精细档）
    R                 平滑回到待命姿态（只动手臂，不动方块）
    空格              切换录制：没录就是「开始」，在录就是「结束 + 存盘」。
                      当前是哪一种，面板第二行直接写出来（SPACE -> START / STOP）
    退格              丢弃当前 episode 并复位场景
    Esc               释放鼠标（要用电脑时按）。再按一次 Esc、或在窗口里点一下，
                      都会重新锁定；切走再切回来也算。

**鼠标是相对锁定的，这是整个操作手感的关键。** 光标被锁在窗口中心并隐藏，鼠标的
「移动量」累加到舵机角度上，手停臂停。它解决的是「要同时管太多」：绝对映射下你
必须把指针一直压在画面里正确的位置上，手一离开鼠标去按 W，指针就跟着走、机械臂
跟着甩 —— 鼠标和键盘被迫「同时用」。锁定之后两者变成「分时用」：手随便离开鼠标
去敲键盘，机械臂纹丝不动。FPS 游戏就是这么解决的。

代价是拿不到 raw input（cv2 不给 WM_INPUT 通路），增量来自 Windows 加速后的指针
位置，所以快甩比慢移更「冲」。对操控其实是好事（慢=精细、快=粗调），但如果觉得
飘，把系统鼠标设置里的「增强指针精确度」取消勾选就变成纯线性了。

只有 "SO-101 preview" 这一个窗口，既是主视角也是唯一输入面，四宫格布局：

    ┌────────────────┬────────┬────────┐
    │                │ wrist  │  top   │   top 是操作员专用俯视图，
    │  front 512²    │ 256²   │  256²  │   只为看清进深和 pan 转角，
    │  （主视角）     ├────────┴────────┤   绝不进数据集（见 Preview.show 的
    │                │  文字面板 512×256 │   render_view 调用处）
    └────────────────┴─────────────────┘

数据集口径和真机一致：state/action 都是舵机 0~180°，顺序 shoulder_pan → gripper。
"""
from __future__ import annotations

import argparse
import atexit
import ctypes
import ctypes.wintypes as wt
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

import paths
import sim_env

# 中文 Windows 的控制台是 GBK：万一某行里混进了它编不了的字符（上标、emoji…），
# print 会直接抛 UnicodeEncodeError 把整个采集打断。宁可少个字符也不能崩。
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:            # noqa: BLE001 —— 重定向到文件等场景可能不支持
    pass

# ======================= 控制映射 =======================
#
# 每行：(关节名, 初始角, 下限, 上限, 通道说明)
#
# 上下限来自实测的无碰撞行程（单轴从 home 扫 0~180°，统计机械臂自碰撞 + 扎桌面），
# 又各留了几度余量。不是拍脑袋定的，也不是 jnt_range 本身 —— jnt_range 是机械
# 极限，撞上之前早就自己碰上桌子了。
#
# 夹爪的两个端点也是实测的：0° 两爪尖相距 6.8mm（闭合），180° 相距 141.8mm（全开）；
# 夹这个 40mm 的方块大约走到 38° 就贴住。
JOINTS = [
    ("shoulder_pan",  95.0,  20.0, 160.0, "mouse X"),
    ("shoulder_lift", 92.0,  20.0, 120.0, "mouse Y"),
    ("elbow_flex",   108.0,  20.0, 130.0, "W / S"),
    ("wrist_flex",   118.0,  20.0, 160.0, "Q / E"),
    ("wrist_roll",   110.0,   0.0, 180.0, "A / D"),
    ("gripper",      150.0,   0.0, 180.0, "wheel"),
]
# 初值同时是「开场的静止姿态」和 R 键回的那个姿态。它由随机搜索挑出来
# （4 万组关节角里夹爪离方块顶 75mm 最近的一组，误差 2.4mm），再往上抬一点让夹爪
# 不碰到方块：实测 [95,100,110,120] 会压到方块上（2 个接触点），[95,92,108,118] 是 0。
# 选它而不是伸手抬臂的「home」是因为 front_cam 离基座只有 30cm —— 手臂一抬到
# z=346mm 就顶出画面了，整条大臂看不见，没法操作。

# ---- 鼠标（相对锁定）----
DEG_PER_COUNT = 0.05    # 鼠标每移动 1 个计数，舵机转多少度。嫌快/慢就调这个
PAN_SIGN = 1.0          # 鼠标右移 → pan 增大（= 俯视顺时针，跟转头方向一致）
LIFT_SIGN = 1.0         # 鼠标下移 → lift 增大（实测 lift 变大是夹爪下降）
FINE_SCALE = 0.25       # 按住 Shift 时所有速度乘这个
WHEEL_DEG = 10.0        # 滚轮每格夹爪走多少度（上滚张开、下滚闭合）

# 每个舵机的最大角速度（度/秒）。目标角可以瞬间跳（回 R、重新锁定光标），
# 但没有这一层的话机械臂会直接弹过去。
SLEW_RATE = 180.0
# 键盘/滚轮的速率（度/秒）
RATE = {"elbow_flex": 60.0, "wrist_flex": 60.0,
        "wrist_roll": 90.0, "gripper": 90.0}

# 自检时每个舵机扫一趟给几秒。里面是一段梯形波，两端各有 0.6s 停靠、上下各
# 1.2s 行程 —— 1.2s × 180°/s = 216° > 最长行程 180°，保证真能扫到两个端点。
# 用三角波（不停靠）会扫不到：波峰只在一瞬间出现，而限速还在爬坡（踩过，
# 夹爪实测只到了 58.7°，因为它要从上一个姿态先走 110° 才轮到）。
SEG_S = 4.0
# 梯形波各拐点（段内相位 0~1）
WAVE = ((0.00, 0.0), (0.15, 0.0), (0.45, 1.0), (0.55, 1.0), (0.85, 0.0), (1.00, 0.0))


def wave_at(frac):
    """段内相位 → 0~1 的梯形波（两端停靠，方便限速追上）。"""
    f = frac % 1.0
    for (f0, v0), (f1, v1) in zip(WAVE, WAVE[1:]):
        if f <= f1:
            return v0 + (v1 - v0) * (0.0 if f1 == f0 else (f - f0) / (f1 - f0))
    return WAVE[-1][1]

# ---- 窗口布局（客户区像素）----
FRONT_TILE = 512        # 主视角；也是光标锁定的位置（它的中心）
SMALL_TILE = 256        # wrist / top 两路小图
PANEL_W, PANEL_H = 512, 256
CANVAS_W = FRONT_TILE + SMALL_TILE * 2      # 1024
CANVAS_H = FRONT_TILE                       # 512
# 光标锁在这里（front 那格的中心）
LOCK_PT = (FRONT_TILE // 2, FRONT_TILE // 2)

# Win32 虚拟键码（用 GetAsyncKeyState 轮询，不依赖任何回调的按下/抬起语义）
VK = {"SPACE": 0x20, "BACK": 0x08, "ESC": 0x1B, "SHIFT": 0x10, "R": 0x52,
      "LBTN": 0x01, "RBTN": 0x02,          # 只在「Esc 释放后点回窗口」时用
      "W": 0x57, "S": 0x53, "Q": 0x51, "E": 0x45, "A": 0x41, "D": 0x44}
HELD_KEYS = ("W", "S", "Q", "E", "A", "D", "LBTN", "RBTN", "SHIFT")

REPO_ID = "so101_sim/teleop_pick_cube"

# 短于此帧数（30fps 下 = 1 秒）的 episode 直接丢弃，不进数据集。
# 开窗后机械臂要先花 ~1.2s 从默认折起姿态走到待命位，这期间按空格会把那一团
# 折起姿态录成一条 17 帧的废片（还带不上一路视频），训练加载时会硬报错。
MIN_EPISODE_FRAMES = 30


# ======================= 输入 =======================

class WinInput:
    """预览窗口（cv2 建的）的鼠标/键盘 + 相对鼠标锁定。

    cv2 窗口是个普通 Win32 窗口，标题已知，FindWindowW 一把就拿到，不用像
    MuJoCo viewer 那样按窗口类名 EnumWindows 去猜（还容易抓到离屏 renderer
    那个隐藏窗口）。
    """

    def __init__(self):
        self.u32 = ctypes.windll.user32
        # 必须在建窗口前调用，否则客户区坐标会被 DPI 缩放，光标锁定位置会偏
        self.u32.SetProcessDPIAware()
        self.hwnd = None
        self.locked = False
        self._lock_screen = (0, 0)
        self._wheel = 0           # 滚轮累积格数，由 cv2 的回调写入
        self._prev = {}           # 上一帧按键状态，用于边沿检测
        # 主循环的 finally 之外再加一道：任何正常的解释器退出（包括异常、
        # sys.exit）都把光标还回去。防不住强杀，但那已经没办法了。
        atexit.register(self._show)

    def bind(self, hwnd):
        self.hwnd = hwnd

    # ---- 窗口 ----

    def client_rect(self):
        """客户区的屏幕坐标 (l, t, r, b)。"""
        if not self.hwnd:
            return None
        r = wt.RECT()
        self.u32.GetClientRect(self.hwnd, ctypes.byref(r))
        tl, br = wt.POINT(r.left, r.top), wt.POINT(r.right, r.bottom)
        self.u32.ClientToScreen(self.hwnd, ctypes.byref(tl))
        self.u32.ClientToScreen(self.hwnd, ctypes.byref(br))
        return (tl.x, tl.y, br.x, br.y)

    @property
    def focused(self):
        """窗口是不是前台 —— 免得你在别的窗口打字时机械臂乱动。"""
        return bool(self.hwnd) and self.u32.GetForegroundWindow() == self.hwnd

    # ---- 相对鼠标 ----

    def lock(self, pt=None):
        """把光标钉在 pt（客户区像素坐标）并隐藏。

        之后 take_delta() 每帧读它离 pt 多远、再拽回去，差值就是这一帧的鼠标位移。
        不做 ClipCursor：每帧回中本身就能纠偏，而 ClipCursor 万一崩了会把光标
        **困在窗口里**，那是比「某帧少读一点位移」糟糕得多的失败模式。
        """
        if self.locked or not self.hwnd:
            return
        rect = self.client_rect()
        if rect is None:
            return
        l, t = rect[0], rect[1]
        px, py = pt or (FRONT_TILE // 2, FRONT_TILE // 2)
        self._lock_screen = (l + px, t + py)
        self.u32.SetCursorPos(*self._lock_screen)
        self._hide()
        self.locked = True
        self._wheel = 0

    def unlock(self):
        if not self.locked:
            return
        self.locked = False
        self._show()

    # 光标隐藏只能用 ShowCursor。试过 SetCursor(NULL)（以及把窗口类的 GCLP_HCURSOR
    # 设成 NULL）：它确实能隐藏，但**挡不住 WM_SETCURSOR** —— 鼠标一动，窗口就把
    # 光标改回箭头，30fps 下就是每帧闪一下。实测在真实的 SendInput 移动下
    # GetCursorInfo 100% 报「正在显示」。ShowCursor 没有这个问题。
    #
    # ShowCursor 的代价是它改的是**全系统共享**的显示计数器：进程被强杀
    # （任务管理器、直接关掉控制台窗口）那个 -1 会留在系统里，用户的鼠标光标在
    # 整个桌面上消失。对策是 lock() 里先把它归零再隐藏 —— 上次崩了留下的负数，
    # 下次一启动就被抹平；再配上 atexit 兜住正常退出。这是能拿到的最好结果，
    # 真要防「进程被强杀」Windows 没给别的口子。
    def _hide(self):
        self._normalize_cursor()
        for _ in range(64):                 # 计数器要调到负数才真的隐藏
            if self.u32.ShowCursor(False) < 0:
                break

    def _show(self):
        self._normalize_cursor()

    def _normalize_cursor(self):
        """把全局显示计数器推到 >= 0（撤销上一次异常退出留下的隐藏）。"""
        for _ in range(64):
            if self.u32.ShowCursor(True) >= 0:
                break

    def take_delta(self):
        """这一帧的鼠标位移 (dx, dy)，像素计数；未锁定时恒为 (0, 0)。"""
        if not self.locked:
            return 0, 0
        pt = wt.POINT()
        self.u32.GetCursorPos(ctypes.byref(pt))
        dx = pt.x - self._lock_screen[0]
        dy = pt.y - self._lock_screen[1]
        if dx or dy:
            self.u32.SetCursorPos(*self._lock_screen)
        # ShowCursor 是一劳永逸的，这里**不能**每帧再调一次 —— _normalize_cursor
        # 会先把计数器推回可见再隐藏，每帧闪一下。
        return dx, dy

    # ---- 滚轮（由 Preview 的 cv2 回调喂进来）----

    def add_wheel(self, notches):
        self._wheel += notches

    def take_wheel(self):
        n, self._wheel = self._wheel, 0
        return n

    # ---- 按键 ----

    def down(self, name):
        return bool(self.u32.GetAsyncKeyState(VK[name]) & 0x8000)

    def pressed(self, name):
        """这一帧刚按下（边沿）。"""
        now, was = self.down(name), self._prev.get(name, False)
        return now and not was

    def latch(self):
        """每帧末尾调一次，记录按键状态。"""
        for k in VK:
            self._prev[k] = self.down(k)


# ======================= 会话 =======================

class TeleopSession:

    def __init__(self, dataset_root, selftest=False):
        self.env = sim_env.SO101SimEnv()
        self.m, self.d = self.env.m, self.env.d
        self.selftest = selftest
        self.dataset_root = Path(dataset_root)

        self.lo = np.array([j[2] for j in JOINTS], dtype=float)
        self.hi = np.array([j[3] for j in JOINTS], dtype=float)
        self.rate = np.array([RATE.get(j[0], 0.0) for j in JOINTS], dtype=float)
        self.home = np.array([j[1] for j in JOINTS], dtype=float)

        # tgt = 目标角（鼠标/键盘改它），cmd = 实际发给仿真的角（tgt 过一道限速）。
        # 分开是为了让 R 回待命、重新锁定光标这类「目标瞬跳」不会把机械臂也瞬移。
        self.tgt = self.home.copy()
        self.cmd = self.home.copy()

        self.recording = False
        self.frames = []
        self.saved = 0
        self.ds = None
        self.status = "ready"
        self.held = []
        self.fine = False
        self.locked = False           # 光标是否锁着，只用来显示
        self.active = True            # 窗口是不是前台且没按 Esc（否则冻住机械臂）
        self.esc_free = False         # 按过 Esc = 主动交出光标，点回窗口才重新锁

        # reset 只把起始角写进 ctrl，qpos 还停在模型默认零位（折成一团）。
        # 前几十毫秒物理会把机械臂带过去，主循环开跑前还会再等它一会儿。
        self.env.reset(cube_xy=(paths.PICK_X, paths.PICK_Y), home_deg=self.home)

    # ---- 场景 ----

    def reset_scene(self):
        self.env.reset(cube_xy=(paths.PICK_X, paths.PICK_Y), home_deg=self.home)
        self.tgt[:] = self.home
        self.cmd[:] = self.home
        self.frames.clear()

    # ---- 数据集 ----

    def open_dataset(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        root = self.dataset_root
        names = paths.JOINT_NAMES
        H = W = paths.IMG_H
        feats = {
            "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
            "action": {"dtype": "float32", "shape": (6,), "names": names},
        }
        for ck in (paths.CAM_FRONT, paths.CAM_WRIST):
            feats[f"observation.images.{ck}"] = {
                "dtype": "video", "shape": (H, W, 3),
                "names": ["height", "width", "channels"]}
        if (root / "meta" / "info.json").exists():
            self.ds = LeRobotDataset.resume(REPO_ID, root=str(root))
            print(f"续写已有数据集：{root}（已有 {self.ds.num_episodes} 条）")
        else:
            self.ds = LeRobotDataset.create(
                repo_id=REPO_ID, fps=paths.FPS, root=str(root),
                robot_type=paths.ROBOT_TYPE, use_videos=True, features=feats)
            print(f"新建数据集：{root}")

    def save_episode(self):
        if not self.frames:
            self.status = "empty episode, discarded"
            return
        if len(self.frames) < MIN_EPISODE_FRAMES:
            print(f"  丢弃过短 episode（{len(self.frames)} 帧 < {MIN_EPISODE_FRAMES}），"
                  f"多半是机械臂还没走到待命位就按了空格")
            self.status = f"too short ({len(self.frames)} frames), discarded"
            self.frames.clear()
            return
        # 视频编码在这里面，几十秒。面板这一帧还没机会刷新，得先在终端喊一声，
        # 不然用户只看到窗口冻住、以为崩了。
        print(f"正在存盘 {len(self.frames)} 帧（视频编码约 1 分钟，窗口会暂时冻结，"
              f"别关窗口）...")
        for f in self.frames:
            f["task"] = paths.TASK
            self.ds.add_frame(f)
        self.ds.save_episode()
        self.saved += 1
        self.status = f"saved #{self.saved} ({len(self.frames)} frames)"
        self.frames.clear()
        print(f"  存好了，共 {self.saved} 条，可以继续采集或直接关窗口退出")

    # ---- 输入 → 目标角 ----

    def input_step(self, inp, dt, mouse_ok=True):
        """mouse_ok=False 时整帧忽略左右键（「点回窗口」那一下不算操作）。"""
        self.held = [k for k in HELD_KEYS if inp.down(k)]
        self.fine = inp.down("SHIFT")
        s = FINE_SCALE if self.fine else 1.0

        dx, dy = inp.take_delta()
        if dx or dy:
            self.tgt[0] += PAN_SIGN * dx * DEG_PER_COUNT * s
            self.tgt[1] += LIFT_SIGN * dy * DEG_PER_COUNT * s

        n = inp.take_wheel()
        if n:
            self.tgt[5] += n * WHEEL_DEG * s

        delta = np.zeros(6)
        if inp.down("W"):
            delta[2] += self.rate[2] * dt * s
        if inp.down("S"):
            delta[2] -= self.rate[2] * dt * s
        if inp.down("Q"):
            delta[3] += self.rate[3] * dt * s
        if inp.down("E"):
            delta[3] -= self.rate[3] * dt * s
        if inp.down("A"):
            delta[4] += self.rate[4] * dt * s
        if inp.down("D"):
            delta[4] -= self.rate[4] * dt * s
        # 左右键回夹爪：左键按住 = 闭合，右键按住 = 张开（跟滚轮上滚张开同向，
        # 右键那一路就是滚轮的「慢速版」，适合一点点夹住方块）。
        if mouse_ok:
            if inp.down("LBTN"):
                delta[5] -= self.rate[5] * dt * s
            if inp.down("RBTN"):
                delta[5] += self.rate[5] * dt * s
        self.tgt[2:] += delta[2:]
        self.tgt[:] = np.clip(self.tgt, self.lo, self.hi)

        if inp.pressed("R"):
            self.tgt[:] = self.home
            self.status = "recentering to ready pose"
        if inp.pressed("SPACE"):
            if self.recording:
                # 结束 = 存盘（视频编码几十秒，见 save_episode 里的提示）
                self.recording = False
                self.save_episode()
            else:
                # 开始：先清掉上一轮可能残留的帧，再开录
                self.recording = True
                self.frames.clear()
                self.status = "recording..."
        if inp.pressed("BACK"):
            self.reset_scene()
            self.recording = False
            self.status = "scene reset"

    def slew(self, dt, tgt=None):
        """把 cmd 朝目标推一步，并限速。自检模式直接给 tgt。"""
        t = self.tgt if tgt is None else tgt
        step = SLEW_RATE * dt
        self.cmd[:] = np.clip(t, self.cmd - step, self.cmd + step)
        self.cmd[:] = np.clip(self.cmd, self.lo, self.hi)

    def control_step(self):
        self.env.set_action(self.cmd)

    def record_step(self, obs):
        """obs 由主循环每帧渲染一次后传进来 —— 面板和录制共用同一份，不重复渲染。

        别为了面板另开一个小尺寸的 renderer：render_observation 内部只缓存一个
        _renderer、尺寸取首次调用的那次，换尺寸不会重建（只会渲染出错的尺寸）。
        """
        if not self.recording:
            return
        self.frames.append({
            "observation.state": self.env.state_deg(),
            "action": self.cmd.astype(np.float32),
            f"observation.images.{paths.CAM_FRONT}": obs[f"observation.images.{paths.CAM_FRONT}"],
            f"observation.images.{paths.CAM_WRIST}": obs[f"observation.images.{paths.CAM_WRIST}"],
        })

    # ---- 面板文字（必须是 ASCII：cv2.putText 画中文是问号）----

    # 256px 高、行距 15px、首行基线 16px：能排下的行数是 16（末行基线 241）。
    # 别再往上加 —— 加过 16 就会被 _panel 的窗口切片悄悄裁掉最后几行。
    # test_teleop_input.py 里按像素核过这个数。
    PANEL_LINES = 16

    def panel_lines(self):
        st = self.env.state_deg()
        # 接触数要现算（mj_collision），而这函数每帧都调。正常跑在 mj_step 之后
        # 一定没问题，但万一某帧状态退化它抛的是 FatalError —— 那条路会直接把
        # 整场录制带走，不值得为一个面板数字冒险。
        try:
            ncon = self.env.n_cube_contacts()
            placed = self.env.is_success()
        except Exception:                            # noqa: BLE001
            ncon, placed = -1, False

        rec = f"REC {len(self.frames)}f" if self.recording else "idle"
        # 空格是**切换**键：只写「SPACE=rec」看不出按下去是开始还是结束，得写
        # 「下一次按会发生什么」。面板是给人扫一眼的，状态和动作要在一行里说全。
        space = ("SPACE -> STOP and save" if self.recording
                 else "SPACE -> START recording")
        if not self.active:
            mouse = ("ESC: mouse released - click window or Esc" if self.esc_free
                     else "FROZEN - click window to control")
        elif self.locked:
            mouse = "mouse LOCKED (Esc to free)"
        else:
            mouse = "mouse FREE"
        lines = [
            f"{rec}   saved={self.saved}",
            space,
            mouse,
            f"held {','.join(self.held) if self.held else '-'}"
            f"   fine(Shift) {'ON' if self.fine else 'off'}",
        ]
        for i, (name, _, _, _, ch) in enumerate(JOINTS):
            # 实际/目标：两个数差得多说明这个舵机顶住了（扭矩不够或被挡住），
            # 光看实际角会以为是自己没按对键。
            lines.append(f"{name:<13}{st[i]:6.1f}/{self.cmd[i]:5.1f} {ch}")
        lines += [
            f"place={'OK' if placed else '--'}   cube_n={ncon}",
            "mouse = pan/lift   wheel = gripper   LMB/RMB = close/open",
            "W/S = elbow_flex   Q/E = wrist_flex   A/D = wrist_roll",
            "Shift = fine (25%)   R = ready pose   BKSP = reset scene",
            "Esc = free mouse  (Esc / click window to relock)",
        ]
        return lines


# ======================= 预览窗口 =======================

class Preview:
    """唯一的窗口，四宫格：front（主视角）/ wrist / top（操作员俯视）/ 文字面板。

    front 和 wrist 走环境自己的离屏 renderer；top 是额外的 MjvCamera 自由视角，
    **只在这里显示，不进数据集**。
    """

    TITLE = "SO-101 preview"
    PANEL_SCALE = 0.42

    def __init__(self, inp, enabled=True):
        self.on = False
        self.hwnd = None
        self.why = ""
        self._cv2 = None
        self._inp = inp
        if not enabled:
            self.why = "已关掉"
            return
        try:
            import cv2
        except Exception as e:                       # noqa: BLE001
            self.why = f"cv2 不可用：{e}"
            return
        self._cv2 = cv2
        try:
            cv2.namedWindow(self.TITLE, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.TITLE, self._on_mouse)
            cv2.imshow(self.TITLE, np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8))
            cv2.waitKey(1)
        except Exception as e:                       # noqa: BLE001
            self.why = f"开窗口失败：{e}"
            self._cv2 = None
            return
        self.on = True
        self.hwnd = ctypes.windll.user32.FindWindowW(None, self.TITLE)

    def _on_mouse(self, event, x, y, flags, param):
        """cv2 在 waitKey 里泵消息时回调（和主循环同线程，不用加锁）。

        只关心滚轮：光标被锁定后指针事件本来就没有意义，移动量走
        GetCursorPos 的差值，比回调更早、更全。
        """
        if event != self._cv2.EVENT_MOUSEWHEEL:
            return
        # WM_MOUSEWHEEL 的增量在高位字（有符号），±120 一格
        d = ctypes.c_short((flags >> 16) & 0xFFFF).value
        if d == 0:                                   # 万一 OpenCV 把 flags 换算了
            d = 1 if flags > 0 else -1
        n = max(1, abs(d) // 120)
        self._inp.add_wheel(n if d > 0 else -n)      # 上滚（远离自己）= 张开

    def _tile(self, img, size, label):
        """MuJoCo 给 RGB，cv2 要 BGR；标签描黑边，浅色背景上也看得清。"""
        cv2 = self._cv2
        t = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        t = np.ascontiguousarray(t[:, :, ::-1])
        for thick, color in ((3, (0, 0, 0)), (1, (255, 255, 255))):
            cv2.putText(t, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, thick, cv2.LINE_AA)
        return t

    def _panel(self, lines):
        cv2 = self._cv2
        p = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
        for i, s in enumerate(lines[:TeleopSession.PANEL_LINES]):
            cv2.putText(p, s, (6, 16 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        self.PANEL_SCALE, (200, 255, 200), 1, cv2.LINE_AA)
        return p

    def show(self, obs, top_img, lines):
        if not self.on:
            return
        cv2 = self._cv2
        try:
            # 关掉窗口的话再 imshow 会抛异常，这里先探一下，关了就把预览停掉
            if cv2.getWindowProperty(self.TITLE, cv2.WND_PROP_VISIBLE) < 1:
                self.on = False
                self.why = "窗口被关掉了"
                return
            front = self._tile(obs[f"observation.images.{paths.CAM_FRONT}"],
                               FRONT_TILE, f"{paths.CAM_FRONT} (front)")
            wrist = self._tile(obs[f"observation.images.{paths.CAM_WRIST}"],
                               SMALL_TILE, f"{paths.CAM_WRIST} (wrist)")
            top = self._tile(top_img, SMALL_TILE, "top (operator only)")
            # 左大图 | 右上两小图并排 | 右下文字
            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
            canvas[:, :FRONT_TILE] = front
            canvas[:SMALL_TILE, FRONT_TILE:FRONT_TILE + SMALL_TILE] = wrist
            canvas[:SMALL_TILE, FRONT_TILE + SMALL_TILE:] = top
            canvas[SMALL_TILE:, FRONT_TILE:] = self._panel(lines)
            cv2.imshow(self.TITLE, canvas)
            cv2.waitKey(1)          # 同时在这里泵消息，上面的鼠标回调才会被调用
        except Exception as e:                       # noqa: BLE001
            self.on = False
            self.why = f"显示失败，预览已停：{e}"

    def close(self):
        if self._cv2 is not None:
            try:
                self._cv2.destroyWindow(self.TITLE)
                self._cv2.waitKey(1)
            except Exception:                        # noqa: BLE001
                pass
        self.on = False


# ======================= 主循环 =======================

def run(dataset_root, selftest=False, preview=True):
    sess = TeleopSession(dataset_root, selftest)
    sess.open_dataset()

    m, d = sess.m, sess.d

    # 开窗之前先让机械臂真的走到起始姿态。reset() 只把 6 个目标角写进 ctrl，
    # qpos 还停在模型的默认零位（手腕转到底、小臂折起来的一团），不推进物理的话
    # 窗口一打开就是那副样子 —— 看着像坏了，其实只是没跑仿真。
    sess.env.step(sess.home, n_frames=35)      # ≈1.2s：实测 0.5s 时 wrist_roll 还差 28°
    # 把腕部相机的画面摆正到「初始时水平对着地平线」。
    # 必须在 step() 之后、建 renderer 之前 —— 它按 home 姿态算，跟当前 qpos 无关，
    # 但早于窗口打开才能保证操作员看到的第一帧就是正的（而不是闪一下才转过来）。
    sess.env.level_wrist_cam(sess.home)

    # 先点一次观测，把离屏 renderer 建起来（首次约 0.5s，别让它落在录制里）。
    # 俯视图的 render_view() 复用的就是它，所以这一步必须先跑。
    sess.env.observation()
    top_cam = sess.env.top_cam()

    inp = WinInput()                 # 先调 SetProcessDPIAware，再让 cv2 建窗口
    prev = Preview(inp, enabled=preview)
    if not prev.on:
        print(f"❌ 预览窗口没起来（{prev.why}）—— 它是唯一的显示和输入面，退出")
        return 1
    if not prev.hwnd:
        print("❌ 拿不到预览窗口句柄，退出")
        return 1
    inp.bind(prev.hwnd)
    rect = inp.client_rect()
    l, t, r, b = rect
    print(f"预览窗口 hwnd={prev.hwnd} 客户区 {r-l}x{b-t} @ ({l},{t})"
          f"  （画面应为 {CANVAS_W}x{CANVAS_H}）")
    if abs((r - l) - CANVAS_W) > 2 or abs((b - t) - CANVAS_H) > 2:
        print("⚠️ 客户区尺寸和画面尺寸对不上，鼠标增量会偏（DWM 缩放？）")
    print(f"鼠标：相对锁定，{DEG_PER_COUNT}°/计数，按住 Shift 降到 "
          f"{FINE_SCALE*100:.0f}%")
    print("各舵机通道：" + "  ".join(f"{j[0]}={j[4]}" for j in JOINTS))

    if selftest:
        sess.recording = True
        sess.status = "selftest recording"
        print(f"[自检] 逐个舵机扫行程，共 {6 * SEG_S:.0f} 秒，然后存一条")

    dt = 1.0 / paths.FPS
    t_start = time.time()
    n_frame = 0
    # 自检：每个舵机实际到达的最小/最大角，以及 tgt 自己走到过的范围。
    # 两个都要看 —— tgt 必须严格落在上下限内（越界就是映射写错了），
    # 实测角度则允许几度超调，那是伺服跟随时本来就有的。
    track = {j[0]: [999.0, -999.0] for j in JOINTS}
    tgt_track = {j[0]: [999.0, -999.0] for j in JOINTS}
    was_focused = False
    was_active = False
    try:
        while prev.on:
            t0 = time.perf_counter()

            if selftest:
                ph = n_frame / paths.FPS
                seg = int(ph / SEG_S) % 6
                wv = wave_at(ph / SEG_S)
                t = sess.home.copy()
                t[seg] = sess.lo[seg] + wv * (sess.hi[seg] - sess.lo[seg])
                sess.tgt[:] = t        # 目标也要落到 tgt 上，自检才检查得到
                sess.slew(dt, t)       # 走同一条限速通路，轨迹才真实
                if time.time() - t_start > 6 * SEG_S:
                    break
                sess.active = True
            else:
                # 焦点决定一切：窗口在前台才锁光标、才收输入。切走或按 Esc 就
                # 立刻还你一个正常的光标 —— 这条比任何显示都重要，锁死的光标
                # 没法操作电脑。
                #
                # esc_free 有三个出口，缺一不可：Esc 是**来回切**（不是单向置位），
                # 在窗口里点一下也解锁，焦点重新进来也解锁。
                # 曾经的写法 `if focused and not was_focused: esc_free = False`
                # 是个单向闩：窗口一直有焦点时 was_focused 恒为 True，点回已经
                # 有焦点的窗口并不会改 GetForegroundWindow()，于是 esc_free 永远
                # 清不掉，整个采集从此失去响应（用户报的「鼠标回不来、按键卡住」）。
                focused = inp.focused
                if focused and not was_focused:
                    sess.esc_free = False             # 焦点回来 = 重新锁定
                if focused and inp.pressed("ESC"):
                    sess.esc_free = not sess.esc_free
                if sess.esc_free and focused and inp.pressed("LBTN"):
                    sess.esc_free = False             # 在窗口里点一下 = 重新锁定
                was_focused = focused
                if focused and not sess.esc_free:
                    inp.lock(LOCK_PT)
                else:
                    inp.unlock()
                sess.locked = inp.locked
                sess.active = focused and not sess.esc_free
                # 「点回窗口重新锁定」和「点一下把后台窗口拉到前台」这两种点击走的是
                # 同一条 LMB 按下沿。要是这一帧照常喂给 input_step，用户每次回来
                # 都会顺手夹一下（甚至夹住方块）。所以重新取得控制权的**第一帧**
                # 整个鼠标按键屏蔽掉 —— 按住不放的话从第二帧起照常生效。
                just_activated = sess.active and not was_active
                was_active = sess.active
                if sess.active:
                    sess.input_step(inp, dt, mouse_ok=not just_activated)
                    sess.slew(dt)
                # latch 必须排在 input_step **之后**。pressed() 是拿「当前状态」和
                # 上一帧记下的 _prev 比，latch 要是先跑，input_step 里比的就是它自己，
                # 恒为 False —— R / 空格 / 退格 会全部失灵（踩过，而且很隐蔽：
                # 上面那几行 Esc/LBTN 因为排在 latch 之前，一直是好的）。
                # 也必须在 continue 之前跑，否则失焦期间按下的键会在重新聚焦那一帧
                # 被当成「刚按下」，凭空切一次录制或回一次 home。
                inp.latch()

                if not sess.active:
                    # 冻结时画面照样刷新（不然窗口在你点它之前是块黑板，分不清
                    # 是没反应还是在等焦点），但降到 ~10fps 省得白白渲染。
                    if n_frame % 3:
                        time.sleep(0.03)
                        n_frame += 1
                        continue

            sess.control_step()
            for _ in range(sim_env.STEPS_PER_FRAME):
                mujoco.mj_step(m, d)

            # 每帧渲染一次，面板和录制共用同一份
            obs = sess.env.observation()
            # 冻结期间不录：那时物理根本没推进，记进去的是一串一模一样的帧。
            if sess.active:
                sess.record_step(obs)
            # 俯视图单独渲 —— 它走 render_view 而不是 observation()，所以永远
            # 不会出现在 obs 里，也就永远进不了数据集。
            prev.show(obs, sess.env.render_view(top_cam), sess.panel_lines())

            if selftest:
                a = sess.env.state_deg()
                for i, j in enumerate(JOINTS):
                    tr, gr = track[j[0]], tgt_track[j[0]]
                    tr[0], tr[1] = min(tr[0], a[i]), max(tr[1], a[i])
                    gr[0], gr[1] = min(gr[0], sess.tgt[i]), max(gr[1], sess.tgt[i])
            n_frame += 1

            rest = dt - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)
    except KeyboardInterrupt:
        print("\n收到中断")
    finally:
        inp.unlock()                     # 无论如何都要把光标还回去
        # 时长必须在存盘**之前**算：视频编码要几十秒，算进去会得出假的低帧率
        elapsed = max(1e-9, time.time() - t_start)
        t_enc = time.time()
        if sess.recording:
            sess.save_episode()          # 视频编码在这里面，不在 finalize 里
        if sess.ds is not None:
            sess.ds.finalize()
        if sess.saved:
            print(f"存盘/编码耗时 {time.time()-t_enc:.1f}s")
        prev.close()

    if selftest:
        print("[自检] 每个舵机的行程（实际角度 / 目标角度，单位都是舵机度）：")
        bad = []
        for i, j in enumerate(JOINTS):
            lo, hi = track[j[0]]
            glo, ghi = tgt_track[j[0]]
            span, gspan = hi - lo, ghi - glo
            # 目标必须严格在限内（越界 = 映射写错）；实际幅度至少要走到行程的 6 成，
            # 否则就是「这个舵机根本没被驱动」。
            ok = (glo >= j[2] - 0.01 and ghi <= j[3] + 0.01
                  and gspan > 0.6 * (j[3] - j[2]))
            print(f"    {j[0]:<14}{j[4]:<10} 实际 {lo:6.1f}~{hi:6.1f}"
                  f"  目标 {glo:6.1f}~{ghi:6.1f}  限 [{j[2]:.0f},{j[3]:.0f}]"
                  f"  {'OK' if ok else '!! 有问题'}")
            if not ok:
                bad.append(j[0])
        print("[自检] " + ("全部正常" if not bad else f"异常：{bad}"))

    print(f"共录入 {sess.saved} 条 episode，主循环 {n_frame} 帧 / {elapsed:.1f}s "
          f"≈ {n_frame / elapsed:.1f} fps")
    print(f"数据集：{dataset_root}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="SO-101 遥操作采集")
    ap.add_argument("--root", default=str(paths.DATA_ROOT.parent / "so101_smolvla_sim_teleop"))
    ap.add_argument("--selftest", action="store_true",
                    help="自检：每个舵机各扫一遍行程，自动存一条并退出")
    ap.add_argument("--no-preview", action="store_true",
                    help="关掉预览窗口（那样就没有窗口了，只会直接退出）")
    a = ap.parse_args()
    sys.exit(run(a.root, a.selftest, preview=not a.no_preview))


if __name__ == "__main__":
    main()
