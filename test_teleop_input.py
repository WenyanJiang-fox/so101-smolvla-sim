"""回归测试：鼠标左右键 = 夹爪，以及 latch 顺序（R/空格/退格 能不能触发）。

不开窗口、不建 MuJoCo 场景：直接喂一个假输入源给 TeleopSession.input_step。
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np

import teleop_collect as tc


class StubInput:
    """只实现 input_step 真正用到的那几个方法。"""

    def __init__(self, keys=(), wheel=0, delta=(0, 0)):
        self.state = set(keys)
        self._prev = set()
        self._wheel = wheel
        self._delta = delta
        self.just_pressed = set()      # 本帧当作「刚按下」的键

    def down(self, name):
        return name in self.state

    def pressed(self, name):
        return name in self.just_pressed

    def take_delta(self):
        return self._delta

    def take_wheel(self):
        n, self._wheel = self._wheel, 0
        return n

    def latch(self):
        self._prev = set(self.state)


def make_session():
    """不碰环境的会话：只测输入→目标角那一段。"""
    s = tc.TeleopSession.__new__(tc.TeleopSession)
    s.rate = np.array([tc.RATE.get(j[0], 0.0) for j in tc.JOINTS], dtype=float)
    s.home = np.array([j[1] for j in tc.JOINTS], dtype=float)
    s.lo = np.array([j[2] for j in tc.JOINTS], dtype=float)
    s.hi = np.array([j[3] for j in tc.JOINTS], dtype=float)
    s.tgt = s.home.copy()
    s.cmd = s.home.copy()
    s.held, s.fine = [], False
    s.recording, s.frames, s.saved, s.status = False, [], 0, ""
    s.reset_scene = lambda: None
    return s


def run(fn):
    try:
        fn()
    except AssertionError as e:
        print(f"  FAIL {fn.__name__}: {e}")
        return False
    print(f"  pass {fn.__name__}")
    return True


DT = 1.0 / 30.0


def test_lmb_closes_gripper():
    s = make_session()
    s.input_step(StubInput(keys={"LBTN"}), DT)
    want = s.home[5] - s.rate[5] * DT
    assert abs(s.tgt[5] - want) < 1e-9, f"左键应该闭合到 {want}，得到 {s.tgt[5]}"
    assert abs(s.tgt[5] - s.home[5]) > 2.0, "一帧就该动 3°（90°/s @30fps）"


def test_rmb_opens_gripper():
    s = make_session()
    s.input_step(StubInput(keys={"RBTN"}), DT)
    assert s.tgt[5] > s.home[5], "右键应该张开"


def test_buttons_respect_limits():
    s = make_session()
    s.tgt[5] = s.lo[5] + 1.0
    s.input_step(StubInput(keys={"LBTN"}), DT)
    assert s.tgt[5] == s.lo[5], f"下限没夹住：{s.tgt[5]}"


def test_fine_scale_applies_to_buttons():
    s = make_session()
    s.input_step(StubInput(keys={"LBTN", "SHIFT"}), DT)
    coarse = make_session()
    coarse.input_step(StubInput(keys={"LBTN"}), DT)
    d_fine = s.home[5] - s.tgt[5]
    d_coarse = coarse.home[5] - coarse.tgt[5]
    assert abs(d_fine / d_coarse - tc.FINE_SCALE) < 1e-9, \
        f"Shift 该是 {tc.FINE_SCALE} 倍，实测 {d_fine / d_coarse}"


def test_mouse_ok_false_ignores_buttons():
    s = make_session()
    s.input_step(StubInput(keys={"LBTN", "RBTN"}), DT, mouse_ok=False)
    assert np.allclose(s.tgt, s.home), f"点回窗口那一帧不该动夹爪：{s.tgt}"


def test_buttons_do_not_touch_other_joints():
    s = make_session()
    s.input_step(StubInput(keys={"LBTN", "RBTN"}), DT)
    assert np.allclose(s.tgt[:5], s.home[:5]), "左右键不该动前 5 个舵机"


def test_r_recenters():
    s = make_session()
    s.tgt[:] = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
    inp = StubInput()
    inp.just_pressed = {"R"}
    s.input_step(inp, DT)
    assert np.allclose(s.tgt, s.home), f"R 没回待命：{s.tgt}"


def test_space_toggles_recording():
    s = make_session()
    saved = []
    s.save_episode = lambda: saved.append(len(s.frames))
    inp = StubInput()
    inp.just_pressed = {"SPACE"}
    s.input_step(inp, DT)
    assert s.recording, "空格没开始录制"
    s.frames = [1, 2, 3]
    s.input_step(inp, DT)
    assert not s.recording, "空格没停止录制"
    assert saved == [3], f"空格停止时没存盘（save_episode 调用记录 = {saved}）"


def test_back_resets_scene():
    s = make_session()
    called = []
    s.reset_scene = lambda: called.append(1)
    s.recording = True
    inp = StubInput()
    inp.just_pressed = {"BACK"}
    s.input_step(inp, DT)
    assert called and not s.recording, "退格没复位场景"


def test_latch_ordering_in_source():
    """latch() 必须在 input_step 之后 —— 顺序一反，pressed() 就恒为 False，
    R / 空格 / 退格 会集体失灵（用户报的「按键没了」就是这个）。"""
    src = Path(tc.__file__).read_text(encoding="utf-8")
    body = src[src.index("while prev.on:"):]
    i_step = body.index("sess.input_step(")
    i_latch = body.index("inp.latch()")
    assert i_latch > i_step, "inp.latch() 跑到 input_step 前面去了"


def test_held_keys_include_buttons():
    assert "LBTN" in tc.HELD_KEYS and "RBTN" in tc.HELD_KEYS, \
        "HELD_KEYS 里没有鼠标键，面板不会显示它们"


class StubEnv:
    def state_deg(self):
        return np.full(6, 90.0)

    def n_cube_contacts(self):
        return 0

    def is_success(self):
        return False


def test_panel_fits():
    """面板一行排不下就被 _panel 的切片悄悄裁掉 —— 直接问真正的 panel_lines()，
    并按像素核，不手数行数（手数的常数最容易在改文案时过期）。"""
    s = make_session()
    s.env = StubEnv()
    s.active, s.locked = True, True
    lines = s.panel_lines()
    n = len(lines)

    assert n <= tc.TeleopSession.PANEL_LINES, \
        f"面板 {n} 行 > PANEL_LINES={tc.TeleopSession.PANEL_LINES}，会被裁掉"

    # 行数上限本身要真的排得下：首行基线 16px、行距 15px、字高约 11px
    bottom = 16 + 15 * (tc.TeleopSession.PANEL_LINES - 1) + 11
    assert bottom <= tc.PANEL_H, \
        f"PANEL_LINES={tc.TeleopSession.PANEL_LINES} 会画到 {bottom}px，超出 {tc.PANEL_H}px"

    # 宽度同理：最长的行不能顶出右边界
    for ln in lines:
        w = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX,
                            tc.Preview.PANEL_SCALE, 1)[0][0]
        assert 6 + w <= tc.PANEL_W, \
            f"这行宽 {6 + w}px 顶出 {tc.PANEL_W}px：「{ln}」"


def test_panel_says_what_space_does():
    """空格是切换键，面板得写清「下一次按是开始还是结束」（用户提的）。"""
    s = make_session()
    s.env = StubEnv()
    s.active, s.locked = True, True

    s.recording = False
    start = [ln for ln in s.panel_lines() if "SPACE" in ln]
    assert len(start) == 1, f"面板里 SPACE 的行数不对：{start}"
    assert "START" in start[0], f"没在录时应写 START：{start[0]}"
    assert "STOP" not in start[0], f"没在录时不该出现 STOP：{start[0]}"

    s.recording = True
    stop = [ln for ln in s.panel_lines() if "SPACE" in ln]
    assert len(stop) == 1, f"面板里 SPACE 的行数不对：{stop}"
    assert "STOP" in stop[0], f"在录时应写 STOP：{stop[0]}"
    assert "START" not in stop[0], f"在录时不该出现 START：{stop[0]}"


def test_short_episode_discarded():
    """过短 episode（机械臂还没走到待命位就按空格）不进数据集。"""
    s = make_session()
    calls = []
    s.ds = type("StubDS", (), {
        "add_frame": lambda self, f: calls.append(f),
        "save_episode": lambda self: calls.append("save"),
    })()
    s.frames = [1] * (tc.MIN_EPISODE_FRAMES - 1)   # 比阈值少 1 帧
    s.save_episode()
    assert s.saved == 0 and not s.frames, "过短 episode 没被丢弃"
    assert not calls, f"过短 episode 不该写进数据集，但 ds 被调了：{calls}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = all([run(t) for t in tests])
    print(("全部通过" if ok else "有失败") + f"（{len(tests)} 项）")
    sys.exit(0 if ok else 1)
