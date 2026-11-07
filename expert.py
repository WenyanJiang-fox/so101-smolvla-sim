"""scripted expert：在窄柱场景里做一次「取→放」，纯物理（weld 全程关闭）。

这是数据采集的动作来源。设计要点：
- 用 MujocoServoController 的 headless IK **只算关节角**（headless 下它会瞬移 qpos），
  拿到路点后自己驱动 d.ctrl 逐步 mj_step，做确定性推进、不依赖 wall-clock。
- 不调用 start_physics_loop()，否则 set_servo_angle 不再瞬移、IK 路点就算不出来。
"""
from __future__ import annotations

import sys

import mink
import mujoco
import numpy as np

import paths
import sim_env

sys.path.insert(0, str(paths.MCP_SOURCES))
from mujoco_servo_controller import (  # noqa: E402
    HOME_ANGLES_DEG,
    MujocoServoController,
    deg_to_qpos,
)

J6 = paths.JOINT_NAMES
GRIP_OPEN_DEG = 180.0
GRIP_CLOSE_DEG = 0.0
GRIP_PRELOAD_DEG = 5.0    # 夹住后维持的额外闭合量，见 run_episode 里的说明

# 松手时的**额外**抬高量（米），加在"方块正好落到静止高度"之上。默认 0 = 就在静止
# 高度松手；想让它从高处掉下来就传正值。
#
# 原值是 0.05，但那时候它承担的是**另一个**职责：假装方块中心比 grasp_point 低 50mm，
# 于是 z_bot = CUBE_Z + 0.05 就恰好让方块落在 20mm。问题是那个 50mm 是写死的，
# 而方块在爪里滑落之后实测是 55mm——差这 5mm 让下探多压 5mm 进台面，把方块压歪、
# 松手时推走 20mm。现在挂低多少由 grasp_offset() 实测（见 plan_place 的调用处），
# 这个常数退回它字面意思：额外抬高。
RELEASE_LIFT = 0.0

# 夹爪指尖比 grasp_point 站点低这么多（mesh 真实顶点量出来的；sim_scene.xml 注释里
# 写的 64mm 是错的）。爪板就从 grasp_point 往下伸这么长。
FINGER_DROP = 0.0469

# 下探列底端（夹持点瞄准高度）。抬到 85mm = 只夹方块上半段，让**闭合的爪**离台面有富余。
#
# 为什么是抬高而不是别的：压台面的不是手臂，是**闭爪那一下**——动爪绕铰链转 -23.4°，
# 最低顶点向下扫 15~36mm（扫程随手臂伸出长度变）。夹持点设在方块中心（20mm）时闭合
# 爪尖落到 -1.1mm、插进工作面 1.1mm 并停 3.2 秒；真机同样的几何，= 每抓一次让夹爪
# 舵机堵转三秒。
#
# **这只是默认值和解算起点，不要靠调它来避坑。** 真正决定夹持高度的是
# solve_pick_z()：它在每条 episode 的实际姿态下反推 z_bot（见那里的说明）。
#
# 85mm 是在**默认方块位置**（250,-50、偏航 0）标出来的（_grip_depth.py）：张开爪尖
# 28.2mm → 闭合爪尖 +13.1mm。但那个 +13.1mm 只对**那个姿态**成立——方块挪到
# (280,-20)、偏航 20° 后手臂伸出长度变了，闭爪下扫更深，13mm 余量被吃穿、爪子压台面。
# 试过把它抬到 100mm：不压了，但爪整个跑到方块顶面之上，夹不到、只把方块带倒。
# 两头都撞墙说明固定常数无解——同一个 z_bot 在不同位置对应不同的实际爪尖高度。
# 所以它退回 85mm 当起点，交给 solve_pick_z() 按姿态解。
PICK_Z_BOT = 0.085
PICK_Z_TOP = PICK_Z_BOT + 0.08     # 列顶 165mm，横移时爪尖远高于方块顶面

# solve_pick_z() 的目标：**闭合**爪尖离工作面的间隙。解出来的 z_bot 让闭合爪尖正好
# 落在这个高度——即"夹得最深、又不压台面"的那个位置。
#
# 为什么不是 0（贴着台面）：这是**静态运动学**估的值（摆好姿态 mj_forward 读顶点），
# 而真实闭合有动力学——位置伺服压着闭、方块被夹时会被下压几个毫米，末端也会沉一点。
# 设成 3mm 时 episode4 实测仍然蹭到台面（方块被弹飞），所以留到 10mm 当余量。
# 再大就会把 z_bot 顶高、夹持面积变小，两头拉扯的点在 10mm 附近。
TIP_CLEARANCE = 0.010              # 10mm

# 腕转对齐：抓之前先把 wrist_roll 转到跟方块棱平行，让爪做**面抓**而不是角抓。
# 方块偏航 θ 时它在爪开合轴上的跨距是 40·(|cosθ|+|sinθ|)：θ=45° 时 56.6mm 比爪全开
# 55.9mm 还大（伸不进），θ=30° 时 54.6mm 刚好塞得进但夹的是**角**，横移时滑脱。
# 对齐后跨距恒为 40mm。这是加宽偏航抖动的前提，关掉就只能在小偏航角下采。
#
# **实测关掉了**：同一个抖动样本（±30mm、偏航 41°）A/B 对照 ——
#   关：at_pick 2 触点 → closed 2 → lifted 4 → 终点 137.4,53.9  ✅
#   开：at_pick 4 触点 → closed 7 → lifted 0 → 终点 279.0,-86.4 ❌
# 开着闭合时触点反而更多（7 个），但**抬升那一下掉**、方块退回桌面：腕角一转，
# 下探时爪板改成从方块侧面刮过去，把方块铲到爪尖上（at_pick 时方块已经被顶高
# 4.8mm、横移 16mm），夹住的是方块顶棱而不是腰，一抬就脱。
# 跑 5 条的冒烟测试也是对不开：开着 1/5，关着 5/6。
# 结论：对齐对"爪能不能伸进去"是对的，但这条下探轨迹没跟着改，反而更差。
# 要用它得连下探列一起重做（爪尖在别的高度、别的接触位置），不是开个开关的事。
ALIGN_WRIST = False

# 各阶段的推进时长（秒），probe 验证过的慢速参数
DUR_APPROACH = 2.0     # 空手横移/升到方块上方
DUR_DESCEND = 2.5      # 下探到夹持位（慢，避免撞飞方块）
DUR_CLOSE = 0.8
DUR_SETTLE = 0.6
DUR_LIFT = 2.5         # 带方块抬起（慢，避免甩脱）
                       # **别再加长它**。试过 6.0（≈13mm/s）：抬升中偏移在 42.5mm 上
                       # 稳稳停了 11 秒，然后在抬升段突然失稳，方块整个掉出去
                       # （0 触点，留在取件点 298,-70）。所以这里的滑落是**重力对
                       # 摩擦的缓慢蠕动**，不是惯性甩脱——放慢等于给重力更多时间赢。
                       # 短时间(0.84s)内滑 25mm 但夹得住，长时间(>11s)反而脱手。
DUR_TRANSFER = 3.0     # 带方块横移到放置台上方
DUR_PLACE_DESCEND = 2.5
DUR_RELEASE = 0.8
DUR_RETREAT = 2.0

# 单个舵机的速度上限（舵机度/秒）。**全程生效**：ramp / ramp_chain 算出这一段
# 需要转多少度，超过上限就自动把时长拉长，而不是照原时长硬推。所以下面那些
# DUR_* 是"最快也快不过这个"，不是"就这么快"。
#
# 为什么用时长而不是限幅 ctrl：限幅会让实际角度追不上命令角，路径被悄悄改掉
# （下探列就不走直线了）；拉长时长保持路径不变，只是走慢。
#
# 真机意义：SO-101 的 STS3215 位置伺服全速能到 600°/s 量级，快速摆臂会把方块
# 甩脱（_recovery.py 的"搬快了甩脱"就是故意用 60ms 横移做到的），也伤舵机。
# 120°/s 比现有轨迹本来就慢的部分松、比真会甩脱的动作紧。
#
# 注意：这也会管住 _recovery.py 的 drop_it —— 它靠 60ms 快速横移甩脱方块，
# 被限速后甩不掉。要用那招得显式绕过（ramp(..., duration=...) 之外再给开关）。
MAX_SERVO_DEG_S = 120.0

# ---- IK（mink） ----
# 换掉了原来对 scipy.optimize.least_squares 的裸调（见 mujoco_servo_controller._ik_angles）。
# 那个调用**从不检查 result.success / result.cost**，只查 isfinite —— 所以差 2mm 的解也
# 照原样返回、当成好解用。而 solve_pick_z() 是拿"闭合爪尖正好落在 TIP_CLEARANCE"反推
# 夹持高度的：IK 差 2mm，爪尖就低 2mm，正好踩进会蹭台面的那一档。实测默认取件点
# (250,-50,66.7) 上 scipy 残差 1.96mm、mink 0.015mm。下游那一堆防御常数（TIP_CLEARANCE、
# solve_pick_z 的来回迭代、逐点热启动锁分支）大部分是在补这个"解不准还看不出来"。
#
# mink 是 MuJoCo 原生的微分 IK（QP 形式），直接吃我们自己的 MjModel，顺带解决三件事：
#   1. 残差**可查** —— 不收敛就返回 None，不再假装成功（这是换它的首要理由）
#   2. 关节限位从 scipy 的 bounds 升级成 QP 硬约束；wrist_roll 用 DofFreezingTask 硬钉
#   3. 每步解最小速度，冗余自由度自然贴近种子 → 竖直列相邻点跳变实测 23° → 4°
# 单次求解 scipy 约 133ms（move_to 内部走 15 个路点）、mink 约 1ms：约 100×。
MINK_DT = 0.05           # 微分 IK 的积分步长（不是物理步长，只影响迭代速度）
MINK_MAX_ITERS = 200     # 单次求解的迭代上限
MINK_TOL = 1e-4          # 0.1mm：迭代到这个残差就停
MINK_TOL_HARD = 1e-3     # 1mm：迭代用完还没进这个精度就判失败（返回 None）

# 姿态权重（mink FrameTask 的 orientation_cost）。逐轴作用在 grasp_point **站点自己的
# 局部系**上——实测站点系和夹爪体系完全重合（R_bs = 单位阵）。
#
# 为什么是 0.01 而不是 1.0：SO-101 的臂只有 **5 个自由度**（第 6 个 joint 是夹爪开合，
# 不是臂的关节），在一个给定的 (x,y,z) 上能实现的姿态只有 **2 个自由度**（5−3），
# 所以"位置 + 姿态"这个 6 维任务是**结构性超定**的，硬解必然无解：
#   - 三轴都给 1.0：实测除了 R_ref 的标定点本身，其余位置**全部无解**（循环论证：
#     拿取件点标定的姿态去解取件点，当然解得出来）
#   - 给 (1,1,0)：接近方向收敛到 0.82°，但位置从第 500 次迭代到第 5000 次**卡死在
#     77.99mm 一动不动**——是关节限位在让步，不是收敛慢
#
# 0.01 是**全生态的统一答案**，不是我们试出来的：lerobot 官方
# `RobotKinematics.inverse_kinematics` 的签名就是 `position_weight=1.0,
# orientation_weight=0.01`；IB_Robot 的 `so101_placo_servo` 同样是位置主任务 +
# 朝向 0.01 软任务。含义是「**有余力就跟姿态，和位置冲突就立刻让步**」——QP 里
# 位置项比姿态项重 100 倍，所以位置照常收敛到亚毫米，姿态只在零空间里被顺带优化。
# 这比纯位置好：纯位置等于彻底放弃姿态，0.01 是让腕子尽量保持那个录出过成功
# episode 的姿态，但绝不为此牺牲位置。
MINK_ORIENT_COST = (0.01, 0.01, 0.01)

# 笛卡尔直线上的采样步长（米）。ik_line 用它把「两点之间的直线」密采样成路点。
#
# 为什么是 10mm：竖直列 ik_column 是 7 点跨 80mm ≈ 13mm 间距，实测那根列不会走弧
# 线。10mm 比它略密，因为横移的距离通常更长（取放点相距约 130mm → 14 个点）。
#
# 为什么按**长度**定路点数而不是写死个数（ik_column 写死 7）：横移距离随取放点变化，
# 写死个数等于让步长随距离变——取放点一拉开，步长就变大，相邻路点之间的关节插值又
# 退化成"走弧线"这个原始问题。
CART_STEP = 0.010


def _wrap(x):
    return float(np.arctan2(np.sin(x), np.cos(x)))


class Expert:
    """窄柱场景的 scripted 取放专家。"""

    def __init__(self, scene_xml: str | None = None, controller=None):
        xml = str(scene_xml or paths.SCENE_XML)
        # controller 可以由 SO101SimEnv 传进来共用一份仿真状态（采集时就是这么用的）
        self.c = controller or MujocoServoController(xml, headless=True)
        self.m, self.d = self.c.model, self.c.data
        # 离屏帧缓冲默认 480，渲染 512² 会报 "Image height 512 > framebuffer height 480"
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, paths.IMG_W)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, paths.IMG_H)
        self._disable_weld()
        self._resolve_ids()
        self.grip_hold = None   # 夹住后要维持的夹爪关节角（rad），见 _grip_ctrl
        self.pick_z_bot = PICK_Z_BOT   # 上一条 episode 按姿态解出的夹持高度
        self.measure_jaw_tips()        # 两个爪尖的本地坐标，之后靠它算离地高度
        sim_env.apply_grip_friction(self.m)   # 爪面高摩擦（独立实例时自己也得加）

        # 专用 IK 规划器：ik_pose() 依赖 move_to_grasp_point()，而它会 reset() 整个
        # 仿真状态。用独立实例算 IK，才不会把手上夹着的方块一起清掉（踩过这个坑）。
        self.p = MujocoServoController(xml, headless=True)
        for eq in range(self.p.model.neq):
            self.p.data.eq_active[eq] = 0
            self.p.model.eq_active0[eq] = 0
        # mink 的配置**惰性**建（见 _mink_setup）：它自己的 MjData，跟上面两个都无关。
        self._mink_c = None

    # ---- 初始化 ----

    def _disable_weld(self):
        """真实摩擦抓取的前提：weld 必须全程关闭，且 reset() 也不会重新打开。"""
        for eq in range(self.m.neq):
            self.d.eq_active[eq] = 0
            self.m.eq_active0[eq] = 0

    def _resolve_ids(self):
        m = self.m

        def jid(n):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

        self.acts = [next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == jid(n)) for n in J6]
        self.qidx = [m.jnt_qposadr[jid(n)] for n in J6]
        # wrist_roll 在速度空间的地址（qidx[4] 是它的 qpos 地址）。mink 钉腕角要的是 dof。
        self.wr_dofadr = m.jnt_dofadr[jid("wrist_roll")]
        self.grip_range = m.jnt_range[jid("gripper")]
        # 六个关节各自的行程，用来把关节角换算成**舵机度**（限速用）。
        # 舵机 0~180° 线性映到关节 [lo,hi]，所以 deg = (q-lo)/(hi-lo)*180。
        # _full 是六个关节再拼上夹爪（限速要连夹爪一起算，见 slow_enough）。
        lo_hi = np.array([m.jnt_range[jid(n)] for n in J6], dtype=float)
        self.jnt_lo, self.jnt_span = lo_hi[:, 0], np.maximum(lo_hi[:, 1] - lo_hi[:, 0], 1e-9)
        gspan = max(float(self.grip_range[1] - self.grip_range[0]), 1e-9)
        self.jnt_span_full = np.append(self.jnt_span, gspan)
        self.gp_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_point")
        self.cube_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.grip_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.jaw_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        self.gripper_bodies = {self.grip_b, self.jaw_b}
        # 爪上所有 geom（下探闭环要每步量它们的最低顶点，id 预先取好，别每步遍历）
        self._tip_geoms = [(g, int(m.geom_dataid[g])) for g in range(m.ngeom)
                           if m.geom_bodyid[g] in self.gripper_bodies]
        self.cube_qadr = self.m.jnt_qposadr[jid("cube")]

    # ---- 底层步进 ----

    def _grip_ctrl(self, deg):
        # grip_hold 一旦设上就接管：位置伺服会一直朝指令角加力，夹住方块后
        # 还继续压向 0° 会把它挤出去（实测方块在带件下探时被打飞 180mm）。
        if self.grip_hold is not None:
            return self.grip_hold
        lo, hi = self.grip_range
        return lo + (deg / 180.0) * (hi - lo)

    def drive(self, q, grip_deg):
        """写一帧 ctrl 并推进一步物理。"""
        for a, qq in zip(self.acts[:5], q[:5]):
            self.d.ctrl[a] = qq
        self.d.ctrl[self.acts[5]] = self._grip_ctrl(grip_deg)
        mujoco.mj_step(self.m, self.d)

    def grip_q(self, deg):
        """夹爪**舵机角**（0=闭/180=开）→ 夹爪关节角。"""
        lo, hi = self.grip_range
        return lo + (deg / 180.0) * (hi - lo)

    def slow_enough(self, duration, q_from, q_to, grip_from, grip_to):
        """舵机限速：这一段里转得最多的那个舵机决定最短时长。

        六个关节 + 夹爪一起算——夹爪 180° 全行程在 0.8s 里走完是 225°/s，
        比手臂任何一段都快，不一起管的话它自己就超了。
        """
        if MAX_SERVO_DEG_S <= 0:
            return duration
        a = np.append(np.asarray(q_from), self.grip_q(grip_from))
        b = np.append(np.asarray(q_to), self.grip_q(grip_to))
        deg = float(np.max(np.abs(b - a) / self.jnt_span_full * 180.0))
        return max(duration, deg / MAX_SERVO_DEG_S)

    def ramp(self, q_from, q_to, grip_from, grip_to, duration, on_frame=None,
             stop_when=None):
        """从 q_from 线性插值到 q_to，每步推进并回调 on_frame。

        stop_when 每步调一次，返回真就提前停、并返回**实际停住的那个 q**。
        下探就是靠它做闭环（见 TIP_CLEARANCE）：停住之后调用方必须拿这个返回值
        当后续路点，否则后面的指令又会把手臂压回原定的更深处。
        """
        duration = self.slow_enough(duration, q_from, q_to, grip_from, grip_to)
        n = max(1, int(duration / self.m.opt.timestep))
        q = q_from
        for k in range(n):
            f = (k + 1) / n
            q = q_from + (q_to - q_from) * f
            self.drive(q, grip_from + (grip_to - grip_from) * f)
            if on_frame is not None:
                on_frame()
            if stop_when is not None and stop_when():
                break
        return q

    def hold(self, q, grip_deg, duration, on_frame=None):
        self.ramp(q, q, grip_deg, grip_deg, duration, on_frame=on_frame)

    def ramp_chain(self, poses, grip_from, grip_to, total_duration, on_frame=None,
                   stop_when=None):
        """沿一串关节姿态连续推进（用于逼近笛卡尔竖直列）。

        夹爪角度在整个链上从 grip_from 线性变到 grip_to。
        stop_when 同 ramp：返回真就提前停，返回值是**实际停住的 q**（不是列底）。
        """
        n_seg = len(poses) - 1
        if n_seg <= 0:
            return None
        # 舵机限速：每段分到的时长都不许短于该段自身需要的最短时长。整条链拉长
        # n_seg 倍，保证**每一段**都够慢（不是只保平均）。
        need = 0.0
        for i in range(n_seg):
            gi = grip_from + (grip_to - grip_from) * i / n_seg
            gj = grip_from + (grip_to - grip_from) * (i + 1) / n_seg
            need = max(need, self.slow_enough(0.0, poses[i], poses[i + 1], gi, gj))
        total_duration = max(total_duration, need * n_seg)
        seg_dur = total_duration / n_seg
        steps = max(1, int(seg_dur / self.m.opt.timestep))
        q = poses[0]
        for i in range(n_seg):
            a, b = poses[i], poses[i + 1]
            for k in range(steps):
                f = (k + 1) / steps
                g = grip_from + (grip_to - grip_from) * (i * steps + k + 1) / (n_seg * steps)
                q = a + (b - a) * f
                self.drive(q, g)
                if on_frame is not None:
                    on_frame()
                if stop_when is not None and stop_when():
                    return q
        return q

    # ---- IK ----

    def _mink_setup(self):
        """惰性建 mink 的 Configuration。用 self.p.model（本就存在的规划器模型），
        mink 内部会自建 MjData，所以既不碰 self.d（仿真）也不碰 self.p.data。"""
        if self._mink_c is not None:
            return
        m = self.p.model
        for eq in range(m.neq):
            m.eq_active0[eq] = 0        # mink 的 MjData 从 model 继承 eq_active，weld 得关
        self._mink_c = mink.Configuration(m)

        # 冷启动姿态 = HOME_ANGLES_DEG（不是 qpos0：reset() 之后 go_home() 会把臂
        # 再推到 HOME_ANGLES_DEG，那才是旧 scipy 路径下 seed=None 的真实起点）。
        self._mink_home = np.array([
            deg_to_qpos(HOME_ANGLES_DEG[i + 1],
                        np.array([self.jnt_lo[i], self.jnt_lo[i] + self.jnt_span[i]]))
            for i in range(len(self.qidx))], dtype=float)

        # 参考朝向 R_ref：在名义取件点上**走一遍已经跑通的旧路**（scipy 那条），
        # 把 grasp_point 的姿态矩阵记下来，之后所有目标都用「这个朝向 + 新位置」。
        #
        # 为什么拿它当参考：那个姿态是录出过成功 episode 的姿态。用 mink 自己冷解
        # 出来的姿态当参考是循环论证（还没控朝向就拿来当朝向目标），而手填一个欧拉
        # 角则是凭空造魔数。标定一次、只此一处，是零魔数的做法。
        #
        # 注意它是**软目标**（MINK_ORIENT_COST=0.01），不是硬约束：mink 会尽量把
        # 姿态拉向它，但位置一冲突就放弃。所以这个标定值不精确也无所谓。
        self.p.reset()
        self._mink_R = np.eye(3)
        if self.p.move_to_grasp_point(paths.PICK_X, paths.PICK_Y, PICK_Z_BOT,
                                      wrist_deg=0.0):
            self._mink_R = np.array(self.p.data.site_xmat[self.gp_id],
                                    dtype=float).reshape(3, 3).copy()

        self._mink_task = mink.FrameTask("grasp_point", "site",
                                         position_cost=1.0,
                                         orientation_cost=MINK_ORIENT_COST)
        self._mink_limits = [mink.ConfigurationLimit(m)]
        # 刻意**不**冻 wrist_roll：它是管爪子滚转的那个关节，冻掉就等于放弃姿态。
        # 臂统共只有 5 个自由度（第 6 个是夹爪开合），3 位置 + 3 姿态本来就超定 1 维，
        # 再冻一个就只剩 4 个自由度去追 6 维任务，mink 会大面积让步。

    def ik_pose(self, x, y, z, seed=None, wrist_deg=0.0):
        """求把 grasp_point 站点放到 (x,y,z) 的关节角；**不收敛返回 None**。

        只碰 mink 自己的 MjData，绝不动 self.c 的仿真状态。

        seed 给一组关节角时热启动到那个姿态再解。SO-101 对同一个 (x,y,z) 有好几支
        解（肘上/肘下、腕翻转），分支锁不住的话 ramp_chain 沿关节空间插值会把手臂
        甩出去再甩回来（实测末端偏离 42mm、方块半路掉）。mink 每步解最小速度，冗余
        自由度天然贴近种子——实测竖直列相邻点跳变 4°，换 scipy 是 23°。

        **seed=None 会显式回 home**（见下面的注释）——别依赖 c.q 的残留状态。

        wrist_deg 现在**只当 wrist_roll 的初值**，不再是硬约束——姿态已经由 R_ref
        软引导（见 _mink_setup），钉单一关节反而把自由度卡死。所以 align_wrist()
        那套"转腕对齐全靠钉 wrist_roll"的做法在这条路上失效了（它现在恒不调用，
        ALIGN_WRIST=False）；将来要腕转对齐，得改成把 R_ref 绕接近轴旋转，而不是
        钉关节。
        """
        self._mink_setup()
        c = self._mink_c
        q = c.q.copy()
        if seed is not None:
            q[self.qidx] = np.asarray(seed, dtype=float)[:len(self.qidx)]
        else:
            # 冷启动**必须**显式回 home。c.q 是 mink 配置自己的**残留**状态：上一条
            # 竖直列解完就留在那儿，下一条列的第一个点会从上一列的末尾姿态开始解。
            # 旧 scipy 版在 seed=None 时走 p.reset() → go_home()，这里得对齐——这是
            # 换 mink 时漏掉的一步：plan_column 里"首个点从 home 冷启动"那句注释，
            # 在此之前一直是假的。
            q[self.qidx] = self._mink_home
        q[self.qidx[4]] = deg_to_qpos(float(wrist_deg),
                                      np.array([self.jnt_lo[4], self.jnt_lo[4] + self.jnt_span[4]]))
        c.update(q)

        self._mink_task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(self._mink_R),
            np.array([float(x), float(y), float(z)])))
        err = self._mink_task.compute_error(c)
        for _ in range(MINK_MAX_ITERS):
            if float(np.linalg.norm(err[:3])) < MINK_TOL:
                break
            try:
                vel = mink.solve_ik(c, [self._mink_task], MINK_DT, "daqp",
                                    limits=self._mink_limits)
            except mink.NoSolutionFound:
                return None
            except mink.NotWithinConfigurationLimits:
                return None
            c.integrate_inplace(vel, MINK_DT)
            err = self._mink_task.compute_error(c)

        # 迭代用完还没进 1mm 就当没解出来 —— 这正是旧实现缺的那一步：它只查 isfinite，
        # 差 2mm 的解照样返回，下游拿它当准的用。
        if float(np.linalg.norm(err[:3])) > MINK_TOL_HARD:
            return None
        return np.array([c.q[i] for i in self.qidx], dtype=float)

    # ---- 观测 ----

    def state_action(self):
        return self.c.get_state_action()

    def image(self, camera):
        return self.c.render_observation(camera=camera, height=paths.IMG_H, width=paths.IMG_W)

    def cube_pos(self):
        return self.d.xpos[self.cube_b].copy()

    def cube_yaw(self):
        """方块绕 z 的偏航角（弧度）。"""
        a = self.cube_qadr
        return float(2.0 * np.arctan2(self.d.qpos[a + 6], self.d.qpos[a + 3]))

    def jaw_axis_angle(self):
        """爪开合轴在水平面内的方位角（弧度），从**规划器**的当前姿态上量。

        ik_pose() 算完会把解写进 self.p 的 qpos，所以紧跟在它后面调就是那个姿态下的
        爪方位。别用量 self.d：那是仿真里手臂当前的姿态，不是待求路点的姿态。
        """
        v = self.p.data.xpos[self.jaw_b] - self.p.data.xpos[self.grip_b]
        return float(np.arctan2(v[1], v[0]))

    def n_cube_contacts(self):
        # 与 sim_env.is_success 共用同一份接触判定，避免采集闸门和评估判据分叉
        return sim_env.n_gripper_cube_contacts(self.m, self.d, self.cube_b, self.gripper_bodies)

    def tip_z(self):
        """爪上所有 mesh 顶点的**最低世界 z**（精确值，扫全量顶点）。抓取面 z=0，
        所以这就是"爪离工作面还有多远"，负数=压进去了。

        用 mesh 真实顶点，不用 geom 包围球——包围球在这上面粗到没用（爪是长条，
        球半径由最远顶点定，算出来能差几十毫米）。
        **只当基准用**：solve_pick_z 走的是 tip_z_geo()，见那里的说明。
        """
        lo = 1e9
        for g, mid in self._tip_geoms:
            p = self.d.geom_xpos[g]
            if mid < 0:                       # primitive geom：中心减半径
                z = p[2] - self.m.geom_size[g][0]
            else:
                # 只要 z：V@R.T 的第三列就是 V@R[2]，取第三行做矩阵-向量乘就够，
                # 不用建 Nx3 的中间数组。
                a, n = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
                R2 = self.d.geom_xmat[g].reshape(3, 3)[2]
                z = float((self.m.mesh_vert[a:a + n] @ R2).min()) + p[2]
            if z < lo:
                lo = z
        return lo

    def measure_jaw_tips(self):
        """量一次两个爪尖，之后算离地高度只靠刚体几何。

        几何模型：已知**爪长**和**爪与地面的角度**就能算出爪尖离地多少。
        对每个爪 geom 取"离自己体原点最远"的那个顶点当爪尖，存下它在**体坐标系**
        里的坐标 p。之后：

            爪尖世界 z = 体原点 z + (体姿态矩阵 @ p) 的 z 分量

        体姿态矩阵里含"与地面的角度"，p 里含"爪长"——正好是那两个量。每次一次
        3 维点积（µs 级），不再把 76k 个顶点全转到世界系取 min（0.37ms）。

        p 与姿态无关——它在**体坐标系**里，体自己带着爪一起转；闭爪时变的是体姿态
        矩阵（xmat），不是 p。所以只量一次就够，闭爪那一下的下扫天然被 xmat 捕捉到。

        代价：p 取的是"离体原点最远的顶点"，不是"永远最低的那个顶点"。爪板是带
        弧度的长条，某些朝向下最低点会落在别处，误差上限是爪板厚度（几毫米）。
        TIP_CLEARANCE 要留得比这个误差大。solve_pick_z 会同时打 tip_z() 作对照。
        """
        tips = []
        for g, mid in self._tip_geoms:
            b = int(self.m.geom_bodyid[g])
            if mid < 0:                       # primitive geom：中心减半径当爪尖
                p = self.m.geom_pos[g] - np.array([0.0, 0.0, self.m.geom_size[g][0]])
            else:
                a, n = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, self.m.geom_quat[g])
                # mesh 顶点在 geom 坐标系里；转到体坐标系：geom_pos + R_geom @ v
                v = self.m.mesh_vert[a:a + n] @ R.reshape(3, 3).T + self.m.geom_pos[g]
                p = v[int(np.argmax(np.linalg.norm(v, axis=1)))].copy()
            tips.append((b, np.asarray(p, dtype=float)))
        self._tips = tips

    def tip_z_geo(self):
        """按爪尖 + 体姿态矩阵算**最低爪尖的世界 z**（闭式，不扫 mesh）。

        每个爪尖各自投影一次，取最低的那个。`xmat[2]` 是体姿态矩阵的第三行，
        点乘本地爪尖坐标就是它在世界 z 上的分量——所以"爪与地面的角度"和"爪长"
        都在这一个点积里。
        """
        return min(float(self.d.xpos[b][2] + self.d.xmat[b].reshape(3, 3)[2] @ p)
                   for b, p in self._tips)

    # ---- 一次完整取放 ----

    def grasp_offset(self):
        """方块中心相对 grasp_point 站点的世界偏移，**三维**。

        夹爪不是关于 grasp_point 对称的，夹住的方块会挂在中心线外 ~12mm；
        放置时用它反推瞄准点，方块才能落在柱子正中（专家用真值自标定是合法的，
        VLA 学的只是这段轨迹）。

        z 分量（负值 = 方块在夹持点下方）原来是被一个写死的假设替掉的
        （RELEASE_LIFT，假定挂低 50mm）。那个假设在方块滑落之后就不成立了——
        实测挂低 55mm，差这 5mm 会让下探多压 5mm，把方块压歪、松手时推走。
        所以 z 也得实测。
        """
        return (self.cube_pos() - self.d.site_xpos[self.gp_id]).copy()

    def plan_pick(self, pick_xy=None, z_bot=None, wrist_deg=0.0):
        """算取件侧路点的关节角。返回 None 表示 IK 有解不出的点。

        z_bot 是下探列底端（闭合时夹持点对准的高度）。**别按方块中心高度
        （CUBE_Z=0.02）去对**：指尖比夹持点低 FINGER_DROP，那样指尖会落到桌面
        以下、伺服失速、爪子蹭着方块把它推走。默认 PICK_Z_BOT 让指尖停在台面
        上方 PICK_CLEARANCE —— 64mm 的爪板仍完整覆盖 40mm 方块的 37mm 高，够夹。
        """
        px, py = (paths.PICK_X, paths.PICK_Y) if pick_xy is None else pick_xy
        zb = PICK_Z_BOT if z_bot is None else float(z_bot)
        # 列顶要高出方块顶面（列顶 - FINGER_DROP > 40mm），否则张着爪横过去时指尖
        # 已经低于方块顶棱，横扫就会把它推走。
        z_top = zb + 0.08

        self.p.reset()
        q_home = np.array([self.p.data.qpos[i] for i in self.qidx], dtype=float)

        # 先算列，再让 above_pick/at_pick 直接**取自列的两端**：这样三者天然同一
        # 分支，也省掉两次重复 IK。（以前是各自独立求解的。）
        col = self.ik_column(px, py, z_top, zb, wrist_deg=wrist_deg)
        if col is None:
            return None
        q = {"home": q_home, "above_pick": col[0], "at_pick": col[-1], "column": col,
             "top_xyz": (px, py, z_top)}
        if any(v is None for v in q.values()):
            return None
        return q

    def closing_tip_z(self, q_arm=None):
        """**在这一刻闭爪**的话，爪尖会到多高？负数 = 会压进工作面。

        纯运动学预判：把手臂摆到 q_arm（不给就用当前实际姿态）、夹爪角设成闭合，
        mj_forward 后读爪尖，然后把姿态和爪角原样还回去。不推进物理、不动 ctrl，
        所以调用它不会扰动正在跑的 episode。

        为什么要在**闭合前**问这一句：压台面的不是下探（那时爪还张着、爪尖高），
        是**闭爪那一下**——动爪绕铰链转，最低顶点向下扫十几到几十毫米。所以必须
        拿"闭合后"的姿态去量，拿张开的状态量是量不出来的。
        """
        keep = self.d.qpos[self.qidx].copy()
        if q_arm is not None:
            self.d.qpos[self.qidx] = np.asarray(q_arm, dtype=float)
        self.d.qpos[self.qidx[5]] = self.grip_q(GRIP_CLOSE_DEG)
        mujoco.mj_forward(self.m, self.d)
        z = self.tip_z_geo()
        self.d.qpos[self.qidx] = keep
        mujoco.mj_forward(self.m, self.d)
        return z

    def closed_tip_at(self, pick_xy, zb, z_top=None, wrist_deg=0.0, n=7):
        """在 (pick_xy, zb) 这个姿态下，**闭合**爪的最低顶点 z。量不到返回 None。

        纯运动学：把主仿真的 qpos 摆到那个姿态、夹爪角设成闭合，mj_forward 一次
        就读 tip_z。不推进物理、不碰 ctrl，所以不会扰动 episode 的状态机。
        **但它确实改写了 d.qpos**，调用方必须事后 reset_scene 把它收回去。
        """
        z_top = zb + 0.08 if z_top is None else z_top
        col = self.ik_column(pick_xy[0], pick_xy[1], z_top, zb, n=n, wrist_deg=wrist_deg)
        if col is None:
            return None
        self.d.qpos[self.qidx] = col[-1]
        self.d.qpos[self.qidx[5]] = self.grip_q(GRIP_CLOSE_DEG)
        mujoco.mj_forward(self.m, self.d)
        return self.tip_z_geo()

    def solve_pick_z(self, pick_xy, wrist_deg=0.0, clearance=TIP_CLEARANCE,
                     iters=5, z0=None):
        """按**当前姿态**反推下探列底端 z_bot：让闭合爪尖刚好落在 clearance 上。

        为什么必须按姿态算：爪尖离 grasp_point 的实际落差随手臂伸出长度变——同一个
        z_bot，方块在 (250,-50) 时闭合爪尖 +13.1mm，挪到 (280,-20) 就压穿台面
        （实测两条 episode 一头一尾撞的正是这个）。固定常数没有解，要么夹不到、
        要么压台面。这里解出"闭合爪尖 = clearance"的那个 z_bot，也就是**夹得最深、
        又不压台面**的那个高度。

        解出来通常比 PICK_Z_BOT(100mm) **更深**：抬到 100mm 是为了绕开压台面，
        而按姿态解能贴着 clearance 走，夹持面积反而回来了。

        爪尖对 z_bot 近似 1:1（两者差的是姿态，姿态随 z 变得慢），所以直接拿
        误差回代，5 次足够收敛。返回 None = IK 无解。
        """
        zb = float(PICK_Z_BOT if z0 is None else z0)
        for _ in range(iters):
            t = self.closed_tip_at(pick_xy, zb, wrist_deg=wrist_deg)
            if t is None:
                return None
            err = t - clearance
            if abs(err) < 5e-4:
                break
            zb = float(np.clip(zb - err, 0.02, 0.20))
        # 收敛判据在循环里，最后一次量的是更新前的 zb。返回前按最终的 zb 再量一次，
        # 这样打出来的两个数和返回的 z_bot 是同一个姿态的（顺带也是 mesh 对照）。
        t = self.closed_tip_at(pick_xy, zb, wrist_deg=wrist_deg)
        if t is None:
            return None
        print(f"  [expert] 夹持高度 z_bot={zb*1000:.1f}mm  闭合爪尖 "
              f"几何 {t*1000:+.1f}mm / mesh 精确 {self.tip_z()*1000:+.1f}mm")
        return zb

    def ik_column(self, x, y, z_top, z_bot, n=7, wrist_deg=0.0):
        """竖直列：在 (x,y) 上取 n 个高度的 IK 解，从 z_top 排到 z_bot。

        只给首尾两点让关节空间插值的话，末端实际走的是弧线（下探/抬离时会
        蹭到刚放下的方块）；中间插几个路点才逼成直线。

        逐点用**上一个解**热启动（首个点从 home 冷启动）——这不只是求快点，
        是保证整列落在同一支 IK 分支上。各自冷启动的话相邻点会跨分支跳几十度，
        ramp_chain 沿关节空间插值就会把手臂甩飞（见 ik_pose 的说明）。
        """
        poses = []
        seed = None
        for i in range(n):
            z = z_top + (z_bot - z_top) * i / (n - 1)
            q = self.ik_pose(x, y, z, seed=seed, wrist_deg=wrist_deg)
            if q is None:
                return None
            poses.append(q)
            seed = q
        return poses

    def ik_line(self, p_from, p_to, wrist_deg=0.0, seed=None, step=CART_STEP):
        """笛卡尔**直线**：从 p_from 到 p_to 之间取一串 IK 解。

        和 ik_column 是同一件事的两种形状——那根是竖直列，这根是任意方向。路点数
        由「长度 ÷ 步长」定（MimicGen 的 step_size 做法，见 CART_STEP）。

        为什么必须密采样而不能只给两端：ramp / ramp_chain 是**关节空间**线性插值，
        两个相隔远的位姿在关节空间走直线、在笛卡尔空间就是一条大弧。原来 ⑧ 那段
        横移就是拿 ramp() 直接连首尾，实测方块在 above_place 处 y 偏 38.8mm。
        MimicGen 的 WaypointSequence 从不这么干：它插值**位姿**、由控制器去追位姿。
        这里等价的做法就是「先按笛卡尔直线密采样、再逐点 mink 解 IK」——路径真的
        是直的，且点密到相邻点之间的关节插值不再产生可见弧线。

        注意旋转部分**不用插值**：mink 的朝向目标是常数 R_ref（软约束），不像
        MimicGen 要 slerp 位姿旋转。所以这里的 "pose 序列" 退化成纯位置序列。

        逐点用上一个解热启动，理由同 ik_column（锁同一支 IK 分支）。
        """
        p_from = np.asarray(p_from, dtype=float)
        p_to = np.asarray(p_to, dtype=float)
        n = max(2, int(np.ceil(float(np.linalg.norm(p_to - p_from)) / step)) + 1)
        poses = []
        for i in range(n):
            p = p_from + (p_to - p_from) * (i / (n - 1))
            q = self.ik_pose(p[0], p[1], p[2], seed=seed, wrist_deg=wrist_deg)
            if q is None:
                return None
            poses.append(q)
            seed = q
        return poses

    def align_wrist(self, xy, z, target_yaw, a0=0.0, da=25.0):
        """求让爪开合轴平行于方块棱的 wrist_roll **舵机角**。返回 (舵机角, 残余°)。

        不解析补偿「转多少补多少」：wrist_roll 绕腕部自己的轴转，那根轴和竖直有
        夹角，投到水平面上不是 1:1。所以解两次量一次灵敏度再外推。

        方块四个面等价（爪轴平行于 yaw+k·90° 都是合法的面抓），目标取离当前爪方位
        最近的 90° 倍数线，转幅不超过 45°；再把 tgt±90° 几条等价线都试一遍取行程内
        离原姿势最近的那个——舵机有上下限，直接补过去会被裁到位（实测裁到 1°、
        残余 7.8°）。

        a0=0.0 是 `_ik_angles` 原本钉死的那个舵机角，从它起步意味着"能不对齐就不转"。
        """
        def jaw_at(a):
            if self.ik_pose(xy[0], xy[1], z, wrist_deg=a) is None:
                return None
            return self.jaw_axis_angle()

        th0, th1 = jaw_at(a0), jaw_at(a0 + da)
        if th0 is None or th1 is None:
            return a0, 90.0
        s = _wrap(th1 - th0) / da                       # 弧度/度
        if abs(s) < 1e-3:
            return a0, 90.0                             # 近似奇异，转不动
        k = np.pi / 2
        tgt = target_yaw + round(_wrap(th0 - target_yaw) / k) * k
        cands = [a0 + _wrap(tgt + m * k - th0) / s for m in (-2, -1, 0, 1, 2)]
        cands = [a for a in cands if 1.0 <= a <= 179.0]
        if not cands:
            return a0, 90.0
        a = float(min(cands, key=lambda v: abs(v - a0)))
        th2 = jaw_at(a)
        if th2 is None:
            return a0, 90.0
        d = _wrap(th2 - tgt)                            # 残余按 mod 90° 算（偏 90° 也是面抓）
        return a, float(np.degrees(abs(d - k * round(d / k))))

    def plan_place(self, aim_xy, z_bot=None, wrist_deg=0.0):
        """放置侧路点：竖直列（含下探与抬离）。

        z_bot 是列底端（= 松手时方块中心的目标高度）。默认降到台面高度，
        但 40mm 方块全开仍被夹爪楔住，降到台面会让撤退把方块拖走；
        抬高一点让它自己落下反而干净。

        wrist_deg 跟取件侧保持一致：搬运途中腕角一变，夹着的方块就在爪里被拧一下
        （带着件转腕），能避免就避免。
        """
        zb = paths.CUBE_Z if z_bot is None else float(z_bot)
        # 列顶和取件侧用**同一条规则**（zb + 0.08，见 plan_pick）。
        #
        # 原来这里写死 max(0.32, zb + 0.10) —— 0.32 是窄柱场景的遗留，注释说它是
        # "home 高度附近的悬停点"。后果：取件列顶在 165mm、放置列顶在 320mm，而两列
        # 之间是 ramp() 关节空间插值（见 ramp 的说明），于是这段转移在笛卡尔空间被拉成
        # 一条大弧——实测方块在 above_place 处窜到 302.7mm、y 偏出 38.8mm。
        # 换句话说：那个弧一半是关节插值造的，另一半是这 320mm 硬把它垫高的。
        #
        # 两侧用同一条规则后，两列顶高度接近（约 134 vs 150mm），转移段几乎水平。
        z_top = zb + 0.08
        col = self.ik_column(aim_xy[0], aim_xy[1], z_top, zb, wrist_deg=wrist_deg)
        return None if col is None else {"column": col,
                                         "top_xyz": (aim_xy[0], aim_xy[1], z_top)}

    def reset_scene(self, cube_xy=None, cube_yaw=0.0):
        """回 home、张爪、把方块放回取件柱顶（可带抖动）。"""
        self.grip_hold = None
        self.c.reset()
        self._disable_weld()
        if cube_xy is not None:
            a = self.cube_qadr
            self.d.qpos[a:a + 3] = (cube_xy[0], cube_xy[1], paths.CUBE_Z)
            h = cube_yaw / 2.0
            self.d.qpos[a + 3:a + 7] = (np.cos(h), 0.0, 0.0, np.sin(h))
            mujoco.mj_forward(self.m, self.d)

    def run_episode(self, cube_xy=None, cube_yaw=0.0, on_frame=None, trace=None,
                    release_lift=RELEASE_LIFT):
        """跑一次取放。on_frame 每物理步调一次（用于采样记录）。

        trace 传入一个 list 的话，每个阶段结束会 append 一行诊断（方块位置 + 接触数）。
        路点在 reset 之后才算——方块抖动过的话，瞄准点必须是它真实的位置。
        返回 trace；IK 解不出路点时返回 None。
        """
        cb = on_frame

        def mark(stage):
            if trace is None:
                return
            p = self.cube_pos()
            grip_deg = float(np.degrees(self.d.qpos[self.qidx[5]]))
            trace.append((stage, float(p[0]), float(p[1]), float(p[2]),
                          self.n_cube_contacts(), grip_deg))

        # ① 先按方块位置解出这条 episode 的夹持高度，再 reset 摆场景。
        #    顺序不能反：solve_pick_z 靠改写 d.qpos 做运动学测量，reset_scene 的
        #    c.reset() 会把姿态收回去（先 reset 再解就被清掉了）。
        #    ALIGN_WRIST 开着的话这里少了 wdeg——那开关是关的，真要开得连带重排。
        pick_xy = (paths.PICK_X, paths.PICK_Y) if cube_xy is None else tuple(cube_xy)
        wdeg = 0.0
        zb = self.solve_pick_z(pick_xy, wrist_deg=wdeg)
        if zb is None:
            print(f"  [expert] 夹持高度解不出：方块 {np.round(pick_xy, 3)}")
            return None
        self.pick_z_bot = zb
        # ② 回 home，张爪，摆好方块
        self.reset_scene(cube_xy, cube_yaw)
        q = self.plan_pick(self.cube_pos()[:2], z_bot=zb, wrist_deg=wdeg)
        if q is None:
            print(f"  [expert] 取件列无解：方块 {self.cube_pos()[:2].round(3)}")
            return None
        self.ramp(q["home"], q["home"], GRIP_OPEN_DEG, GRIP_OPEN_DEG, 0.3, cb)
        mark("home")
        # ③ 升到方块上方
        self.ramp(q["home"], q["above_pick"], GRIP_OPEN_DEG, GRIP_OPEN_DEG, DUR_APPROACH, cb)
        mark("above_pick")
        # ④ 下探到夹持位（竖直列；实际会停住，夹爪下端/方块顶面决定停点）
        self.ramp_chain(q["column"], GRIP_OPEN_DEG, GRIP_OPEN_DEG, DUR_DESCEND, cb)
        mark("at_pick")

        # ④′ 闭爪前预判：这一下闭上去会不会压台面？
        #
        #     会压台面的不是下探 —— 下探时爪还张着，爪尖是抬起来的。是**闭合那一下**：
        #     动爪绕铰链往里转，最低的顶点向下扫十几到几十毫米（扫多少取决于手臂伸
        #     多长），这一扫才扎进桌面、把方块弹飞。所以必须拿「闭合后」的姿态去量，
        #     拿张爪状态量是量不出来的。在哪量、怎么量见 closing_tip_z()。
        #
        #     判完分三种走法：够高 → 就在原位夹；不够 → 沿这条竖直列往上退，退到
        #     第一个闭合安全的高度再夹（夹浅一点，但好过把方块顶飞）；整列都不够 →
        #     干脆不夹，空手回去。宁可这条 episode 不出货，也别出一条"把方块弹飞
        #     到 455mm"的脏数据 —— 那种帧进数据集是有毒的。
        k_safe = None
        for k in range(len(q["column"]) - 1, -1, -1):
            if self.closing_tip_z(q["column"][k]) >= TIP_CLEARANCE:
                k_safe = k          # 从下往上找，第一个安全的 = 夹得最深的那个安全位
                break
        if k_safe is None:
            print(f"  [expert] 整条取件列上闭爪都会压台面（爪尖 < {TIP_CLEARANCE*1000:.0f}mm），"
                  f"这次不夹")
            return None
        at = q["column"][k_safe]
        self.ramp(q["at_pick"], at, GRIP_OPEN_DEG, GRIP_OPEN_DEG, DUR_LIFT, cb)
        if k_safe != len(q["column"]) - 1:
            print(f"  [expert] 闭爪会压台面：从夹持位沿同一竖直列后退 "
                  f"{len(q['column']) - 1 - k_safe} 段（{self.closing_tip_z(at)*1000:.1f}mm）再夹")
            mark("backed_up")
        else:
            print(f"  [expert] 闭爪预判通过：爪尖 {self.closing_tip_z(at)*1000:.1f}mm"
                  f" ≥ {TIP_CLEARANCE*1000:.0f}mm，原位夹取")

        # ⑤ 闭合
        self.ramp(at, at, GRIP_OPEN_DEG, GRIP_CLOSE_DEG, DUR_CLOSE, cb)
        self.hold(at, GRIP_CLOSE_DEG, DUR_SETTLE, cb)
        mark("closed")
        # 保持 None = 全程全闭合指令（位置伺服顶着 3.35 N·m 死命夹）。
        # 原实现 grip_hold = 当前角 - 5°：只留 5° 预紧，位置保持几乎停在平衡点，
        # 方块在抬起时稍微一滑爪就卸力，脱手（实测成功率 2/5）。全闭合指令让夹持力
        # 在爪开到指令位之前一直饱和，抓得稳（4/5）。剩下 1/5 是固定爪手指比方块
        # 长 64mm 的几何问题（见 sim_scene.xml 注释），不是夹持参数能调的。
        self.grip_hold = None
        # ⑥ 沿竖直列抬起（慢，避免甩脱）
        #    起点必须是**实际闭合的那一段**：④′ 后退过的话，整列 reversed 会先从列底
        #    爬起，等于把刚退出来的高度又压回去、方块再吃一次台面（踩过）。
        self.ramp_chain(list(reversed(q["column"][:k_safe + 1])),
                        GRIP_CLOSE_DEG, GRIP_CLOSE_DEG, DUR_LIFT, cb)
        mark("lifted")

        # ⑦ 按实测偏移修正瞄准点，让**方块**而不是夹持点落在柱子正中
        #
        #    z 同样按实测反推：方块中心比夹持点低 |off[2]|，要让它正好停在静止高度
        #    CUBE_Z，夹持点就该停在 CUBE_Z - off[2]（off[2] 是负的，所以是加）。
        #    原来这里写死 CUBE_Z + 0.05，等价于假定挂低 50mm——实测 55mm，差 5mm。
        off = self.grasp_offset()
        aim = (paths.PLACE_X - off[0], paths.PLACE_Y - off[1])
        z_place = paths.CUBE_Z - off[2] + release_lift
        qp = self.plan_place(aim, z_bot=z_place, wrist_deg=wdeg)
        if qp is None and wdeg:
            # 对齐角在放置侧无解（够不着 / 超行程）就退回 0°：带着件转腕有掉的风险，
            # 但比整条 episode 丢掉强，而且成功率闸门会把掉了的筛掉。
            qp = self.plan_place(aim, z_bot=z_place)
        if qp is None:
            print(f"  [expert] 放置列无解：瞄准 {aim[0]:.3f},{aim[1]:.3f}"
                  f"（爪偏移 {off[0]*1000:+.1f},{off[1]*1000:+.1f}mm）")
            return None
        col = qp["column"]  # 从 z=+0.20 竖直降到 z=cz

        # ⑧ 横移到柱子正上方
        #
        #    **必须走笛卡尔直线**，不能像以前那样 ramp(above_pick, col[0]) 直接连首尾——
        #    那是关节空间线性插值，末端在笛卡尔空间走的是一条大弧：实测方块在
        #    above_place 处 y 偏出 38.8mm、z 被垫到 302.7mm（z 那半是 plan_place 里
        #    写死的 0.32 干的，已改成 zb+0.08）。改成"密采样 + 逐点 IK"之后实测 z 落到
        #    95.3mm、x 误差 35.7mm → 0.3mm。
        #
        #    末点用 col[0]（放置列已经解出来的列顶）而不是重解一次：两者是同一个
        #    (x,y,z) 上的 IK，热启动链也一致，所以数值上等价，但这样能保证 ⑧ 的终点
        #    和 ⑨ 的起点**是同一个姿态**，不会在接缝处跳一下。
        line = self.ik_line(q["top_xyz"], qp["top_xyz"], wrist_deg=wdeg,
                            seed=q["above_pick"])
        if line is None:
            print(f"  [expert] 转移直线无解：{np.round(q['top_xyz'], 3)} → "
                  f"{np.round(qp['top_xyz'], 3)}")
            return None
        line[-1] = col[0]
        self.ramp_chain(line, GRIP_CLOSE_DEG, GRIP_CLOSE_DEG, DUR_TRANSFER, cb)
        mark("above_place")

        # ⑧′ 到站后按**方块实测位置**再修一次，然后才下探
        #
        #     ⑦ 的 aim 是拿 grasp_offset() 算的，但那是**抬升刚结束**那一刻的偏移。
        #     方块在抬升里滑了约 25mm、掉到爪尖吊着，之后像钟摆一样随姿态转——横移到位
        #     时 y 向偏移已经从 +4.4mm 变成 +39.8mm，⑦ 的补偿过期了 35mm。
        #
        #     为什么修在这一刻：实测下探段(⑨)里偏移几乎不变（+30.3 → +27.9mm，只动
        #     2mm），而横移段(⑧)里它一直在变。所以下探**之前**修最有效，修完能一直
        #     保持到落点；下探中途再修反而会把方块在台面附近拖一下。
        #
        #     用方块的**真值**位置：scripted expert 允许开特权（MimicGen 的 DatagenInfo
        #     也是直接拿物体位姿的）。VLA 要学的是"看到什么就做什么"，老师能给出干净
        #     轨迹就够了，不需要 VLA 自己会算这个。
        cur = self.cube_pos()
        dxy = np.array([paths.PLACE_X - cur[0], paths.PLACE_Y - cur[1]])
        if float(np.linalg.norm(dxy)) > 5e-4:
            top = np.asarray(qp["top_xyz"], dtype=float)
            fix_top = (top[0] + dxy[0], top[1] + dxy[1], top[2])
            fix = self.ik_line(top, fix_top, wrist_deg=wdeg, seed=col[0])
            # 列必须跟着搬到新 (x,y)，否则 ⑨ 会先横着把方块拖回旧瞄准点。
            # z 用**这一刻**实测的挂低量——比 ⑦ 那一刻的更接近下探时的真实值。
            qp2 = self.plan_place(fix_top[:2],
                                  z_bot=paths.CUBE_Z - self.grasp_offset()[2] + release_lift,
                                  wrist_deg=wdeg)
            if fix is not None and qp2 is not None:
                self.ramp_chain(fix, GRIP_CLOSE_DEG, GRIP_CLOSE_DEG, 1.2, cb)
                col = qp2["column"]
                print(f"  [expert] 到站修正 {np.linalg.norm(dxy)*1000:.1f}mm"
                      f"（方块 {cur[0]*1000:.1f},{cur[1]*1000:.1f} → 目标 "
                      f"{paths.PLACE_X*1000:.0f},{paths.PLACE_Y*1000:.0f}）")
            else:
                print(f"  [expert] 到站修正 {np.linalg.norm(dxy)*1000:.1f}mm 无解，按原瞄准点下探")
        # ⑨ 沿竖直列下探（每段单独 IK，避免关节空间插值走弧线）
        self.ramp_chain(col, GRIP_CLOSE_DEG, GRIP_CLOSE_DEG, DUR_PLACE_DESCEND, cb)
        mark("at_place")
        # ⑩ 张开释放（解除保持，才真的张得开）
        self.grip_hold = None
        self.ramp(col[-1], col[-1], GRIP_CLOSE_DEG, GRIP_OPEN_DEG, DUR_RELEASE, cb)
        mark("released")
        # ⑪ 沿同一竖直列抬离，停一下让方块落定再走
        self.ramp_chain(list(reversed(col)), GRIP_OPEN_DEG, GRIP_OPEN_DEG, DUR_RETREAT, cb)
        self.ramp(col[0], col[0], GRIP_OPEN_DEG, GRIP_OPEN_DEG, 1.0, cb)
        mark("settled")
        return trace

    def success(self, place_xy=None):
        return sim_env.cube_placed(self.m, self.d, self.cube_b,
                                   self.gripper_bodies, place_xy)
