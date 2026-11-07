"""SO-101 抓取仿真环境：给 SmolVLA 用的观测/动作接口。

刻意做成和真机 SO-101 一致的 LeRobot 口径：
- state/action 都是 **舵机角度 0~180°**（不是弧度、不是关节原始角），顺序
  shoulder_pan → gripper，跟 mcp_robot 的 SERVO_TO_JOINT 完全一致。
  这样在仿真里采的数据集，feature 定义和真机数据集是同一套。
- 图像键名 observation.images.camera1/2，对应 SmolVLA 期望的输入；
  camera3 由 empty_cameras=1 补常数图（我们只有两路真相机）。

动作语义：一步 = 把 6 个关节的目标角写进 ctrl，然后推进 1/fps 秒物理。
"""
from __future__ import annotations

import sys

import mujoco
import numpy as np

import paths

sys.path.insert(0, str(paths.MCP_SOURCES))
from mujoco_servo_controller import (  # noqa: E402
    MujocoServoController, deg_to_qpos, qpos_to_deg)

# 物理步长固定 0.002s（MuJoCo 默认），30fps 一帧 = 1/(30*0.002) ≈ 17 步
STEPS_PER_FRAME = max(1, round(1.0 / (paths.FPS * 0.002)))

# 观测里的图像键 → 场景里的相机名
CAMERAS = {paths.CAM_FRONT: "front_cam", paths.CAM_WRIST: "wrist_cam"}


def n_gripper_cube_contacts(m, d, cube_body, gripper_bodies) -> int:
    """方块与夹爪之间的接触点个数；0 表示已脱手。"""
    mujoco.mj_collision(m, d)
    n = 0
    for i in range(d.ncon):
        b1 = m.geom_bodyid[d.contact[i].geom1]
        b2 = m.geom_bodyid[d.contact[i].geom2]
        if cube_body in (b1, b2) and (b1 in gripper_bodies or b2 in gripper_bodies):
            n += 1
    return n


def apply_grip_friction(m):
    """给夹爪两片 mesh 加高摩擦（等效软胶垫），只改材质、不动结构。

    夹爪 mesh 默认摩擦 1.0，抓 40mm 方块抓不牢、一甩就滑。实测遥操作口径下
    （全闭合指令、不锁 grip_hold）：爪面摩擦 1.0→4.0 把甩动滑动从 190mm 压到
    44mm、接触从 0 变 6。**别动 forcerange** —— 加大夹持力反而顶穿方块、更糟
    （实测 fric+force 把直抬从 29mm 砸到 4.9mm）。

    这是材质属性，不是几何，所以 SO-ARM100/ 上游 XML 一个字不用改。
    """
    for g in range(m.ngeom):
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        if body in ("gripper", "moving_jaw_so101_v1"):
            m.geom_friction[g][0] = 4.0


def cube_placed(m, d, cube_body, gripper_bodies, place_xy=None) -> bool:
    """放置成功判定：方块落在放置台中心 PLACE_RADIUS 内、高度对、且已脱离夹爪。

    采集时给专家当闸门、评估时给 VLA 打分，用的是同一个函数——两边判据
    一旦分叉，采出来的"成功"数据就不一定是评估认可的成功。
    """
    tx, ty = place_xy or (paths.PLACE_X, paths.PLACE_Y)
    p = d.xpos[cube_body]
    return bool(np.hypot(p[0] - tx, p[1] - ty) < paths.PLACE_RADIUS
                and abs(p[2] - paths.CUBE_Z) < paths.PLACE_Z_TOL
                and n_gripper_cube_contacts(m, d, cube_body, gripper_bodies) == 0)


class SO101SimEnv:
    """窄柱取件 + 宽台面放置的单任务环境（里程碑 1）。"""

    def __init__(self, scene_xml=None, render=True):
        self.c = MujocoServoController(str(scene_xml or paths.SCENE_XML), headless=True)
        self.m, self.d = self.c.model, self.c.data
        # 离屏帧缓冲默认 480²，渲染 512² 会报 "Image height 512 > framebuffer height 480"
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, paths.IMG_W)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, paths.IMG_H)
        self.render = render

        # 真实摩擦抓取：weld 恒关，且 reset() 也不会重新打开
        for eq in range(self.m.neq):
            self.d.eq_active[eq] = 0
            self.m.eq_active0[eq] = 0

        def jid(n):
            return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)

        self.acts = [next(a for a in range(self.m.nu) if self.m.actuator_trnid[a, 0] == jid(n))
                     for n in paths.JOINT_NAMES]
        self.qidx = [self.m.jnt_qposadr[jid(n)] for n in paths.JOINT_NAMES]
        self.ranges = [self.m.jnt_range[jid(n)] for n in paths.JOINT_NAMES]
        self.grip_range = self.m.jnt_range[jid("gripper")]
        self.cube_b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_qadr = self.m.jnt_qposadr[jid("cube")]
        self.gripper_bodies = {
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("gripper", "moving_jaw_so101_v1")}

        self._fix_wrist_cam()
        apply_grip_friction(self.m)

    # ---- 状态 ----

    # 腕部相机的挂载（gripper body 坐标系：-z = 接近方向，±x = 两爪开合方向）。
    #
    # 上游 so101_new_calib.xml 里是 pos="0.05 0 0"、fovy=70、朝接近方向看，实测画面
    # 被夹爪侧壁糊满 —— 既看不见方块也看不见桌面，当观测和当预览都没用。而且 0.05
    # 正好落在两爪开合方向（+x）上，看过去夹爪本体把方块挡得只剩一条边。
    # 换成从 -y 侧退后 7cm、广角俯视「夹持点与方块之间」：两爪和方块同时在画面里。
    #
    # 改在这里而不是改 XML：SO-ARM100/ 是上游 vendored 模型，不该动。
    # 位置/朝向是渲染对比挑出来的，见提交里的 _solve_wristcam.py 记录。
    WRIST_CAM_EYE = (0.0, -0.07, 0.02)
    WRIST_CAM_AIM = (0.0, 0.0, -0.065)
    WRIST_CAM_FOVY = 85.0

    @staticmethod
    def _look_at_quat(eye, target, up=(0.0, 1.0, 0.0)):
        """算出「位于 eye、看向 target」的相机四元数。

        MuJoCo 相机看自身 -z、+x 右、+y 上；cam_xmat 是相机系→世界系，它的**列**
        就是相机的三个基向量。mju_mat2Quat 要的是行主序展平，所以这里先按列拼再展平。
        """
        f = np.asarray(target, float) - np.asarray(eye, float)
        f /= np.linalg.norm(f)
        r = np.cross(f, np.asarray(up, float))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.stack([r, u, -f], axis=1).flatten())
        return q

    # 操作员专用俯视角的参数。不是模型里的相机，是每帧现造的 MjvCamera ——
    # 这样不用往 XML 里加东西（SO-ARM100/ 是上游 vendored 模型，保持只读）。
    TOP_CAM_LOOKAT = (0.16, 0.0, 0.02)
    TOP_CAM_DIST = 0.65         # ±0.27m 见方：整条手臂都在画面里，40mm 方块约 19px
    TOP_CAM_AZIMUTH = 90.0      # 这个值让画面里 +x 朝右、+y 朝上，跟 front 左右一致
    TOP_CAM_ELEVATION = -88.0   # 别用 -90：正俯视时相机的 up 向量会退化

    def top_cam(self):
        """操作员看的正上方俯视图。

        补的是 front 视角最大的盲区 —— 进深。正视画面里看不出夹爪是在方块前面
        还是后面，也看不出 pan 转了多少；俯视一眼就清楚。屏幕方向跟 front 对齐：
        +x 朝右（远离基座）、+y 朝上。

        elevation 用 -88 而不是 -90：正俯视时相机的 up 向量和视线共线，frame 退化。
        """

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.TOP_CAM_LOOKAT
        cam.distance = self.TOP_CAM_DIST
        cam.azimuth = self.TOP_CAM_AZIMUTH
        cam.elevation = self.TOP_CAM_ELEVATION
        return cam

    def _fix_wrist_cam(self):
        cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        if cid < 0:
            return
        self.m.cam_pos[cid] = self.WRIST_CAM_EYE
        self.m.cam_quat[cid] = self._look_at_quat(self.WRIST_CAM_EYE, self.WRIST_CAM_AIM)
        self.m.cam_fovy[cid] = self.WRIST_CAM_FOVY

    def level_wrist_cam(self, home_deg):
        """把腕部相机的横滚摆正：让它在 home_deg 这个姿态下画面水平 = 地平线。

        相机是刚性固定在夹爪上的，横滚**随手臂姿态变** —— 换个姿态它又歪了。所以
        「水平」只能对某一个姿态成立，这里选待命姿态：开窗后的第一眼，也是按 R
        之后回到的地方。（实测未摆正时那个姿态下横滚 +30.65°，地板线明显斜着。）

        做法不是去解一个横滚角，而是换 look-at 的 up：把**世界竖直方向**表示到
        相机的父坐标系里，拿它当 up。于是相机在 home_deg 下正好没有横滚，而眼睛
        位置和瞄准方向一个字没改 —— `_look_at_quat` 里 up 只用来定横滚，
        换句话说「看到什么」完全不变，变的只是画面绕视线的旋转。

        只写 qpos 再 mj_forward：mj_forward 不积分，探姿态只能这么探（写 ctrl
        再 forward 是看不到效果的）。探完把 qpos 原样放回去。
        """

        cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        if cid < 0:
            return
        keep = self.d.qpos.copy()
        for adr, rng, deg in zip(self.qidx, self.ranges, home_deg):
            self.d.qpos[adr] = deg_to_qpos(float(deg), rng)
        mujoco.mj_forward(self.m, self.d)

        # d.xmat 是「刚体系 → 世界」的旋转（不是 body_xmat，那个名字不存在），
        # 转置就把世界方向送回刚体系
        r_body = np.array(self.d.xmat[self.m.cam_bodyid[cid]]).reshape(3, 3)
        up_body = r_body.T @ np.array([0.0, 0.0, 1.0])
        self.m.cam_quat[cid] = self._look_at_quat(
            self.WRIST_CAM_EYE, self.WRIST_CAM_AIM, up_body)

        self.d.qpos[:] = keep
        mujoco.mj_forward(self.m, self.d)

    def reset(self, cube_xy=None, cube_yaw=0.0, home_deg=None):
        """回 home、把方块放回取件柱顶（可选水平抖动 / 绕 z 转）。"""
        self.c.reset()
        for eq in range(self.m.neq):
            self.d.eq_active[eq] = 0
            self.m.eq_active0[eq] = 0
        if home_deg is not None:
            self.set_action(np.asarray(home_deg, dtype=np.float64))
        if cube_xy is not None:
            a = self.cube_qadr
            self.d.qpos[a:a + 3] = (cube_xy[0], cube_xy[1], paths.CUBE_Z)
            h = cube_yaw / 2.0
            self.d.qpos[a + 3:a + 7] = (np.cos(h), 0.0, 0.0, np.sin(h))
            mujoco.mj_forward(self.m, self.d)
        return self.observation()

    def set_action(self, action_deg):
        """把 6 个目标角写进 ctrl（不推进物理）。"""
        for a, rng, deg in zip(self.acts, self.ranges, action_deg):
            self.d.ctrl[a] = deg_to_qpos(float(np.clip(deg, 0.0, 180.0)), rng)

    def step(self, action_deg, n_frames=1):
        """执行动作，推进 n_frames 个控制周期。"""
        self.set_action(action_deg)
        for _ in range(n_frames * STEPS_PER_FRAME):
            mujoco.mj_step(self.m, self.d)
        return self.observation()

    # ---- 观测 ----

    def state_deg(self):
        """6 关节当前角度（舵机 0~180° 口径）。"""
        return np.array([qpos_to_deg(float(self.d.qpos[i]), r)
                         for i, r in zip(self.qidx, self.ranges)], dtype=np.float32)

    def observation(self):
        obs = {"observation.state": self.state_deg()}
        if self.render:
            for key, cam in CAMERAS.items():
                obs[f"observation.images.{key}"] = self.c.render_observation(
                    camera=cam, height=paths.IMG_H, width=paths.IMG_W)
        return obs

    def render_view(self, mjv_cam):
        """渲染一个额外的自由视角（RGB HxWx3 uint8），**只给操作员看**。

        刻意不进 observation()：观测里的每一路都会变成策略的输入、写进数据集，
        加一路就是改了策略口径。这个函数是给遥操作界面开的后门，采数据时不参与。

        也刻意不接 height/width —— render_observation 的离屏 renderer 只在首次
        调用时创建、尺寸从此定死，另传一个分辨率不会新建、只会渲出错尺寸的图
        （踩过）。所以尺寸永远是 observation() 那套，要小图自己缩。
        """
        if not hasattr(self.c, "_renderer"):
            raise RuntimeError("先调一次 observation() 把离屏 renderer 建起来")
        self.c._renderer.update_scene(self.c._sync_render_data(), camera=mjv_cam)
        return self.c._renderer.render()

    def cube_pos(self):
        return self.d.xpos[self.cube_b].copy()

    # ---- 判定 ----

    def n_cube_contacts(self):
        return n_gripper_cube_contacts(self.m, self.d, self.cube_b, self.gripper_bodies)

    def is_success(self):
        return cube_placed(self.m, self.d, self.cube_b, self.gripper_bodies)
