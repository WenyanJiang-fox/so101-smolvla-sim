import logging
import math
import threading
import time

import mujoco
import mujoco.viewer  # noqa: F401
import numpy as np

logger = logging.getLogger("so101.controller")

SERVO_TO_JOINT = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
                  4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
HOME_ANGLES_DEG = {1: 90.0, 2: 90.0, 3: 40.0, 4: 90.0, 5: 0.0, 6: 90.0}  # servo5=0 水平开合，夹爪指不向下伸到桌面

GRIPPER_CLOSE_DEG = 0.0
GRIPPER_OPEN_DEG = 180.0
GRASP_DIST_THRESHOLD = 0.12  # grasp_point 到方块中心 < 12cm 判定可 weld 焊住
GRASP_APPROACH = 0.10  # grasp 时 grasp_point 停在方块中心上方的高度（= weld 激活后 cube 相对 gp 的 z 偏移）


def _quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(a, b):
    wa, xa, ya, za = a
    wb, xb, yb, zb = b
    return np.array([
        wa * wb - xa * xb - ya * yb - za * zb,
        wa * xb + xa * wb + ya * zb - za * yb,
        wa * yb - xa * zb + ya * wb + za * xb,
        wa * zb + xa * yb - ya * xb + za * wb,
    ])


def deg_to_qpos(angle_deg, jnt_range):
    lo, hi = jnt_range
    return lo + (angle_deg / 180.0) * (hi - lo)


def qpos_to_deg(qpos, jnt_range):
    lo, hi = jnt_range
    return (qpos - lo) / (hi - lo) * 180.0


class MujocoServoController:
    """ServoControllerStandalone 的 MuJoCo 孪生实现：同接口，驱动 SO-101 模型。

    viewer 模式用 `mujoco.viewer.launch`（主动 viewer，物理+渲染同线程），
    工具只写 data.ctrl，由 viewer 的物理循环平滑驱动机械臂；
    headless 模式则直接置 qpos 瞬移，便于确定性单元测试。
    """

    def __init__(self, mjcf_path, headless=True):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self._lock = threading.Lock()
        self._jnt_id = {}
        self._qpos_adr = {}
        self._range = {}
        for jname in SERVO_TO_JOINT.values():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self._jnt_id[jname] = jid
            self._qpos_adr[jname] = self.model.jnt_qposadr[jid]
            self._range[jname] = (float(self.model.jnt_range[jid][0]),
                                  float(self.model.jnt_range[jid][1]))
        self._act_id = {}
        for servo_id, jname in SERVO_TO_JOINT.items():
            self._act_id[servo_id] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)

        # 初始 ctrl 对齐当前 qpos，避免启动后关节被位置执行器拉回 0
        for servo_id, jname in SERVO_TO_JOINT.items():
            self.data.ctrl[self._act_id[servo_id]] = self.data.qpos[self._qpos_adr[jname]]

        # 抓取相关对象（可选，grasp_scene 才有）
        self._cube_id = self._try_id(mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._gripper_id = self._try_id(mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self._target_id = self._try_id(mujoco.mjtObj.mjOBJ_BODY, "target")
        self._gripperframe_id = self._try_id(mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._grasp_point_id = self._try_id(mujoco.mjtObj.mjOBJ_SITE, "grasp_point")
        self._grasp_eq_id = self._try_id(mujoco.mjtObj.mjOBJ_EQUALITY, "grasp")

        self._physics_thread = None
        self._physics_running = threading.Event()
        self._viewer_thread = None
        if not headless:
            self._viewer_thread = threading.Thread(
                target=mujoco.viewer.launch, args=(self.model, self.data), daemon=True)
            self._viewer_thread.start()

    def _try_id(self, obj_type, name):
        try:
            return mujoco.mj_name2id(self.model, obj_type, name)
        except Exception:
            return -1

    def _sync_render_data(self):
        """把当前状态同步到独立 data 并前向运动学（避免与 viewer 物理循环竞争崩溃）。"""
        if not hasattr(self, "_render_data"):
            self._render_data = mujoco.MjData(self.model)
        with self._lock:
            self._render_data.qpos[:] = self.data.qpos
            self._render_data.qvel[:] = self.data.qvel
        mujoco.mj_forward(self.model, self._render_data)
        return self._render_data

    def start_physics_loop(self):
        """启动后台物理循环（连续步进，实现平滑运动 + 实时渲染）。"""
        if self._physics_thread is not None:
            return
        self._physics_running.set()
        self._physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self._physics_thread.start()

    def _physics_loop(self):
        while self._physics_running.is_set():
            with self._lock:
                mujoco.mj_step(self.model, self.data)
            time.sleep(self.model.opt.timestep)

    # ---- 基础舵机接口（与 ServoControllerStandalone 一致）----

    def set_servo_angle(self, servo_id, angle, speed=0):
        angle = float(angle)
        if not np.isfinite(angle):
            return False
        angle = max(0.0, min(180.0, angle))
        jname = SERVO_TO_JOINT[servo_id]
        target = deg_to_qpos(angle, self._range[jname])
        aid = self._act_id[servo_id]
        adr = self._qpos_adr[jname]
        with self._lock:
            self.data.ctrl[aid] = target
            if self._viewer_thread is None and not self._physics_running.is_set():
                self.data.qpos[adr] = target
                mujoco.mj_step(self.model, self.data)
        return True

    def get_servo_angle(self, servo_id):
        jname = SERVO_TO_JOINT[servo_id]
        qpos = float(self.data.qpos[self._qpos_adr[jname]])
        return qpos_to_deg(qpos, self._range[jname])

    def set_multiple_servo_angles(self, angles, speed=0):
        for sid, ang in angles.items():
            self.set_servo_angle(int(sid), float(ang), speed)
        return True

    def get_multiple_servo_angles(self, servo_ids):
        return {sid: self.get_servo_angle(sid) for sid in servo_ids}

    def get_state_action(self):
        """返回 (state, action)，均为 6 关节弧度值：
        state=当前 qpos，action=当前执行器 ctrl（命令目标）。"""
        state = np.array([float(self.data.qpos[self._qpos_adr[jname]])
                          for jname in SERVO_TO_JOINT.values()], dtype=np.float32)
        action = np.array([float(self.data.ctrl[self._act_id[sid]])
                           for sid in SERVO_TO_JOINT.keys()], dtype=np.float32)
        return state, action

    def go_home(self, speed=0):
        return self.set_multiple_servo_angles(HOME_ANGLES_DEG, speed)

    def sync_enable_torque(self, servo_ids):
        return True

    def sync_disable_torque(self, servo_ids):
        return True

    # ---- 抓取接口 ----

    def _gripper_cube_distance(self):
        if self._gripperframe_id < 0 or self._cube_id < 0:
            return float("inf")
        rd = self._sync_render_data()
        return float(np.linalg.norm(
            rd.site_xpos[self._gripperframe_id] - rd.xpos[self._cube_id]))

    def _set_grasp_active(self, active):
        if self._grasp_eq_id < 0:
            return
        with self._lock:
            self.data.eq_active[self._grasp_eq_id] = 1 if active else 0

    def _update_weld_pose(self):
        """用当前相对位姿更新 weld 的 relpos[3:6]+relquat[6:10]，激活时不瞬移。

        weld 约束的 relpose 在编译期从 q0 定死，若直接激活会把方块拽回 q0 位姿（甩飞）。
        这里把 body2(cube) 相对 body1(gripper) 的当前位姿写回 eq_data。
        """
        if self._grasp_eq_id < 0:
            return
        rd = self._sync_render_data()
        p1 = rd.xpos[self._gripper_id]
        R1 = rd.xmat[self._gripper_id].reshape(3, 3)
        p2 = rd.xpos[self._cube_id]
        relpos = R1.T @ (p2 - p1)
        q1 = rd.xquat[self._gripper_id]
        q2 = rd.xquat[self._cube_id]
        relquat = _quat_mul(_quat_inv(q1), q2)
        with self._lock:
            self.model.eq_data[self._grasp_eq_id, 3:6] = relpos
            self.model.eq_data[self._grasp_eq_id, 6:10] = relquat

    def close_gripper(self):
        """闭合夹爪并激活 weld 吸附（运动学快照抓取）。

        激活前用当前相对位姿更新 weld 的 relpos/relquat，避免约束用编译期 q0 的
        相对位姿把方块甩飞。
        """
        self.set_servo_angle(6, GRIPPER_CLOSE_DEG)
        if self._grasp_eq_id < 0:
            return False
        rd = self._sync_render_data()
        gp = rd.site_xpos[self._grasp_point_id]
        cube = self.get_cube_position()
        if cube is None:
            return False
        dist = float(np.linalg.norm(gp - np.array(cube)))
        logger.info(f"[抓取] 夹爪距方块 {dist*1000:.1f}mm (阈值 {GRASP_DIST_THRESHOLD*1000:.0f}mm)")
        if dist < GRASP_DIST_THRESHOLD:
            self._update_weld_pose()
            self._set_grasp_active(True)
            logger.info("[抓取] weld 吸附激活成功")
            return True
        logger.warning("[抓取] 距离过远，未吸附")
        return False

    def _gripper_contacts_cube(self):
        """检查爪子（gripper + moving_jaw）是否接触到方块。"""
        if self._cube_id < 0:
            return False
        gripper_bodies = {self._gripper_id}
        jaw_id = self._try_id(mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        if jaw_id >= 0:
            gripper_bodies.add(jaw_id)
        with self._lock:
            mujoco.mj_collision(self.model, self.data)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            ba = self.model.geom_bodyid[c.geom1]
            bb = self.model.geom_bodyid[c.geom2]
            if (ba in gripper_bodies and bb == self._cube_id) or (bb in gripper_bodies and ba == self._cube_id):
                return True
        return False

    def open_gripper(self):
        """释放方块（deactivate weld）+ 打开夹爪。"""
        self._set_grasp_active(False)
        self.set_servo_angle(6, GRIPPER_OPEN_DEG)
        return True

    def _body_pos(self, body_id):
        if body_id < 0:
            return None
        rd = self._sync_render_data()
        return tuple(float(v) for v in rd.xpos[body_id])

    def get_cube_position(self):
        return self._body_pos(self._cube_id)

    def get_gripper_position(self):
        return self._body_pos(self._gripper_id)

    def get_target_position(self):
        return self._body_pos(self._target_id)

    def render_observation(self, camera="wrist_cam", height=240, width=320):
        """渲染相机图像观测（RGB numpy 数组 HxWx3 uint8），Perception 层核心。"""
        cam_id = self._try_id(mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            return None
        if not hasattr(self, "_renderer"):
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        rd = self._sync_render_data()
        self._renderer.update_scene(rd, camera=camera)
        return self._renderer.render()

    def start_camera_views(self, cameras=("wrist_cam", "global_cam"), height=480, width=640, gamma=1.6):
        """后台线程实时渲染多个相机到独立 OpenCV 窗口（与运行中的仿真同步）。

        每个相机一个窗口。renderer 必须在同一线程内创建和渲染（Windows WGL 线程亲和，
        主线程创建、后台线程 render 会因 context 切不过去而全黑）。按 q 关所有窗口。
        场景偏暗，用 gamma 提亮让画面看清。
        """
        if getattr(self, "_live_view_thread", None) is not None:
            return True
        import cv2
        cameras = list(cameras)

        def _loop():
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            rd = mujoco.MjData(self.model)
            while True:
                with self._lock:
                    rd.qpos[:] = self.data.qpos
                    rd.qvel[:] = self.data.qvel
                mujoco.mj_forward(self.model, rd)
                for cam in cameras:
                    renderer.update_scene(rd, camera=cam)
                    img = renderer.render()
                    img = ((img.astype(np.float32) / 255.0) ** (1.0 / gamma) * 255.0).astype(np.uint8)
                    cv2.imshow(f"{cam} (live)", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

        self._live_view_thread = threading.Thread(target=_loop, daemon=True)
        self._live_view_thread.start()
        return True

    # ---- 感知：图像 -> 3D 坐标（去作弊，替代直读 data.xpos）----

    @staticmethod
    def _color_mask(img, color):
        """颜色分割掩码。方块红 rgba(0.9,0.2,0.2)，机械臂橙 rgba(1,0.82,0.12)，目标区绿 rgba(0.2,0.9,0.2)。"""
        r = img[..., 0].astype(int)
        g = img[..., 1].astype(int)
        b = img[..., 2].astype(int)
        if color == "red":
            return (r > 150) & (g < 100) & (b < 100)
        if color == "green":
            return (g > 100) & (r < 100) & (b < 100)
        return None

    @staticmethod
    def _largest_blob_centroid(mask):
        """最大连通域的质心（滤掉反光/抗锯齿散点噪声，避免散点带偏均值）。返回 (cx, cy) 或 None。"""
        import cv2
        if mask is None or int(mask.sum()) == 0:
            return None
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if n < 2:
            return None
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return float(centroids[largest][0]), float(centroids[largest][1])

    def _camera_params(self, cam_name, height, width):
        """相机内参（fx=fy，主点在图像中心）+ cam_id，从 fovy（度）与渲染分辨率算。"""
        cam_id = self._try_id(mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        fovy = math.radians(float(self.model.cam_fovy[cam_id]))
        fy = (height / 2.0) / math.tan(fovy / 2.0)
        fx = fy
        return cam_id, fx, fy, width / 2.0, height / 2.0

    def _backproject(self, cam_name, cx, cy, z_plane, height, width):
        """像素 (cx,cy) 反投影到世界系 z=z_plane 平面，返回 (x,y,z)。

        MuJoCo 相机约定（已用真值验证，误差 ~3mm）：
        - cam_xmat 是 cam->world（不转置）。
        - 相机看向 -z，故 d_cam 的 z 分量为 -1。
        - 图像 y 轴向下，故 y 分量用 (cy_img - cy_pixel)。
        """
        cam_id, fx, fy, cxp, cyp = self._camera_params(cam_name, height, width)
        rd = self._sync_render_data()
        cam_pos = rd.cam_xpos[cam_id].copy()
        cam_mat = rd.cam_xmat[cam_id].reshape(3, 3).copy()  # cam -> world
        d_cam = np.array([(cx - cxp) / fx, (cyp - cy) / fy, -1.0])  # 指向场景
        d_world = cam_mat @ d_cam
        t = (z_plane - cam_pos[2]) / d_world[2]
        return tuple(float(v) for v in (cam_pos + t * d_world))

    def _render_depth(self, camera, height, width):
        """渲染深度图（float32 HxW，值为沿相机 -z 轴到物体表面的距离）。"""
        cam_id = self._try_id(mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            return None
        if not hasattr(self, "_depth_renderer") or getattr(self, "_depth_size", None) != (height, width):
            self._depth_renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._depth_size = (height, width)
        rd = self._sync_render_data()
        self._depth_renderer.update_scene(rd, camera=camera)
        self._depth_renderer.enable_depth_rendering()
        depth = self._depth_renderer.render()
        self._depth_renderer.disable_depth_rendering()
        return depth

    def _depth_backproject(self, cam_name, cx, cy, depth, height, width):
        """深度图像素 (cx,cy) 反投影到世界 3D：沿相机 -z 距离 depth 处的表面点。"""
        cam_id, fx, fy, cxp, cyp = self._camera_params(cam_name, height, width)
        rd = self._sync_render_data()
        cam_pos = rd.cam_xpos[cam_id].copy()
        cam_mat = rd.cam_xmat[cam_id].reshape(3, 3).copy()
        d_cam = np.array([(cx - cxp) / fx, (cyp - cy) / fy, -1.0])
        return tuple(float(v) for v in (cam_pos + cam_mat @ (d_cam * float(depth))))

    def detect_cube(self, camera="global_cam", height=224, width=224):
        """图像感知方块体心：红色最大连通域质心 → 深度反投影顶面 → 减半尺寸得体心。

        用深度图直接读物体表面距离（不再假设固定 z 平面），比 z 平面反投影更准，
        且方块被抬高时也能正确返回实际高度。
        """
        if self._cube_id < 0:
            return None
        img = self.render_observation(camera, height=height, width=width)
        if img is None:
            return None
        centroid = self._largest_blob_centroid(self._color_mask(img, "red"))
        if centroid is None:
            logger.warning("[感知] detect_cube 未在图像中找到红色方块")
            return None
        depth = self._render_depth(camera, height, width)
        if depth is None:
            return None
        cx = int(round(centroid[0]))
        cy = int(round(centroid[1]))
        surface = self._depth_backproject(camera, cx, cy, depth[cy, cx], height, width)
        gid = self._try_id(mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        half = float(self.model.geom_size[gid][2]) if gid >= 0 else 0.02
        res = (surface[0], surface[1], surface[2] - half)
        logger.info(f"[感知] detect_cube 方块定位: x={res[0]:.3f} y={res[1]:.3f} z={res[2]:.3f}")
        return res

    def detect_target(self, camera="global_cam", height=224, width=224):
        """图像感知目标区 xy：绿色最大连通域质心 → 反投影到目标区高度（z 固定为标记高度）。"""
        img = self.render_observation(camera, height=height, width=width)
        if img is None:
            return None
        centroid = self._largest_blob_centroid(self._color_mask(img, "green"))
        if centroid is None:
            logger.warning("[感知] detect_target 未在图像中找到绿色目标区")
            return None
        z_plane = float(self.model.body_pos[self._target_id][2])  # 目标区高度
        res = self._backproject(camera, centroid[0], centroid[1], z_plane, height, width)
        logger.info(f"[感知] detect_target 目标区定位: x={res[0]:.3f} y={res[1]:.3f} z={res[2]:.3f}")
        return res

    def _camera_sees_color(self, camera, color, min_area=30, height=224, width=224):
        """判断相机画面里是否看到目标颜色（最大连通域面积 ≥ min_area 才算看见，滤掉散点噪声）。"""
        img = self.render_observation(camera, height=height, width=width)
        if img is None:
            logger.warning(f"[门控] {camera} 渲染失败")
            return False
        mask = self._color_mask(img, color)
        if mask is None or int(mask.sum()) == 0:
            logger.info(f"[门控] {camera} 未看到 {color} 目标（0 像素）")
            return False
        import cv2
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if n < 2:
            logger.info(f"[门控] {camera} 未看到 {color} 目标")
            return False
        area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
        seen = area >= min_area
        logger.info(f"[门控] {camera} 看到 {color} 目标: 最大连通域面积={area}px (阈值{min_area}) → {'通过' if seen else '未通过'}")
        return seen

    def _ik_angles(self, x, y, z, x0=None, site_id=None, wrist_deg=0.0):
        """数值 IK：解 servo 1-4 关节角（度），wrist_roll 钉在 wrist_deg（默认 0° = 夹爪水平开合）。

        wrist_deg 是用来做**腕转对齐**的：方块偏航一斜，爪开合轴跟方块棱差 45° 时
        跨距 56.6mm > 爪全开 55.9mm，爪板卡在顶棱上一闭合就把方块挤出去。转腕对
        齐后跨距回到 40mm（面抓）。注意腕角必须在求解**之内**钉住——解完再转腕会
        把 grasp_point 甩离目标（实测偏 16mm）。失败返回 None。
        """
        from scipy.optimize import least_squares

        if site_id is None:
            site_id = self._gripperframe_id
        if site_id < 0:
            return None
        ik_data = mujoco.MjData(self.model)
        j5 = SERVO_TO_JOINT[5]
        ik_data.qpos[self._qpos_adr[j5]] = deg_to_qpos(wrist_deg, self._range[j5])
        if x0 is None:
            x0 = np.clip(np.array([self.get_servo_angle(i) for i in range(1, 5)]), 0.0, 180.0)

        def residual(a):
            for i, ang in enumerate(a, start=1):
                jname = SERVO_TO_JOINT[i]
                ik_data.qpos[self._qpos_adr[jname]] = deg_to_qpos(ang, self._range[jname])
            mujoco.mj_forward(self.model, ik_data)
            p = ik_data.site_xpos[site_id]
            return np.array([p[0] - x, p[1] - y, p[2] - z])

        result = least_squares(residual, x0, bounds=(0.0, 180.0))
        if not np.all(np.isfinite(result.x)):
            return None
        return result.x

    def _follow_cartesian_path(self, waypoints, duration_per_seg=1.5, num_waypoints=15,
                               site_id=None, wrist_deg=0.0):
        """沿一串笛卡尔路点逐段平滑运动（最小 jerk 时间参数化 + 每路点 IK）。"""
        self.set_servo_angle(5, wrist_deg)  # 固定 wrist_roll（默认水平开合）
        prev = None
        for k in range(len(waypoints) - 1):
            a = np.array(waypoints[k], dtype=float)
            b = np.array(waypoints[k + 1], dtype=float)
            for i in range(1, num_waypoints + 1):
                t = i / num_waypoints
                s = 10.0 * t ** 3 - 15.0 * t ** 4 + 6.0 * t ** 5
                wp = a + s * (b - a)
                angles = self._ik_angles(wp[0], wp[1], wp[2], x0=prev, site_id=site_id,
                                         wrist_deg=wrist_deg)
                if angles is None:
                    return False
                prev = np.array(angles)
                for j, ang in enumerate(angles, start=1):
                    self.set_servo_angle(j, float(ang))
                time.sleep(duration_per_seg / num_waypoints)
        return True

    def move_to(self, x, y, z, duration=1.5, num_waypoints=15, site_id=None, wrist_deg=0.0):
        """平滑移动到目标；若直线路径撞障碍物，自动用 RRT 避障绕行。

        site_id 指定 IK 目标 site（默认 gripperframe；抓取时用 grasp_point）。
        wrist_deg 指定全程钉住的 wrist_roll 角（见 _ik_angles）。
        """
        if site_id is None:
            site_id = self._gripperframe_id
        if site_id < 0:
            return False
        rd = self._sync_render_data()
        start = np.array(rd.site_xpos[site_id], dtype=float)
        goal = np.array([x, y, z], dtype=float)
        if getattr(self, "_obstacles", None) and not self._edge_free(start, goal):
            path = self._rrt_path(start, goal)
            if path is None:
                return False
            return self._follow_cartesian_path(path, duration_per_seg=duration, num_waypoints=num_waypoints, site_id=site_id, wrist_deg=wrist_deg)
        return self._follow_cartesian_path([start, goal], duration_per_seg=duration, num_waypoints=num_waypoints, site_id=site_id, wrist_deg=wrist_deg)

    def move_to_grasp_point(self, x, y, z, duration=1.5, num_waypoints=15, wrist_deg=0.0):
        """把爪子 pinch 点（grasp_point）平滑移动到目标，用于物理抓取对准。"""
        return self.move_to(x, y, z, duration=duration, num_waypoints=num_waypoints,
                            site_id=self._grasp_point_id, wrist_deg=wrist_deg)

    # ---- 避障（运动规划）----

    def set_obstacles(self, obstacles):
        """设置障碍物列表，每项 (中心 [x,y,z], 半尺寸 [hx,hy,hz])。"""
        self._obstacles = [(np.array(c, dtype=float), np.array(h, dtype=float)) for c, h in obstacles]

    def _point_free(self, p, clearance=0.04):
        for c, h in getattr(self, "_obstacles", []):
            if all(abs(p[i] - c[i]) <= h[i] + clearance for i in range(3)):
                return False
        return True

    def _edge_free(self, a, b, clearance=0.04):
        n = max(2, int(np.linalg.norm(b - a) / 0.008))
        for i in range(n + 1):
            if not self._point_free(a + (b - a) * (i / n), clearance):
                return False
        return True

    def _rrt_path(self, start, goal, num_iter=1500, step=0.025):
        """RRT 在笛卡尔空间规划无碰撞路径，返回路点列表或 None。"""
        import random
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)
        if self._edge_free(start, goal):
            return [start, goal]
        tree = [start]
        parent = {0: -1}
        for _ in range(num_iter):
            if random.random() < 0.15:
                q_rand = goal
            else:
                q_rand = np.array([random.uniform(-0.05, 0.36),
                                   random.uniform(-0.12, 0.12),
                                   random.uniform(0.08, 0.5)])
            dists = [np.linalg.norm(q_rand - n) for n in tree]
            idx = int(np.argmin(dists))
            nearest = tree[idx]
            d = q_rand - nearest
            norm = np.linalg.norm(d)
            if norm < 1e-6:
                continue
            q_new = nearest + d / norm * min(step, norm)
            if self._edge_free(nearest, q_new):
                tree.append(q_new)
                parent[len(tree) - 1] = idx
                if np.linalg.norm(q_new - goal) < 0.02:
                    path = [goal]
                    j = len(tree) - 1
                    while j != 0:
                        path.append(tree[j])
                        j = parent[j]
                    path.append(start)
                    return list(reversed(path))
        return None

    def move_to_avoid(self, x, y, z, duration_per_seg=1.0):
        """带避障的平滑移动：RRT 规划无碰撞路径，再沿路径平滑运动。"""
        if self._gripperframe_id < 0:
            return False
        rd = self._sync_render_data()
        start = np.array(rd.site_xpos[self._gripperframe_id], dtype=float)
        goal = np.array([x, y, z], dtype=float)
        path = self._rrt_path(start, goal)
        if path is None:
            return False
        return self._follow_cartesian_path(path, duration_per_seg=duration_per_seg)

    def grasp_cube(self):
        """图像感知定位 → 下探到方块上方 → 手部相机确认看到方块才张爪吸附（感知门控）。"""
        logger.info("========== grasp_cube 开始 ==========")
        cube = self.detect_cube()
        if cube is None:
            logger.warning("[抓取] 全局相机未定位到方块，放弃")
            return False
        self._set_grasp_active(False)  # 解除可能残留的 weld
        # 三段式接近：垂直升高 -> 高空平移 -> 停在方块中心上方（先不张爪）
        rd = self._sync_render_data()
        cur = rd.site_xpos[self._grasp_point_id]
        safe_z = cube[2] + 0.20  # 方块上方安全高度
        logger.info(f"[抓取] 三段式下探 → 目标 ({cube[0]:.3f}, {cube[1]:.3f}, {cube[2]+GRASP_APPROACH:.3f})")
        self.move_to_grasp_point(cur[0], cur[1], safe_z)          # ① 垂直升高
        self.move_to_grasp_point(cube[0], cube[1], safe_z)        # ② 高空平移
        self.move_to_grasp_point(cube[0], cube[1], cube[2] + GRASP_APPROACH)  # ③ 停到方块中心上方
        # 感知门控：只有手部相机确实看到红色方块，才允许张爪抓取
        if not self._camera_sees_color("wrist_cam", "red"):
            logger.warning("[抓取] 手部相机未看到方块，拒绝张爪")
            return False
        self.open_gripper()  # 确认看到目标后才张爪
        result = self.close_gripper()  # 激活 weld（运动学吸附，方块此后跟随夹爪）
        logger.info(f"========== grasp_cube 结束: {'成功' if result else '失败'} ==========")
        return result

    def place_at(self, x, y, z):
        """把抓着的方块放到 (x,y,z)：带方块到目标上方 → 手部相机确认绿色目标区 → 释放并置位。"""
        logger.info(f"========== place_at 开始 → 目标 ({x:.3f}, {y:.3f}, {z:.3f}) ==========")
        # 带方块（weld 仍激活）到目标上方，手部相机确认看到绿色目标区
        self.move_to_grasp_point(x, y, z + 0.15)
        if not self._camera_sees_color("wrist_cam", "green"):
            logger.warning("[放置] 手部相机未看到绿色目标区，拒绝放置")
            return False
        self._set_grasp_active(False)  # 释放 weld
        self.open_gripper()
        # 先把机械臂升高，避免夹爪指和放置的方块碰撞
        rd = self._sync_render_data()
        cur = rd.site_xpos[self._grasp_point_id]
        self.move_to_grasp_point(cur[0], cur[1], cur[2] + 0.15)
        # 精确置位：直接写方块自由关节（保持直立、速度清零）
        result = self.set_cube_position(x, y, z)
        logger.info(f"========== place_at 结束: {'成功' if result else '失败'} ==========")
        return result

    def place_cube(self):
        """图像感知目标区并放置方块：detect_target 定位 -> 释放 weld -> 置位到目标区上方。

        全程不读坐标，LLM 只需调用 grasp_cube() 再 place_cube() 即可完成 pick-and-place。
        """
        target = self.detect_target()
        if target is None:
            return False
        gid = self._try_id(mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        half = float(self.model.geom_size[gid][2]) if gid >= 0 else 0.02  # 方块半高
        return self.place_at(target[0], target[1], half)  # 方块坐地面（z=half）

    def reset(self):
        """复位：数据回到初始、解除抓取、臂回 home。"""
        with self._lock:
            # 手动复位（避免 mj_resetData 与后台物理循环线程的潜在段错误）
            self.data.qpos[:] = self.model.qpos0
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
            if self._grasp_eq_id >= 0:
                self.data.eq_active[self._grasp_eq_id] = 0
            mujoco.mj_forward(self.model, self.data)
        self.go_home()
        return True

    def set_cube_position(self, x, y, z):
        """把方块（自由体）移到 (x, y, z)，朝向归位、速度清零。"""
        if self._cube_id < 0:
            return False
        jnt_adr = self.model.body_jntadr[self._cube_id]
        jnt_num = self.model.body_jntnum[self._cube_id]
        with self._lock:
            for j in range(jnt_adr, jnt_adr + jnt_num):
                if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    adr = self.model.jnt_qposadr[j]
                    self.data.qpos[adr] = x
                    self.data.qpos[adr + 1] = y
                    self.data.qpos[adr + 2] = z
                    self.data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
                    self.data.qvel[adr:adr + 6] = 0.0
                    break
            mujoco.mj_forward(self.model, self.data)
        return True
