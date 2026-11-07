"""集中路径与网络常量。

HF_ENDPOINT 必须在 huggingface_hub 被 import 之前设置，所以任何脚本都要
`import paths` 放在其它 import 之前；其余模块统一从这里取路径常量。
"""
import os
from pathlib import Path

# huggingface.co 直连被墙；hf-mirror.com 可达（需带 User-Agent，huggingface_hub 自带）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# 本机无 GPU 训练需求，评估时让 transformers 不要试图联网校验
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

PROJECT = Path(__file__).resolve().parent           # 本仓库根目录
SIM2REAL = PROJECT.parent

# 模型目录：环境变量 SO101_MODEL_DIR 优先；否则用同级的 SO-ARM100 仓库；
# 都没有就用本仓库 third_party 里的副本（自包含，可直接跑 smoke_expert）。
_model_env = os.environ.get("SO101_MODEL_DIR", "").strip()
_sibling = SIM2REAL / "SO-ARM100" / "Simulation" / "SO101"
_vendored = PROJECT / "third_party" / "SO-ARM100" / "Simulation" / "SO101"
if _model_env:
    SO101_DIR = Path(_model_env)
elif _sibling.exists():
    SO101_DIR = _sibling
else:
    SO101_DIR = _vendored

# MuJoCo 控制器：原项目里在 mcp_robot/sources 下；独立仓库时代码已放到本项目根目录。
_mcp = SIM2REAL / "mcp_robot" / "sources"
MCP_SOURCES = _mcp if (_mcp / "mujoco_servo_controller.py").exists() else PROJECT

# 场景 XML 必须和 assets/ 同目录：MuJoCo 按主 XML 所在目录解析 <compiler meshdir="assets">，
# 放到别处会报 "Error opening file ... .stl"。既有的 grasp_scene.xml 同理放在这里。
SCENE_XML = SO101_DIR / "sim_scene.xml"
DATA_ROOT = Path(os.environ.get("SO101_DATA_ROOT", str(PROJECT / "data")))

# ---- 任务与场景常量（采集与评估必须共用，否则数据集对不上）----
TASK = "pick the red cube and place it on the green target"
ROBOT_TYPE = "so101"
FPS = 30
IMG_H = IMG_W = 512

# 六关节标准顺序（1=shoulder_pan … 6=gripper），全仓库统一
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

# 平桌面场景（2026-09-11）：z=0 的地面就是桌面，机械臂 base 在桌面上
# —— 对应真机「夹在桌边」。方块和目标区直接放桌上，不再有窄柱/放置台。
PICK_X, PICK_Y = 0.25, -0.05      # 方块初始位置（桌面上）
PLACE_X, PLACE_Y = 0.17, 0.05     # 绿色目标区中心（桌面上）
TABLE_Z = 0.0                     # 桌面高度
CUBE_HALF = 0.02                  # 40mm 方块
CUBE_Z = TABLE_Z + CUBE_HALF      # 0.02，方块中心初始高度
PLACE_RADIUS = 0.05               # 放置成功判定：方块中心落在目标区中心 50mm 内
PLACE_Z_TOL = 0.02                # 且在高度 ±20mm 内
TARGET_RADIUS = 0.045             # 目标区圆盘半径（纯视觉）

# 相机 → SmolVLA 的 camera1/camera2；camera3 由 empty_cameras=1 补常数图
CAM_FRONT = "camera1"
CAM_WRIST = "camera2"
