"""
Diffusion Danger Model - 配置文件

用于生成危险场景数据，支持Diffusion模型训练。
"""

import os

# === 路径配置 ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# NuPlan数据路径 - 只处理Pittsburgh
NUPLAN_DATA_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/train_pittsburgh"
NUPLAN_MAP_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps"
NUPLAN_SENSOR_ROOT = None  # 不需要传感器数据
NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"

# 输出路径 - processed_data在diffusion_danger_model目录下
SAVE_PATH = os.path.join(BASE_DIR, "processed_data")

# === 时间配置 ===
DISCRETE_SIZE = 10  # 帧率：1秒10帧
PAST_TIME_HORIZON = 4   # 4秒历史
FUTURE_TIME_HORIZON = 6 # 6秒未来

NUM_PAST_POSES = DISCRETE_SIZE * PAST_TIME_HORIZON      # 40帧
NUM_FUTURE_POSES = DISCRETE_SIZE * FUTURE_TIME_HORIZON  # 60帧

# === Agent / Lane / Map Configuration ===
NUM_AGENTS = 30  # 最大周围车辆数量（增加到30）

LANE_NUM = 30
ROUTE_LANES_NUM = 10
CROSSWALKS_NUM = 5

LANE_POINTS_NUM = 31
ROUTE_LANES_POINTS_NUM = 51
CROSSWALKS_POINTS_NUM = 31

QUERY_RADIUS = 50  # 轨迹提取半径（增加到50米）

# === 批量处理配置 ===
# 可以根据需要调整处理数量
MAX_SCENARIOS = None  # None表示处理全部，或设置具体数字如10000
BATCH_SIZE = 100      # 每批处理的场景数

# === 场景类型筛选 ===
# 可选的场景类型（Pittsburgh数据集实际有的类型）：
#   - 'near_multiple_vehicles': 多车辆场景
#   - 'traversing_intersection': 穿越交叉口场景
#   - 'on_intersection': 交叉口场景
#   - 'medium_magnitude_speed': 中速场景
#   - 'high_magnitude_speed': 高速场景
#   - None: 不筛选，处理所有类型
SCENARIO_TYPES = None  # 不筛选类型，处理所有场景

# 排除静止场景（位移太小）
MIN_DISPLACEMENT = 10.0  # 最小位移10米，排除静止场景

# === 日志配置 ===
LOG_EVERY = 100  # 每100个场景打印进度
SAVE_EVERY = 500  # 每500个场景保存一次