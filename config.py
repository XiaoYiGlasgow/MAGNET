import os

# === NuPlan Dataset Configuration ===
DATA_PATH = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/train_pittsburgh"
MAP_PATH = '/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
ANALYSIS_RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "analysis")
LSTM_RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "baselines", "lstm")
IDM_RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "baselines", "idm")
TRANSFORMER_ONLY_BASELINES_DIR = os.path.join(BASE_DIR, "experiments", "baselines", "transformer_only")
DANGER_GENERATION_RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "danger_generation")
DANGER_GENERATION_BASELINES_DIR = os.path.join(DANGER_GENERATION_RESULTS_DIR, "baselines")
DIFFUSION_TRAINING_GNN_TRANS_DIR = os.path.join(BASE_DIR, "experiments", "danger_generation", "checkpoints", "diffusion_training_gnn_trans")
DIFFUSION_TRAINING_EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments", "danger_generation", "training_runs")
DIFFUSION_VISUALIZATION_RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "danger_generation", "visualizations")
TTC_FILTER_RESULTS_DIR = os.path.join(DANGER_GENERATION_RESULTS_DIR, "ttc_filter_results")
GENERATED_DANGER_SCENES_DIR = os.path.join(DANGER_GENERATION_RESULTS_DIR, "generated_scenes")
# 生成结果筛选后的危险场景 benchmark 输出目录（B 线，benchmark_filter）
BENCHMARK_DIR = os.path.join(DANGER_GENERATION_RESULTS_DIR, "benchmark")
ALL_RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "analysis", "gnn_trans_ablation")
ALL_TRAINING_DIR = os.path.join(BASE_DIR, "experiments", "analysis", "training_records")
COMPARISON_RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "analysis", "comparisons")

# === Data Directories ===
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
PROCESSED_DATA_BY_TYPE_DIR = os.path.join(DATA_DIR, "processed_data_by_type")
DANGER_PAIR_SCENARIOS_DIR = os.path.join(DATA_DIR, "danger_pair_scenarios")
DANGER_PAIR_REPORTS_DIR = os.path.join(DATA_DIR, "danger_pair_reports")

# === Diffusion Danger Generation Data (diffusion_danger_model/) ===
DIFFUSION_DATA_DIR = os.path.join(BASE_DIR, "diffusion_danger_model")
DIFFUSION_PROCESSED_DATA_DIR = os.path.join(DIFFUSION_DATA_DIR, "processed_data")
DIFFUSION_DANGER_PAIR_DIR = os.path.join(DIFFUSION_DATA_DIR, "danger_pair_scenarios")
# New pipeline: danger pairs produced by diffusion_danger_model/dataset/preprocess.py
NEW_DANGER_PAIR_DIR = os.path.join(DIFFUSION_DATA_DIR, "dataset", "danger_pairs")

SAVE_PATH = PROCESSED_DATA_DIR
TRAIN_PATH = os.path.join(PROCESSED_DATA_BY_TYPE_DIR, "processed_medium_magnitude_speed")
TEST_PATH = os.path.join(PROCESSED_DATA_BY_TYPE_DIR, "processed_medium_magnitude_speed")
MAP_VERSION = "nuplan-maps-v1.0"

SCENARIOS_PER_TYPE = 1000000
TOTAL_SCENARIOS = 20000000

# === Temporal Configuration ===
DISCRETE_SIZE = 10  # 帧率：1秒10帧
PAST_TIME_HORIZON = 2   # 2秒历史（对齐预处理执行值；模型 ego 分支锁死 20 帧）
FUTURE_TIME_HORIZON = 6 # 6秒未来

NUM_PAST_POSES = DISCRETE_SIZE * PAST_TIME_HORIZON      # 20帧
NUM_FUTURE_POSES = DISCRETE_SIZE * FUTURE_TIME_HORIZON  # 60帧

# === Agent / Lane / Map Configuration ===
NUM_AGENTS = 20  # 对齐预处理执行值（原 15）

# --- MAGNET LANE-only 地图契约（决策A：仅 LANE 进 token）---
LANE_NUM = 40          # 对齐预处理执行值（原 30）
LANE_POINTS_NUM = 50   # 对齐预处理执行值（原 31）

# --- 以下 route/crosswalk 常量仅供 src/danger_generation/ 旧线脚本消费，MAGNET LANE-only 不使用 ---
ROUTE_LANES_NUM = 10
CROSSWALKS_NUM = 5
ROUTE_LANES_POINTS_NUM = 51
CROSSWALKS_POINTS_NUM = 31

QUERY_RADIUS = 50  # 轨迹/地图提取半径（对齐预处理执行值，原 30）

# === MAGNET LANE-only 地图契约导出（供预处理/编码 import 的唯一源，决策A/C）===
MAP_FEATURES = ["LANE"]

# === route 注入模式（VT-2026-04191 阶段⑬）===
# 'none'          : lane 节点 col6/col7/col8 强制全 0，与 route 引入前逐字节一致（可复现旧基线）
# 'three_channel' : col6=route 邻近度, col7=沿 route 进度, col8=is_on_route
# 预处理与在线 planner 必须使用同一值；训练会把它写入 checkpoint 同目录 train_meta.json，
# planner 加载时做一致性校验（失配默认 raise）。
ROUTE_MODES = ("none", "three_channel")
ROUTE_MODE = "none"
MAX_ELEMENTS = {"LANE": LANE_NUM}   # 形状对齐 create_default_config 的 dict
MAX_POINTS = {"LANE": LANE_POINTS_NUM}
MAP_RADIUS = QUERY_RADIUS
INTERPOLATION = "linear"

# === Training Configuration ===
WEIGHT_PATH = os.path.join(DATA_DIR, 'model_weights')

SEED = 42
EPOCHS = 100
BATCH_SIZE = 8

LEARNING_RATE = 1e-3
IN_CHANNELS = 9  # 使用9维特征
OUT_CHANNELS = NUM_FUTURE_POSES * 2  # 每个时间步预测 x, y 坐标

DECAY_LR_EVERY = 10
DECAY_LR_FACTOR = 0.5

SHOW_EVERY = 100
VAL_EVERY = 5

VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# === Transformer Architecture Hyperparameters ===
POLYLINE_FEATURE_DIM = IN_CHANNELS  # 输入特征维度
EMBED_DIM = 128                     # 嵌入维度
NUM_HEADS = 8                       # 多头注意力头数
FF_DIM = 256                        # 前馈网络隐藏层维度
NUM_LAYERS = 3                      # Transformer编码层数
OUTPUT_DIM = OUT_CHANNELS          # 输出维度

HIDDEN_DIM = 64
