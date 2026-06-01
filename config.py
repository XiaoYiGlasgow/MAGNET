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
PAST_TIME_HORIZON = 4   # 改为4秒历史
FUTURE_TIME_HORIZON = 6 # 6秒未来

NUM_PAST_POSES = DISCRETE_SIZE * PAST_TIME_HORIZON      # 40帧
NUM_FUTURE_POSES = DISCRETE_SIZE * FUTURE_TIME_HORIZON  # 60帧

# === Agent / Lane / Map Configuration ===
NUM_AGENTS = 15

LANE_NUM = 30
ROUTE_LANES_NUM = 10
CROSSWALKS_NUM = 5

LANE_POINTS_NUM = 31
ROUTE_LANES_POINTS_NUM = 51
CROSSWALKS_POINTS_NUM = 31

NODE_SIZE = (
    CROSSWALKS_NUM * (CROSSWALKS_POINTS_NUM - 1)
    + LANE_NUM * (LANE_POINTS_NUM - 1)
    + ROUTE_LANES_NUM * (ROUTE_LANES_POINTS_NUM - 1)
    + (NUM_AGENTS + 1) * NUM_PAST_POSES  # agents + ego
)

NUM_GRAPH = (
    LANE_NUM + ROUTE_LANES_NUM + CROSSWALKS_NUM + NUM_AGENTS + 1
)

QUERY_RADIUS = 30  # 轨迹提取半径

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
