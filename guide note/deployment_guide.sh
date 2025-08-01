# ============================================================================
# 完整的模型部署指南
# ============================================================================

# 1. 准备 NuPlan 环境
# ============================================================================

# 1.1 设置环境变量
export NUPLAN_DATA_ROOT="/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps"
export MODEL_PATH="/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/GNN_Trans_params/best_model.pth"
export PYTHONPATH="${PYTHONPATH}:/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans"

# 1.2 创建目录结构
mkdir -p nuplan/planning/simulation/planner/custom
mkdir -p nuplan/planning/script/config/planner
mkdir -p nuplan/planning/script/config/simulation

# 1.3 复制模型文件到 NuPlan 项目
cp transformer_planner.py nuplan/planning/simulation/planner/custom/
cp transformer_planner.yaml nuplan/planning/script/config/planner/
cp simulation_transformer.yaml nuplan/planning/script/config/simulation/

# 2. NuPlan 集成测试
# ============================================================================

# 2.1 测试模型加载
cd /mnt/local/Xiao_Yi/nuplan-devkit
python -c "
from nuplan.planning.simulation.planner.custom.transformer_planner import TransformerPlanner
planner = TransformerPlanner('$MODEL_PATH')
print('Model loaded successfully!')
"

# 2.2 运行简单仿真测试
python nuplan/planning/script/run_simulation.py \
    +simulation=simulation_transformer \
    planner=transformer_planner \
    scenario_builder=nuplan \
    scenario_filter.scenario_types=[following_lane_with_lead] \
    scenario_filter.num_scenarios_per_type=5 \
    scenario_filter.limit_total_scenarios=5 \
    worker=sequential \
    experiment_name=transformer_test \
    job_name=simple_test \
    verbose=true

# 2.3 运行完整评估
python nuplan/planning/script/run_simulation.py \
    +simulation=simulation_transformer \
    planner=transformer_planner \
    scenario_builder=nuplan \
    scenario_filter=open_loop_boxes \
    worker=sequential \
    experiment_name=transformer_open_loop \
    job_name=open_loop_evaluation \
    verbose=true

# 2.4 评估结果
python nuplan/planning/script/run_metric_aggregator.py \
    experiment=transformer_open_loop \
    metric=planning_default_metrics \
    aggregator_save_path=/tmp/nuplan_metrics \
    verbose=true

# 2.5 可视化结果
python nuplan/planning/script/run_nuboard.py \
    simulation_path=/tmp/exp/transformer_open_loop \
    port=5006

# 3. NuScenes 集成测试
# ============================================================================

# 3.1 设置 NuScenes 环境
export NUSCENES_DATAROOT="/path/to/nuscenes/dataset"
export NUSCENES_VERSION="v1.0-mini"

# 3.2 测试预测功能
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
python nuscenes_integration.py \
    --model_path $MODEL_PATH \
    --dataroot $NUSCENES_DATAROOT \
    --version $NUSCENES_VERSION \
    --output_path predictions_test.json

# 3.3 评估预测结果
python nuscenes_integration.py \
    --model_path $MODEL_PATH \
    --dataroot $NUSCENES_DATAROOT \
    --version $NUSCENES_VERSION \
    --output_path predictions_full.json \
    --evaluate

# 4. 性能基准测试
# ============================================================================

# 4.1 NuPlan 基准测试 - 开环
python nuplan/planning/script/run_simulation.py \
    +simulation=simulation_transformer \
    planner=transformer_planner \
    scenario_builder=nuplan \
    scenario_filter=open_loop_boxes \
    scenario_filter.num_scenarios_per_type=100 \
    worker=sequential \
    experiment_name=transformer_benchmark_open \
    verbose=true

# 4.2 NuPlan 基准测试 - 闭环
python nuplan/planning/script/run_simulation.py \
    +simulation=simulation_transformer \
    planner=transformer_planner \
    scenario_builder=nuplan \
    scenario_filter=closed_loop_nonreactive_agents \
    scenario_filter.num_scenarios_per_type=50 \
    worker=sequential \
    experiment_name=transformer_benchmark_closed \
    verbose=true

# 4.3 与基线方法对比
python nuplan/planning/script/run_simulation.py \
    +simulation=default_simulation \
    planner=simple_planner \
    scenario_builder=nuplan \
    scenario_filter=open_loop_boxes \
    scenario_filter.num_scenarios_per_type=100 \
    worker=sequential \
    experiment_name=simple_planner_baseline \
    verbose=true

# 5. 结果分析和报告
# ============================================================================

# 5.1 生成详细度量报告
python nuplan/planning/script/run_metric_aggregator.py \
    experiment=transformer_benchmark_open \
    metric=planning_default_metrics \
    aggregator_save_path=/tmp/nuplan_detailed_metrics \
    verbose=true

# 5.2 对比分析
python -c "
import pandas as pd
import json

# 读取度量结果
with open('/tmp/nuplan_detailed_metrics/metrics.json', 'r') as f:
    transformer_metrics = json.load(f)

print('Transformer Planner Results:')
print(f'ADE: {transformer_metrics.get(\"ade\", \"N/A\")}')
print(f'FDE: {transformer_metrics.get(\"fde\", \"N/A\")}')
print(f'Collision Rate: {transformer_metrics.get(\"collision_rate\", \"N/A\")}')
"

# 6. 故障排除常见问题
# ============================================================================

# 6.1 检查模型兼容性
python -c "
import torch
model_path = '$MODEL_PATH'
checkpoint = torch.load(model_path, map_location='cpu')
print('Model state dict keys:')
for key in list(checkpoint.keys())[:10]:
    print(f'  {key}')
print(f'Total parameters: {len(checkpoint)}')
"

# 6.2 检查数据路径
ls -la $NUPLAN_DATA_ROOT
ls -la $NUPLAN_MAPS_ROOT

# 6.3 验证配置文件
python -c "
from hydra import compose, initialize
with initialize(config_path='nuplan/planning/script/config/simulation'):
    cfg = compose(config_name='simulation_transformer')
    print('Configuration loaded successfully')
    print(f'Planner: {cfg.planner._target_}')
"

# 7. 生产部署准备
# ============================================================================

# 7.1 创建 Docker 容器（可选）
cat > Dockerfile << EOF
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt
RUN pip install -e .

ENV PYTHONPATH="/app:\${PYTHONPATH}"

CMD ["python", "nuplan_simulation_script.py", "--mode", "all"]
EOF

# 7.2 性能优化
python -c "
import torch
# 启用 TensorRT 优化（如果可用）
if torch.backends.tensorrt.is_available():
    print('TensorRT available for optimization')

# 启用 CUDA 优化
if torch.cuda.is_available():
    print(f'CUDA available with {torch.cuda.device_count()} GPUs')
    print(f'Current device: {torch.cuda.current_device()}')
"

# 7.3 批量测试脚本
for scenario in open_loop_boxes closed_loop_nonreactive_agents closed_loop_reactive_agents; do
    echo "Testing scenario: $scenario"
    python nuplan/planning/script/run_simulation.py \
        +simulation=simulation_transformer \
        planner=transformer_planner \
        scenario_builder=nuplan \
        scenario_filter=$scenario \
        scenario_filter.num_scenarios_per_type=10 \
        worker=sequential \
        experiment_name=transformer_$scenario \
        verbose=false
done

echo "============================================================================"
echo "部署完成！请检查以下内容："
echo "1. 模型在 NuPlan 仿真中的表现"
echo "2. 与基线方法的对比结果"  
echo "3. NuScenes 预测任务的评估指标"
echo "4. 在不同场景类型下的鲁棒性"
echo "============================================================================"
