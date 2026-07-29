# Safe-Sim Integration

**Paper**: Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries (ECCV 2024)  
**Repo**: https://github.com/jxmmy7777/safe-sim

---

## 接入难度评估：高（需较多适配工作）

### 关键障碍

| 障碍 | 细节 | 所需工作 |
|------|------|---------|
| **输入格式不兼容** | Safe-Sim 要求 rasterized HD map（224×224 图像）+ trajectory；我们的 pkl 只有 trajectory，无地图 | 实现 HD map tile → raster 转换模块 |
| **trajdata 依赖** | 使用 NVIDIA 的修改版 `trajdata` fork；nuPlan 支持仅有 config，无实际实现 | 实现 `nuPlan → AgentBatch` 适配器 |
| **闭环设计** | 无 standalone inference；依赖 `rollout_episodes()` 整个仿真环境 | 编写跳过环境循环的 inference wrapper |
| **Checkpoint 域差** | 公开的 checkpoint 仅在 nuScenes 上训练；nuPlan 无公开权重 | 用 nuPlan 数据重新训练或 fine-tune |
| **环境依赖冲突** | `pytorch-lightning==1.9.0` + `python=3.8` + trajdata fork | 独立 conda 环境，与现有项目隔离 |

### 结论

在 nuPlan pkl 数据上直接运行 Safe-Sim 推理**不可行**，需完整重新适配。
论文对比建议：
- **方案 A（推荐）**：直接引用 Safe-Sim 论文中 nuScenes 上的公开数字，注明 domain 不同
- **方案 B**：完成以下适配步骤，在 nuPlan 上复现（工作量 ~1-2 周）

---

## 方案 B 接入步骤

### Step 1：安装环境

```bash
# 独立 conda 环境
conda create -n safe_sim python=3.8
conda activate safe_sim

# Clone 官方 repo
git clone https://github.com/jxmmy7777/safe-sim.git /tmp/safe-sim
cd /tmp/safe-sim

# 安装修改版 trajdata（必须用 fork）
pip install -e trajdata/

# 安装主依赖
pip install -r requirements.txt
```

### Step 2：下载 Checkpoint

```bash
# Hugging Face 上的 nuScenes 预训练 checkpoint（见论文 README）
# nuPlan checkpoint 目前无公开权重
```

### Step 3：地图 Rasterizer（最大工作量）

Safe-Sim 模型输入 `obs_dict["image"]` 是 `[B, C, 224, 224]` 的栅格化地图。
需要：
1. 读取 nuPlan HD map（`maps/*.db` 文件）
2. 以 ego 当前位置为中心，生成 224×224 的地图图像（道路、lane、边界等 channel）
3. 对应文件：`safe_sim/adapter.py` 的 `rasterize_map()` 函数（待实现）

### Step 4：数据适配器

将我们的 pkl 格式转换为 Safe-Sim 所需的 `obs_dict`（见 `adapter.py`）。

### Step 5：Inference Wrapper

绕过 `rollout_episodes()` 闭环，仅执行单次 forward pass，获取生成轨迹。

---

## 文件结构

```
safe_sim/
├── README.md           # 本文件
├── __init__.py
├── adapter.py          # pkl → obs_dict 格式转换（Step 3-4 的核心）
└── inference.py        # 单次 forward pass wrapper（Step 5）
```
