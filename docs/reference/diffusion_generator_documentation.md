# Diffusion 危险场景生成器说明

## 文档定位

本文档面向需要理解 Diffusion 危险场景生成系统整体设计、核心模块和训练目标的开发者。重点内容是模型结构、条件编码方式以及各组件之间的关系。

## 适用范围

- 阅读和维护 Diffusion 生成相关代码
- 理解 `SceneConditionEncoder`、`DangerSceneDiffusion` 和训练损失
- 对接危险评分模块和生成模块

## 相关文件

- `models/scene_condition_encoder.py`
- `models/danger_diffusion.py`
- `utils/danger_score_torch.py`
- `train_diffusion_generator.py`
- `generate_danger_scenes.py`

## 推荐阅读顺序

1. 先阅读本文档，理解模型结构和训练目标
2. 再阅读 `danger_score_documentation.md`，理解风险信号来源
3. 最后阅读 `diffusion_usage_guide.md`，按流程运行训练和生成

## 概述

本文档描述了基于扩散模型（Diffusion Model）的危险场景生成系统，用于生成比 GT（Ground Truth）轨迹更危险的驾驶场景数据。

### 目标

- **输入**：历史轨迹 + SVO角度 + danger_target
- **输出**：生成的未来轨迹（比GT更危险）
- **用途**：数据增强、风险边界测试、极端场景模拟

### 训练目标

1. **重建GT轨迹**：学习基本的轨迹生成能力
2. **Danger超越**：生成轨迹的danger > GT的danger

---

## 系统架构

### 整体结构

```
┌──────────────────────────────────────────────────────────────┐
│              Diffusion危险场景生成器                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  输入条件                                                     │
│  ├── 历史轨迹编码 (来自GNN+Trans encoder, 384维)             │
│  ├── SVO角度条件 (控制行为倾向)                               │
│  │   - 竞争型(-45°) → 更危险                                  │
│  │   - 合作型(45°) → 更安全                                   │
│  └── danger_target (目标危险等级)                             │
│      - danger_target > GT_danger → 生成更危险轨迹             │
│                                                              │
│  条件编码器                                                   │
│  ├── SceneConditionEncoder                                   │
│  ├── 输入: scene_feat(384) + svo_angle + danger              │
│  └── 输出: 256维条件向量                                      │
│                                                              │
│  Diffusion生成                                                │
│  ├── DangerSceneDiffusion (DDPM)                             │
│  ├── 100步去噪过程                                            │
│  ├── Transformer-based Denoiser                              │
│  └── 输出: 轨迹增量序列 (120维 = 60步 × dx,dy)               │
│                                                              │
│  评估                                                         │
│  ├── compute_danger_score_torch (生成轨迹)                   │
│  ├── 与GT对比                                                 │
│  └── 保存为新的训练样本                                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 文件结构

### 新增文件

| 文件路径 | 功能 |
|----------|------|
| `utils/danger_score_torch.py` | PyTorch版本的danger计算，支持GPU和自动梯度 |
| `models/scene_condition_encoder.py` | 条件编码器（场景+SVO+Danger） |
| `models/danger_diffusion.py` | Diffusion生成器（DDPM） |
| `train_diffusion_generator.py` | 训练脚本 |
| `generate_danger_scenes.py` | 生成脚本 |

### 修改文件

| 文件路径 | 修改内容 |
|----------|----------|
| `models/transformer_predictor.py` | 添加 `get_encoder_features()` 方法 |

---

## 核心组件详解

### 1. danger_score_torch.py

PyTorch版本的danger计算模块，用于Diffusion训练中的loss计算。

#### 主要函数

```python
# 计算单段轨迹的危险指数
compute_single_trajectory_danger_torch(ego_traj, agent_traj, config)

# 主危险评分函数
compute_danger_score_torch(ego_history, ego_future, agent_history, agent_future)

# 从轨迹增量恢复完整轨迹
recover_trajectory_from_increment(increments, last_pos, last_heading, last_vx, last_vy)

# 计算轨迹平滑度loss
trajectory_smoothness_loss_torch(increments)
```

#### DangerExceedLoss

核心训练Loss类：

```python
class DangerExceedLoss(nn.Module):
    """
    Loss组成：
    1. reconstruction_loss: MSE(pred_traj, gt_traj)
    2. danger_exceed_loss: ReLU(gt_danger - pred_danger + margin)
    3. smoothness_loss: 加速度平方和
    """
    
    def forward(self, pred_increments, gt_increments, ...):
        # 重建Loss
        recon_loss = F.mse_loss(pred_increments, gt_increments)
        
        # Danger超越Loss
        danger_exceed_loss = F.relu(gt_danger + margin - pred_danger)
        
        # 平滑Loss
        smoothness_loss = trajectory_smoothness_loss_torch(pred_increments)
        
        return total_loss, loss_dict
```

### 2. scene_condition_encoder.py

将多种条件融合为统一的256维条件向量。

#### SceneConditionEncoder

```python
class SceneConditionEncoder(nn.Module):
    """
    输入：
    - scene_feat: [batch, 384] 来自GNN+Trans encoder
    - svo_angle: [batch, 1] SVO角度（度）
    - inherent_danger: [batch, 1] 历史风险
    - danger_target: [batch, 1] 目标风险
    
    输出：
    - condition: [batch, 256] 条件向量
    """
    
    def __init__(self, config):
        # 场景编码器：384 -> 256
        self.scene_encoder = nn.Sequential(...)
        
        # SVO角度编码器：1 -> 64
        self.svo_encoder = nn.Sequential(...)
        
        # Danger编码器：2 -> 64
        self.danger_encoder = nn.Sequential(...)
        
        # 条件融合层
        self.condition_fusion = nn.Linear(384, 256)
```

#### SimpleSceneEncoder

当GNN+Trans encoder不可用时，直接从历史轨迹提取特征：

```python
class SimpleSceneEncoder(nn.Module):
    """
    输入：
    - ego_history: [batch, T_hist, 5]
    - agent_history: [batch, T_hist, 5]
    
    输出：
    - scene_feat: [batch, 384]
    """
```

### 3. danger_diffusion.py

基于DDPM的去噪扩散模型。

#### DiffusionConfig

```python
@dataclass
class DiffusionConfig:
    condition_dim: int = 256        # 条件向量维度
    trajectory_dim: int = 120       # 轨迹维度 (60步 × 2)
    num_diffusion_steps: int = 100  # 去噪步数
    beta_schedule: str = 'linear'   # beta调度类型
    denoiser_num_layers: int = 4    # Transformer层数
    condition_injection: str = 'cross_attention'  # 条件注入方式
```

#### DangerSceneDiffusion

```python
class DangerSceneDiffusion(nn.Module):
    """
    训练流程：
    1. 随机采样时间步 t
    2. 添加噪声：x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
    3. 预测噪声：noise_pred = denoiser(x_t, condition, t)
    4. 计算loss：MSE(noise_pred, noise)
    
    生成流程：
    1. 初始化纯噪声 x_T
    2. 逐步去噪：从T到1
    3. 得到 x_0（生成的轨迹增量）
    """
    
    def add_noise(self, x_0, t):
        """添加噪声（前向扩散）"""
        
    def denoise_step(self, x_t, condition, t):
        """单步去噪"""
        
    def generate(self, condition, num_samples=1):
        """从噪声生成轨迹"""
```

#### TrajectoryDenoiser

Transformer-based去噪网络：

```python
class TrajectoryDenoiser(nn.Module):
    """
    支持三种条件注入方式：
    1. cross_attention: 使用cross-attention融合条件
    2. concat: 拼接条件到输入
    3. addition: 直接加法融合
    """
```

---

## SVO角度与行为倾向

### SVO角度范围

| 角度范围 | 类型 | 行为倾向 | 生成效果 |
|----------|------|----------|----------|
| -45° ~ -15° | competitive | 竞争型 | 更危险轨迹 |
| -15° ~ 22.5° | individualistic | 个人主义 | 中等风险 |
| 22.5° ~ 57.5° | prosocial | 合作型 | 较安全轨迹 |
| 57.5° ~ 135° | altruistic | 利他型 | 最安全轨迹 |

### 单位圆模型

```
      y (集体权重)
      ↑
      │    altruistic (135°)
      │      ●
      │
      │  prosocial (45°)
      │    ●
      │
──────┼──────────────→ x (个人权重)
      │
      │  individualistic (0°)
      │    ●
      │
      │  competitive (-45°)
      │      ●
```

在单位圆上：
- x = cos(θ)：个人权重
- y = sin(θ)：集体权重
- θ = arctan2(y, x)：SVO角度

---

## 数据流

### 训练数据流

```
danger_pair_*.pkl
    │
    ├── 提取数据
    │   ├── ego_history, agent_history (历史轨迹)
    │   ├── ego_future_gt, agent_future_gt (GT未来轨迹)
    │   ├── svo_angle (计算得到)
    │   ├── inherent_danger (历史风险)
    │   ├── gt_danger (GT未来风险)
    │   └── danger_target = gt_danger + margin
    │
    ├── SceneEncoder
    │   └── scene_feat = encoder(ego_history, agent_history)
    │
    ├── ConditionEncoder
    │   └── condition = encoder(scene_feat, svo, danger)
    │
    ├── Diffusion训练
    │   ├── add_noise(gt_increments, t)
    │   ├── denoiser(noisy_traj, condition, t)
    │   └── compute_loss(pred, gt, danger)
    │
    └── 优化
        └── backward(loss)
```

### 生成数据流

```
推理生成新危险场景:

1. 加载历史轨迹数据
   ├── ego_history, agent_history
   ├── 设定svo_angle（或随机）
   ├── 计算inherent_danger
   └── 设定danger_target（高于GT）

2. SceneEncoder编码
   scene_feat = encoder(ego_history, agent_history)

3. ConditionEncoder编码
   condition = encoder(scene_feat, svo, danger)

4. Diffusion生成
   pred_increments = diffusion.generate(condition)

5. 恢复轨迹
   pred_trajectory = recover_from_increment(pred_increments)

6. 评估生成结果
   pred_danger = compute_danger_score(pred_trajectory)
   assert pred_danger > gt_danger

7. 保存为新训练样本
   save_generated_sample(pred_trajectory, ...)
```

---

## 使用方法

### 训练

```bash
# 基本训练
python train_diffusion_generator.py \
    --data_dir danger_pair_scenarios \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4

# 调试模式（限制样本数）
python train_diffusion_generator.py \
    --data_dir danger_pair_scenarios \
    --epochs 10 \
    --max_samples 50
```

### 生成

```bash
# 批量生成
python generate_danger_scenes.py \
    --model_path diffusion_training/best_model.pt \
    --data_dir danger_pair_scenarios \
    --output_dir generated_danger_scenes \
    --num_samples 3

# 单场景测试
python generate_danger_scenes.py \
    --mode single \
    --model_path diffusion_training/best_model.pt \
    --pkl_path danger_pair_scenarios/danger_pair_features_xxx.pkl \
    --svo_angles -30 0 45 90 \
    --danger_margin 3.0
```

---

## 验证方法

### 1. 训练验证

观察训练日志中的loss下降趋势：

```
Epoch 1: loss = 0.5234
Epoch 10: loss = 0.2156
Epoch 50: loss = 0.0843
Epoch 100: loss = 0.0234
```

关键指标：
- `danger_margin_met`: danger超越成功率

### 2. 生成验证

检查生成的轨迹是否比GT更危险：

```python
# 加载训练好的模型
generator = DangerSceneGenerator('best_model.pt')

# 生成危险场景
result = generator.generate_from_pkl(
    pkl_path,
    svo_angle_override=-30,  # 竞争型
    danger_target_override=gt_danger + 3.0
)

# 检查danger是否超越
print(f"Generated danger: {result['generated_danger']:.2f}")
print(f"GT danger: {result['gt_danger']:.2f}")
print(f"Exceeded: {result['danger_exceeded']}")
```

### 3. 数据增强效果

将生成的危险场景与原始数据混合训练：

```python
# 原始数据：N个场景
original_data = load_danger_pair_data('danger_pair_scenarios')

# 生成数据：M个更危险场景
generated_data = load_generated_data('generated_danger_scenes')

# 混合训练
mixed_data = original_data + generated_data
train_prediction_model(mixed_data)
```

---

## 配置参数说明

### DiffusionConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| condition_dim | 256 | 条件向量维度 |
| trajectory_dim | 120 | 轨迹维度（60步×2） |
| num_diffusion_steps | 100 | 去噪步数 |
| beta_start | 1e-4 | beta起始值 |
| beta_end | 0.02 | beta结束值 |
| beta_schedule | 'linear' | beta调度类型 |
| denoiser_num_layers | 4 | Transformer层数 |
| denoiser_num_heads | 8 | Transformer头数 |

### ConditionEncoderConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| scene_feat_dim | 384 | 场景特征维度 |
| condition_dim | 256 | 输出条件维度 |
| svo_embed_dim | 64 | SVO嵌入维度 |
| danger_embed_dim | 64 | Danger嵌入维度 |

### DangerExceedLoss权重

| 参数 | 默认值 | 说明 |
|------|--------|------|
| recon_weight | 1.0 | 重建Loss权重 |
| danger_weight | 2.0 | Danger超越Loss权重 |
| smoothness_weight | 0.3 | 平滑Loss权重 |
| danger_margin | 0.5 | Danger超越阈值 |

---

## 注意事项

### 1. 数据格式要求

输入的pkl文件需包含以下字段：
- `ego_history`: (T_hist, 5) - [x, y, heading, vx, vy]
- `key_agent_history`: (T_hist, 5)
- `ego_future_gt`: (T_future, 2) 或 (1, T_future, 2)
- `key_agent_future`: (T_future, 3) 或 None

### 2. SVO角度范围

- 有效范围：[-45°, 135°]
- 超出范围会被clamp

### 3. Danger值尺度

- 建议danger值在0-50范围内
- 归一化处理会将值映射到[0, 1]

### 4. GPU内存

- batch_size=8 需约4GB显存
- 可根据GPU调整batch_size

---

## 与现有系统集成

### GNN+Transformer Encoder

修改后的 `transformer_predictor.py` 提供新方法：

```python
class TransformerPredictor(nn.Module):
    def get_encoder_features(self, data) -> torch.Tensor:
        """
        提取encoder特征用于Diffusion
        
        Returns:
            [batch, 384] encoder特征向量
            = ego_feat(128) + context_feat(128) + gnn_feat(128)
        """
```

### Danger Score系统

与现有的 `danger_score.py` 系统兼容：
- numpy版本：用于预处理和评估
- PyTorch版本：用于训练loss计算

---

## 未来扩展

### 1. 多Agent生成

`MultiAgentConditionEncoder` 支持多Agent条件编码：

```python
encoder = MultiAgentConditionEncoder(max_agents=5)
condition = encoder(scene_feat, svo_angles, dangers, agent_mask)
```

### 2. 自适应Danger目标

可根据历史风险自动调整目标：

```python
danger_target = inherent_danger * 1.5  # 基于历史风险
```

### 3. 条件引导采样

使用classifier-free guidance增强条件控制：

```python
# 无条件和有条件生成混合
trajectory = (1 + guidance_scale) * conditional_gen - guidance_scale * unconditional_gen
```

---

## 参考文献

1. **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
2. **Improved DDPM**: Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021
3. **SVO Theory**: Murphy & Ackermann, "Social Value Orientation", 2014

---

## 更新日志

- 2026-04-10: 初版完成，包含完整的训练和生成系统

---

## 作者

Claude Code 辅助生成
