# 危险评分模块说明

## 文档定位

本文档面向需要理解 `danger_score.py` 设计原理、评分逻辑和接口结构的开发者。重点内容是危险评分模块如何为 Diffusion 危险场景生成提供可解释、可扩展的风险信号。

## 适用范围

- 阅读和维护 `utils/danger_score.py`
- 理解 TTC、距离、碰撞、车道关系等评分来源
- 对接 Diffusion 训练中的 danger 指标

## 相关文件

- `utils/danger_score.py`
- `utils/danger_score_torch.py`
- `docs/reference/diffusion_generator_documentation.md`
- `docs/diffusion_usage_guide.md`

## 推荐阅读顺序

1. 先看本文档，理解危险评分的组成和设计目标
2. 再看 `diffusion_generator_documentation.md`，理解评分如何接入生成模型
3. 最后看 `diffusion_usage_guide.md`，按流程运行训练和生成

## 概述

本文档详细描述了 `danger_score.py` 模块的设计原理、实现细节和使用方法。该模块用于计算自动驾驶场景中 ego 车辆与周围车辆轨迹对的危险评分，为 Diffusion 模型生成危险轨迹提供引导信号。

---

## 一、整体设计思路

### 1.1 输入输出

```
输入:
├── ego_history: np.ndarray, shape=(T_hist, 5)
│   └── 过去轨迹 [x, y, heading, vx, vy]
├── ego_future: np.ndarray, shape=(T_future, 5)
│   └── 未来轨迹（待评估）
├── agent_history: np.ndarray, shape=(T_hist, 5)
└── agent_future: np.ndarray, shape=(T_future, 5)

输出:
├── total_score: float
│   └── 综合危险评分
├── component_scores: Dict[str, float]
│   └── 各组件评分详情
├── ttc_info: Dict
│   └── TTC分析详情
├── collision_info: Dict
│   └── 碰撞检测详情
├── lane_info: Dict
│   └── 车道关系信息
└── danger_level: str
    └── 危险等级分类
```

### 1.2 核心思想

1. **多维度危险评估**：危险是TTC、距离、速度、车道关系等多因素的综合体现
2. **车道感知**：不同车道关系的车辆，危险评估逻辑不同
3. **可微分设计**：评分函数可微分，支持diffusion模型的梯度引导

### 1.3 模块结构

```
danger_score.py
├── 配置类
│   └── DangerScoreConfig
├── 车道感知函数
│   ├── get_lane_index()
│   ├── get_lane_relationship()
│   └── detect_lane_change_intent()
├── TTC计算函数
│   ├── compute_ttc_1d()
│   ├── compute_relative_state()
│   └── compute_ttc_at_timestep()
├── 危险评分组件
│   ├── ttc_danger_score()
│   ├── distance_danger_score()
│   ├── collision_detection()
│   ├── relative_velocity_danger_score()
│   ├── acceleration_danger_score()
│   ├── lane_change_conflict_score()
│   └── compute_lane_aware_ttc_danger()
├── 主函数
│   ├── compute_danger_score()
│   └── classify_danger_level()
└── Diffusion工具函数
    ├── compute_danger_gradient()
    └── batch_compute_danger_scores()
```

---

## 二、配置参数详解

### 2.1 TTC阈值

```python
ttc_critical: float = 1.0   # 临界TTC（秒）
ttc_warning: float = 3.0    # 警告TTC（秒）
ttc_max: float = 10.0       # 最大考虑的TTC（秒）
```

| 阈值 | 含义 | 评分策略 |
|------|------|---------|
| ttc_critical | 1秒内碰撞风险极高 | 评分加倍 |
| ttc_warning | 3秒内需要关注 | 正常评分 |
| ttc_max | 超过10秒不考虑 | 评分衰减为0.1x |

### 2.2 距离阈值

```python
min_safe_distance: float = 5.0      # 最小安全距离（米）
collision_distance: float = 2.0     # 认定为碰撞的距离（米）
near_miss_distance: float = 3.0     # 近距离阈值（米）
```

```
距离范围判定:

  0m        2m         3m         5m
   |---------|----------|----------|----------→ 距离
   collision  near_miss   safe
   
   评分高     评分中等    评分低/零
```

### 2.3 车道参数

```python
lane_width: float = 3.5              # 标准车道宽度（米）
same_lane_threshold: float = 1.75    # 同车道判定阈值（米）
adjacent_lane_threshold: float = 5.25  # 相邻车道判定阈值（米）
lane_change_vy_threshold: float = 0.5  # 变道横向速度阈值（米/秒）
```

**阈值计算依据**：
- `same_lane_threshold = lane_width / 2 = 1.75m`
- `adjacent_lane_threshold = lane_width * 1.5 = 5.25m`

### 2.4 权重参数

```python
weight_ttc_longitudinal: float = 1.0
weight_ttc_lateral: float = 0.8
weight_ttc_same_lane: float = 1.5
weight_min_distance: float = 1.5
weight_collision: float = 10.0
weight_near_miss: float = 3.0
weight_relative_velocity: float = 0.5
weight_acceleration: float = 0.3
weight_lane_change_conflict: float = 2.0
```

**权重设计原则**：
- 碰撞权重最高（10.0），因为碰撞是最危险的情况
- 同车道TTC权重加成（1.5x），因为同车道追尾风险最高
- 变道冲突权重（2.0），因为变道碰撞难以预测

---

## 三、车道感知系统

### 3.1 坐标系定义

```
        Y (左)
        ↑
        |
    车道1 | 车道0 | 车道-1
   (Y>0) | (Y≈0) | (Y<0)
        |       |
        |  ego  +------→ X (前)
        |       |
        |       |
        ↓
        Y (右)

约定：
- ego所在车道为车道0，Y坐标约等于0
- 左侧车道Y > 0
- 右侧车道Y < 0
```

### 3.2 车道索引计算

```python
def get_lane_index(y_position: float, lane_width: float = 3.5) -> int:
    return int(round(y_position / lane_width))
```

**示例**：

| Y坐标 (米) | 车道索引 | 相对位置 |
|-----------|---------|---------|
| 0.0 | 0 | 同车道 |
| 3.5 | 1 | 左侧第一车道 |
| -3.5 | -1 | 右侧第一车道 |
| 7.0 | 2 | 左侧第二车道 |

### 3.3 车道关系判定

```python
def get_lane_relationship(ego_y, agent_y, config) -> str:
    rel_y = agent_y - ego_y
    abs_rel_y = abs(rel_y)

    if abs_rel_y < 1.75:
        return 'same_lane'
    elif abs_rel_y < 5.25:
        return 'adjacent_left' if rel_y > 0 else 'adjacent_right'
    else:
        return 'far_left' if rel_y > 0 else 'far_right'
```

**判定逻辑图**：

```
相对Y坐标 (米)
     10.5  |  far_left
      5.25  ├───────────────── 分界线
      3.5   |  adjacent_left
      1.75  ├───────────────── 分界线
      0.0   |  same_lane      ← ego位置
     -1.75  ├───────────────── 分界线
     -3.5   |  adjacent_right
     -5.25  ├───────────────── 分界线
    -10.5   |  far_right
```

### 3.4 变道意图检测

```python
def detect_lane_change_intent(agent_vy, config) -> Tuple[bool, str]:
    abs_vy = abs(agent_vy)

    if abs_vy < 0.5:
        return False, 'none'

    return True, 'left' if agent_vy > 0 else 'right'
```

**物理原理**：
- 车辆在车道内稳定行驶时，横向速度接近0
- 横向速度显著（|vy| > 0.5 m/s）表明正在变道
- 速度符号决定变道方向

**示例**：

| 横向速度 vy (m/s) | 是否变道 | 变道方向 |
|------------------|---------|---------|
| 0.0 | 否 | - |
| 0.3 | 否 | - |
| 0.8 | 是 | 左 |
| -1.0 | 是 | 右 |

---

## 四、TTC计算详解

### 4.1 一维TTC公式

$$TTC = \begin{cases}
\frac{d}{-v_{rel}} & \text{if } v_{rel} < 0 \\
\infty & \text{if } v_{rel} \geq 0
\end{cases}$$

其中：
- $d$ = 两车间距离（米）
- $v_{rel}$ = 相对速度（米/秒），负值表示接近

```python
def compute_ttc_1d(distance, relative_velocity, epsilon=1e-6) -> float:
    if relative_velocity >= 0:
        return np.inf  # 正在远离
    if distance <= 0:
        return 0.0    # 已经重叠
    return distance / (-relative_velocity + epsilon)
```

### 4.2 纵向TTC计算

**场景分析**：

```
情况1: Agent在前方 (rel_x > 0)

    ego ────→     ←──── agent
          rel_x
    
    如果 ego速度 > agent速度 (rel_vx < 0)，则正在接近
    lon_ttc = rel_x / (-rel_vx)

情况2: Agent在后方 (rel_x < 0)

    agent ────→     ←──── ego
              rel_x
    
    如果 agent速度 > ego速度 (rel_vx > 0)，则正在接近
    lon_ttc = |rel_x| / rel_vx
```

### 4.3 横向TTC计算

```python
if rel_state['rel_y'] > 0:
    # Agent在左侧
    lat_ttc = compute_ttc_1d(rel_state['rel_y'], rel_state['rel_vy'])
else:
    # Agent在右侧
    lat_ttc = compute_ttc_1d(abs(rel_state['rel_y']), -rel_state['rel_vy'])
```

### 4.4 相对状态计算

```python
def compute_relative_state(ego_state, agent_state) -> Dict:
    # 相对位置
    rel_x = agent_x - ego_x   # 正值 = agent在前
    rel_y = agent_y - ego_y   # 正值 = agent在左

    # 相对速度
    rel_vx = agent_vx - ego_vx
    rel_vy = agent_vy - ego_vy

    # 距离
    euclidean_dist = np.sqrt(rel_x**2 + rel_y**2)
```

---

## 五、危险评分组件

### 5.1 组件列表

| 组件名 | 计算依据 | 权重 |
|-------|---------|------|
| ttc_score | TTC值 | 1.0 |
| lane_aware_ttc_score | 车道感知TTC | 动态 |
| distance_score | 最小距离 | 1.5 |
| collision_score | 碰撞检测 | 10.0 |
| near_miss_score | 近距离 | 3.0 |
| velocity_score | 相对速度 | 0.5 |
| acceleration_score | 加速度 | 0.3 |
| lane_change_conflict_score | 变道冲突 | 2.0 |

### 5.2 TTC危险评分

```python
def ttc_danger_score(lon_ttc, lat_ttc, config) -> float:
    score = 0.0

    # 纵向TTC
    if lon_ttc < ttc_critical:
        score += weight * (ttc_critical - lon_ttc) * 2.0  # 加倍
    elif lon_ttc < ttc_warning:
        score += weight * (ttc_warning - lon_ttc)
    elif lon_ttc < ttc_max:
        score += weight * (ttc_max - lon_ttc) * 0.1  # 衰减

    # 横向TTC（类似逻辑）
    ...

    return score
```

**评分曲线**：

```
评分
  ↑
  │     ╱──────────╲
  │    ╱            ╲
  │   ╱              ╲
  │  ╱                ╲____________
  │ ╱
  └────────────────────────────────→ TTC
    0    1s    3s    10s
    critical warning max
```

### 5.3 车道感知TTC评分

这是本模块的核心创新，根据车道关系采用不同评分策略：

```python
def compute_lane_aware_ttc_danger(lon_ttc, lat_ttc, lane_rel, lane_intent, config):
    if lane_rel == 'same_lane':
        # 同车道：纵向TTC是主要危险
        # 权重加成1.5x
        return longitudinal_danger(lon_ttc) * 1.5

    elif lane_rel in ['adjacent_left', 'adjacent_right']:
        if has_lane_change_intent:
            # 有变道意图：同时考虑纵向和横向TTC
            return longitudinal_danger(lon_ttc) + lateral_danger(lat_ttc) * 2.0
        else:
            # 无变道意图：不危险
            return 0.0

    else:  # far_left, far_right
        # 远端车道：危险很低
        return longitudinal_danger(lon_ttc) * 0.3
```

**评分策略表**：

| 车道关系 | 变道意图 | 纵向TTC权重 | 横向TTC权重 | 说明 |
|---------|---------|------------|------------|------|
| same_lane | - | 1.5 | 0.8 | 同车道追尾风险最高 |
| adjacent | 无 | 0 | 0 | 相邻车道无干扰 |
| adjacent | 有 | 1.0 | 2.0 | 变道冲突风险 |
| far | - | 0.3 | 0.2 | 距离较远风险低 |

### 5.4 碰撞检测

```python
def collision_detection(ego_traj, agent_traj, config):
    for t in range(len(ego_traj)):
        # 欧氏距离
        dist = sqrt((ego_x - agent_x)² + (ego_y - agent_y)²)

        # 碰撞阈值（简化模型）
        # 考虑两车尺寸
        threshold = (ego_length + agent_length)/4 + (ego_width + agent_width)/4
        # = (4.5 + 4.0)/4 + (2.0 + 2.0)/4 = 3.125米

        if dist < threshold:
            has_collision = True
            collision_timestep = t

        min_distance = min(min_distance, dist)

    return has_collision, min_distance, collision_timestep
```

### 5.5 变道冲突检测

```python
def lane_change_conflict_score(ego_traj, agent_traj, config):
    for t in range(len(ego_traj)):
        ego_intent, ego_dir = detect_lane_change_intent(ego_vy)
        agent_intent, agent_dir = detect_lane_change_intent(agent_vy)

        # 两车都有变道意图，且方向相反（向同一车道变道）
        if ego_intent and agent_intent and ego_dir != agent_dir:
            dist = abs(ego_y - agent_y)
            if dist < adjacent_lane_threshold:
                score += weight_lane_change_conflict

    return score
```

**图示**：

```
变道冲突场景:

    车道1    |  车道0  |  车道-1
            |         |
    agent → | ← ego   |
            |         |
            ↓ 变道冲突 ↓
```

---

## 六、主函数流程

### 6.1 计算流程图

```
                    ┌─────────────────┐
                    │     输入数据     │
                    │ ego/agent轨迹   │
                    └────────┬────────┘
                             ↓
              ┌──────────────┴──────────────┐
              │       遍历未来轨迹每一帧      │
              └──────────────┬──────────────┘
                             ↓
    ┌────────────────────────┼────────────────────────┐
    ↓                        ↓                        ↓
┌─────────┐            ┌─────────┐            ┌─────────┐
│ TTC计算 │            │车道判定 │            │变道检测 │
└────┬────┘            └────┬────┘            └────┬────┘
     └───────────────────────┼───────────────────────┘
                             ↓
                    ┌─────────────────┐
                    │   TTC危险评分    │
                    │ （车道感知版本） │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    碰撞检测      │
                    └────────┬────────┘
                             ↓
              ┌──────────────┴──────────────┐
              │      计算其他组件评分        │
              │ velocity/acceleration/...  │
              └──────────────┬──────────────┘
                             ↓
                    ┌─────────────────┐
                    │   加权求和总分   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   危险等级分类   │
                    └─────────────────┘
```

### 6.2 代码结构

```python
def compute_danger_score(ego_history, ego_future, agent_history, agent_future, config, dt, verbose):
    # 1. 初始化
    component_scores = {各组件: 0.0}
    ttc_info = {TTC详情}
    lane_info = {车道信息}

    # 2. 遍历未来轨迹
    for t in range(len(ego_future)):
        # 计算该帧的TTC和车道信息
        ttc_result = compute_ttc_at_timestep(...)

        # 累积TTC评分
        component_scores['ttc_score'] += ttc_danger_score(...)
        component_scores['lane_aware_ttc_score'] += compute_lane_aware_ttc_danger(...)

        # 统计车道关系
        lane_info['lane_relationships'].append(ttc_result['lane_relationship'])

    # 3. 碰撞检测
    has_collision, min_distance, collision_t = collision_detection(...)

    # 4. 计算其他组件评分
    component_scores['velocity_score'] = relative_velocity_danger_score(...)
    component_scores['acceleration_score'] = acceleration_danger_score(...)
    component_scores['lane_change_conflict_score'] = lane_change_conflict_score(...)

    # 5. 计算总分
    total_score = sum(component_scores.values())

    # 6. 危险等级分类
    danger_level = classify_danger_level(total_score)

    return {
        'total_score': total_score,
        'component_scores': component_scores,
        'ttc_info': ttc_info,
        'collision_info': collision_info,
        'lane_info': lane_info,
        'danger_level': danger_level
    }
```

### 6.3 危险等级分类

```python
def classify_danger_level(score, config) -> str:
    if score >= 50:    # weight_collision * 5
        return 'critical'
    elif score >= 6:   # weight_near_miss * 2
        return 'high'
    elif score >= 1:   # weight_ttc_longitudinal
        return 'medium'
    elif score > 0:
        return 'low'
    else:
        return 'safe'
```

---

## 七、Diffusion引导功能

### 7.1 梯度计算原理

用于classifier guidance，在diffusion采样时引导轨迹向"更危险"方向生成。

**数学原理**：

$$x_{t-1} = x_t - \epsilon_\theta(x_t, t) + \lambda \cdot \nabla_{x_t} D(x_t)$$

其中：
- $\epsilon_\theta$ 是噪声预测网络
- $D(x_t)$ 是危险评分函数
- $\lambda$ 是引导强度
- $\nabla_{x_t} D(x_t)$ 是危险评分对轨迹的梯度

### 7.2 数值梯度实现

```python
def compute_danger_gradient(ego_future, agent_future, ego_history, agent_history, config, epsilon=0.1):
    # 基础评分
    base_score = compute_danger_score(ego_history, ego_future, agent_history, agent_future)

    # 对ego轨迹每个位置计算梯度
    for t in range(len(ego_future)):
        for dim in range(2):  # x, y
            # 前向差分
            ego_perturbed = ego_future.copy()
            ego_perturbed[t, dim] += epsilon

            new_score = compute_danger_score(ego_history, ego_perturbed, ...)

            gradient[t, dim] = (new_score - base_score) / epsilon

    return ego_gradient, agent_gradient
```

### 7.3 批量计算

```python
def batch_compute_danger_scores(ego_futures, agent_futures, ego_history, agent_history, config):
    """
    用于reward-weighted regression:
    1. 生成多条候选轨迹
    2. 计算每条轨迹的危险评分
    3. 用高分轨迹加权更新模型
    """
    scores = np.zeros(batch_size)
    for i in range(batch_size):
        scores[i] = compute_danger_score(..., ego_futures[i], agent_futures[i], ...)
    return scores
```

---

## 八、测试场景分析

### 8.1 场景1: 同车道接近

**设置**：
```
Ego: 前进 10 m/s
Agent: 前进 8 m/s（在前方20米，同车道）
```

**结果**：
```
纵向TTC: 7.00秒
横向TTC: inf
危险评分: 0.38
危险等级: LOW
车道关系: same_lane
```

**分析**：
- TTC较大（7秒），危险较低
- 同车道，主要关注纵向TTC
- 评分主要来自lane_aware_ttc_score

### 8.2 场景2: 碰撞轨迹

**设置**：
```
Agent速度更慢，最终距离只有0.17米
```

**结果**：
```
危险评分: 212.49
危险等级: CRITICAL
碰撞检测: True
最小距离: 0.17m
碰撞时间步: 20
```

**分析**：
- 碰撞触发collision_score（100分）
- min_distance触发distance_score（100分）
- 纵向TTC接近0触发TTC高分

### 8.3 场景3: 相邻车道无变道

**设置**：
```
Agent在左侧车道（Y=3.5米）
横向速度 = 0（无变道意图）
```

**结果**：
```
危险评分: 0.00
危险等级: SAFE
车道关系: adjacent_left
```

**分析**：
- 相邻车道，无变道意图
- lane_aware_ttc_score = 0（正确行为）
- 不同车道的车辆互不影响

### 8.4 场景4: 变道冲突

**设置**：
```
Agent从左侧车道（Y=3.5米）向右变道
横向速度 = -0.8 m/s
```

**结果**：
```
横向TTC: 0.11秒
危险评分: 0.67
危险等级: LOW
主要车道关系: same_lane
```

**分析**：
- 检测到变道意图
- 横向TTC很小（0.11秒）
- Agent逐渐进入ego车道

---

## 九、使用示例

### 9.1 基本使用

```python
from danger_score import compute_danger_score, DangerScoreConfig

# 准备数据
ego_history = np.array(...)  # (41, 5)
ego_future = np.array(...)   # (30, 5)
agent_history = np.array(...) # (41, 5)
agent_future = np.array(...)  # (30, 5)

# 计算危险评分
config = DangerScoreConfig()
result = compute_danger_score(
    ego_history, ego_future,
    agent_history, agent_future,
    config=config,
    dt=0.1,
    verbose=True
)

print(f"危险评分: {result['total_score']}")
print(f"危险等级: {result['danger_level']}")
```

### 9.2 自定义配置

```python
# 调整阈值和权重
config = DangerScoreConfig(
    ttc_critical=0.5,           # 更严格的临界TTC
    lane_change_vy_threshold=0.3,  # 更敏感的变道检测
    weight_collision=20.0,      # 更高的碰撞权重
    weight_ttc_same_lane=2.0    # 更高的同车道TTC权重
)
```

### 9.3 Diffusion引导

```python
from danger_score import compute_danger_gradient

# 在diffusion采样循环中
for t in reversed(range(T)):
    # 预测噪声
    noise_pred = model(x_t, t, condition)

    # 去噪
    x_t_minus_1 = denoise_step(x_t, noise_pred, t)

    # 计算危险梯度
    ego_grad, agent_grad = compute_danger_gradient(
        x_t_minus_1[:30],    # ego future
        x_t_minus_1[30:],    # agent future
        ego_history,
        agent_history
    )

    # 引导向更危险方向
    x_t_minus_1[:30, :, :2] += guidance_scale * ego_grad
    x_t_minus_1[30:, :, :2] += guidance_scale * agent_grad
```

---

## 十、注意事项与局限

### 10.1 当前局限

1. **简化的碰撞检测**：使用AABB而非OBB，未考虑车辆朝向
2. **固定的车道宽度**：假设车道宽度为3.5米，实际可能有变化
3. **无地图集成**：未使用NuPlan地图API确定车道归属
4. **单agent评估**：当前只评估一个agent对的危险

### 10.2 改进方向

1. **集成地图API**：使用NuPlan地图确定精确的车道归属
2. **多agent联合评估**：考虑多个agent的联合危险
3. **动态阈值**：根据场景类型自动调整阈值
4. **神经网络评分**：用学习的方法替代手工设计的评分函数

### 10.3 性能考虑

- 单次评分计算约1-5ms（取决于轨迹长度）
- 梯度计算需要额外2*T*D次评分计算
- 批量计算时建议使用并行化

---

## 十一、参考文献

1. TTC计算: Hayward, J.C. (1972). "Near miss determination through use of a scale of danger"
2. Diffusion Guidance: Dhariwal, P. & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis"
3. NuPlan文档: https://nuplan-devkit.readthedocs.io/

---

## 附录A: 参数调优指南

### A.1 TTC阈值调优

| 应用场景 | ttc_critical | ttc_warning | ttc_max |
|---------|-------------|-------------|---------|
| 保守评估 | 2.0 | 5.0 | 15.0 |
| 标准评估 | 1.0 | 3.0 | 10.0 |
| 激进评估 | 0.5 | 2.0 | 8.0 |

### A.2 权重调优

| 目标 | 调整建议 |
|------|---------|
| 增加碰撞场景 | weight_collision ↑ |
| 减少误报 | weight_ttc_lateral ↓ |
| 强调同车道危险 | weight_ttc_same_lane ↑ |
| 增加变道场景 | weight_lane_change_conflict ↑ |

---

## 附录B: 常见问题

**Q1: 为什么相邻车道无变道时评分为0？**

A: 这是设计意图。相邻车道的车辆如果保持车道，不会对ego造成威胁，因此危险评分应该为0。

**Q2: 梯度计算为什么这么慢？**

A: 数值梯度需要多次计算评分函数。可以考虑：
- 使用自动微分框架（PyTorch）
- 减少评估的时间点
- 使用更大的epsilon（牺牲精度）

**Q3: 如何处理多个agent？**

A: 当前设计是评估一个agent对的危险。对于多agent场景，可以：
- 分别计算每个危险agent的评分，取最大值或加权平均
- 扩展函数支持多agent联合评估

---

文档版本: 1.0
最后更新: 2026-04-07
作者: Claude Code Assistant
