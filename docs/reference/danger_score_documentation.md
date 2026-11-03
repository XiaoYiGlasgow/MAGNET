# Danger Score Module Description

## Purpose of This Document

This document is intended for developers who need to understand the design principles, scoring logic and interface structure of `danger_score.py`. Its focus is how the danger score module provides an interpretable and extensible risk signal for Diffusion-based danger scenario generation.

## Scope

- Reading and maintaining `utils/danger_score.py`
- Understanding the scoring sources, such as TTC, distance, collision and lane relationship
- Interfacing with the danger metric in Diffusion training

## Related Files

- `utils/danger_score.py`
- `utils/danger_score_torch.py`
- `docs/reference/diffusion_generator_documentation.md`
- `docs/diffusion_usage_guide.md`

## Recommended Reading Order

1. Read this document first to understand the components and design goals of the danger score
2. Then read `diffusion_generator_documentation.md` to understand how the score is connected to the generative model
3. Finally read `diffusion_usage_guide.md` and follow the workflow to run training and generation

## Overview

This document describes in detail the design principles, implementation details and usage of the `danger_score.py` module. The module computes the danger score of a trajectory pair between the ego vehicle and a surrounding vehicle in autonomous driving scenarios, providing a guidance signal for the Diffusion model to generate dangerous trajectories.

---

## 1. Overall Design

### 1.1 Inputs and Outputs

```
Inputs:
├── ego_history: np.ndarray, shape=(T_hist, 5)
│   └── Past trajectory [x, y, heading, vx, vy]
├── ego_future: np.ndarray, shape=(T_future, 5)
│   └── Future trajectory (to be evaluated)
├── agent_history: np.ndarray, shape=(T_hist, 5)
└── agent_future: np.ndarray, shape=(T_future, 5)

Outputs:
├── total_score: float
│   └── Overall danger score
├── component_scores: Dict[str, float]
│   └── Per-component score details
├── ttc_info: Dict
│   └── TTC analysis details
├── collision_info: Dict
│   └── Collision detection details
├── lane_info: Dict
│   └── Lane relationship information
└── danger_level: str
    └── Danger level classification
```

### 1.2 Core Ideas

1. **Multi-dimensional danger assessment**: danger is the combined effect of multiple factors, including TTC, distance, speed and lane relationship
2. **Lane awareness**: vehicles with different lane relationships are assessed with different danger logic
3. **Differentiable design**: the scoring function is differentiable, supporting gradient guidance for the diffusion model

### 1.3 Module Structure

```
danger_score.py
├── Configuration class
│   └── DangerScoreConfig
├── Lane-awareness functions
│   ├── get_lane_index()
│   ├── get_lane_relationship()
│   └── detect_lane_change_intent()
├── TTC computation functions
│   ├── compute_ttc_1d()
│   ├── compute_relative_state()
│   └── compute_ttc_at_timestep()
├── Danger score components
│   ├── ttc_danger_score()
│   ├── distance_danger_score()
│   ├── collision_detection()
│   ├── relative_velocity_danger_score()
│   ├── acceleration_danger_score()
│   ├── lane_change_conflict_score()
│   └── compute_lane_aware_ttc_danger()
├── Main functions
│   ├── compute_danger_score()
│   └── classify_danger_level()
└── Diffusion utility functions
    ├── compute_danger_gradient()
    └── batch_compute_danger_scores()
```

---

## 2. Configuration Parameters in Detail

### 2.1 TTC Thresholds

```python
ttc_critical: float = 1.0   # Critical TTC (seconds)
ttc_warning: float = 3.0    # Warning TTC (seconds)
ttc_max: float = 10.0       # Maximum TTC considered (seconds)
```

| Threshold | Meaning | Scoring strategy |
|------|------|---------|
| ttc_critical | Very high collision risk within 1 second | Score doubled |
| ttc_warning | Requires attention within 3 seconds | Normal score |
| ttc_max | Not considered beyond 10 seconds | Score attenuated to 0.1x |

### 2.2 Distance Thresholds

```python
min_safe_distance: float = 5.0      # Minimum safe distance (metres)
collision_distance: float = 2.0     # Distance regarded as a collision (metres)
near_miss_distance: float = 3.0     # Near-distance threshold (metres)
```

```
Distance range classification:

  0m        2m         3m         5m
   |---------|----------|----------|----------→ distance
   collision  near_miss   safe
   
   high score  medium score  low/zero score
```

### 2.3 Lane Parameters

```python
lane_width: float = 3.5              # Standard lane width (metres)
same_lane_threshold: float = 1.75    # Same-lane threshold (metres)
adjacent_lane_threshold: float = 5.25  # Adjacent-lane threshold (metres)
lane_change_vy_threshold: float = 0.5  # Lateral velocity threshold for lane change (metres/second)
```

**Basis for the thresholds**:
- `same_lane_threshold = lane_width / 2 = 1.75m`
- `adjacent_lane_threshold = lane_width * 1.5 = 5.25m`

### 2.4 Weight Parameters

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

**Weight design principles**:
- The collision weight is the highest (10.0), because a collision is the most dangerous situation
- Same-lane TTC receives a weight bonus (1.5x), because the rear-end collision risk is highest in the same lane
- Lane-change conflict weight (2.0), because lane-change collisions are hard to predict

---

## 3. Lane-Awareness System

### 3.1 Coordinate System Definition

```
        Y (left)
        ↑
        |
   lane 1 | lane 0 | lane -1
   (Y>0) | (Y≈0) | (Y<0)
        |       |
        |  ego  +------→ X (forward)
        |       |
        |       |
        ↓
        Y (right)

Conventions:
- The ego lane is lane 0, with Y coordinate approximately 0
- Left lanes have Y > 0
- Right lanes have Y < 0
```

### 3.2 Lane Index Computation

```python
def get_lane_index(y_position: float, lane_width: float = 3.5) -> int:
    return int(round(y_position / lane_width))
```

**Example**:

| Y coordinate (metres) | Lane index | Relative position |
|-----------|---------|---------|
| 0.0 | 0 | Same lane |
| 3.5 | 1 | First lane on the left |
| -3.5 | -1 | First lane on the right |
| 7.0 | 2 | Second lane on the left |

### 3.3 Lane Relationship Classification

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

**Classification logic diagram**:

```
Relative Y coordinate (metres)
     10.5  |  far_left
      5.25  ├───────────────── boundary
      3.5   |  adjacent_left
      1.75  ├───────────────── boundary
      0.0   |  same_lane      ← ego position
     -1.75  ├───────────────── boundary
     -3.5   |  adjacent_right
     -5.25  ├───────────────── boundary
    -10.5   |  far_right
```

### 3.4 Lane-Change Intent Detection

```python
def detect_lane_change_intent(agent_vy, config) -> Tuple[bool, str]:
    abs_vy = abs(agent_vy)

    if abs_vy < 0.5:
        return False, 'none'

    return True, 'left' if agent_vy > 0 else 'right'
```

**Physical principle**:
- When a vehicle drives steadily within its lane, its lateral velocity is close to 0
- A significant lateral velocity (|vy| > 0.5 m/s) indicates that it is changing lanes
- The sign of the velocity determines the direction of the lane change

**Example**:

| Lateral velocity vy (m/s) | Lane change? | Lane-change direction |
|------------------|---------|---------|
| 0.0 | No | - |
| 0.3 | No | - |
| 0.8 | Yes | Left |
| -1.0 | Yes | Right |

---

## 4. TTC Computation in Detail

### 4.1 One-Dimensional TTC Formula

$$TTC = \begin{cases}
\frac{d}{-v_{rel}} & \text{if } v_{rel} < 0 \\
\infty & \text{if } v_{rel} \geq 0
\end{cases}$$

where:
- $d$ = distance between the two vehicles (metres)
- $v_{rel}$ = relative velocity (metres/second); a negative value indicates approach

```python
def compute_ttc_1d(distance, relative_velocity, epsilon=1e-6) -> float:
    if relative_velocity >= 0:
        return np.inf  # Moving apart
    if distance <= 0:
        return 0.0    # Already overlapping
    return distance / (-relative_velocity + epsilon)
```

### 4.2 Longitudinal TTC Computation

**Scenario analysis**:

```
Case 1: Agent ahead (rel_x > 0)

    ego ────→     ←──── agent
          rel_x
    
    If ego speed > agent speed (rel_vx < 0), they are approaching
    lon_ttc = rel_x / (-rel_vx)

Case 2: Agent behind (rel_x < 0)

    agent ────→     ←──── ego
              rel_x
    
    If agent speed > ego speed (rel_vx > 0), they are approaching
    lon_ttc = |rel_x| / rel_vx
```

### 4.3 Lateral TTC Computation

```python
if rel_state['rel_y'] > 0:
    # Agent on the left
    lat_ttc = compute_ttc_1d(rel_state['rel_y'], rel_state['rel_vy'])
else:
    # Agent on the right
    lat_ttc = compute_ttc_1d(abs(rel_state['rel_y']), -rel_state['rel_vy'])
```

### 4.4 Relative State Computation

```python
def compute_relative_state(ego_state, agent_state) -> Dict:
    # Relative position
    rel_x = agent_x - ego_x   # Positive = agent ahead
    rel_y = agent_y - ego_y   # Positive = agent on the left

    # Relative velocity
    rel_vx = agent_vx - ego_vx
    rel_vy = agent_vy - ego_vy

    # Distance
    euclidean_dist = np.sqrt(rel_x**2 + rel_y**2)
```

---

## 5. Danger Score Components

### 5.1 Component List

| Component name | Basis of computation | Weight |
|-------|---------|------|
| ttc_score | TTC value | 1.0 |
| lane_aware_ttc_score | Lane-aware TTC | Dynamic |
| distance_score | Minimum distance | 1.5 |
| collision_score | Collision detection | 10.0 |
| near_miss_score | Near distance | 3.0 |
| velocity_score | Relative velocity | 0.5 |
| acceleration_score | Acceleration | 0.3 |
| lane_change_conflict_score | Lane-change conflict | 2.0 |

### 5.2 TTC Danger Score

```python
def ttc_danger_score(lon_ttc, lat_ttc, config) -> float:
    score = 0.0

    # Longitudinal TTC
    if lon_ttc < ttc_critical:
        score += weight * (ttc_critical - lon_ttc) * 2.0  # Doubled
    elif lon_ttc < ttc_warning:
        score += weight * (ttc_warning - lon_ttc)
    elif lon_ttc < ttc_max:
        score += weight * (ttc_max - lon_ttc) * 0.1  # Attenuated

    # Lateral TTC (similar logic)
    ...

    return score
```

**Score curve**:

```
Score
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

### 5.3 Lane-Aware TTC Score

This is the core innovation of the module: different scoring strategies are applied according to the lane relationship:

```python
def compute_lane_aware_ttc_danger(lon_ttc, lat_ttc, lane_rel, lane_intent, config):
    if lane_rel == 'same_lane':
        # Same lane: longitudinal TTC is the main danger
        # Weight bonus of 1.5x
        return longitudinal_danger(lon_ttc) * 1.5

    elif lane_rel in ['adjacent_left', 'adjacent_right']:
        if has_lane_change_intent:
            # Lane-change intent: consider both longitudinal and lateral TTC
            return longitudinal_danger(lon_ttc) + lateral_danger(lat_ttc) * 2.0
        else:
            # No lane-change intent: not dangerous
            return 0.0

    else:  # far_left, far_right
        # Distant lane: very low danger
        return longitudinal_danger(lon_ttc) * 0.3
```

**Scoring strategy table**:

| Lane relationship | Lane-change intent | Longitudinal TTC weight | Lateral TTC weight | Notes |
|---------|---------|------------|------------|------|
| same_lane | - | 1.5 | 0.8 | Highest rear-end collision risk in the same lane |
| adjacent | No | 0 | 0 | No interference from the adjacent lane |
| adjacent | Yes | 1.0 | 2.0 | Lane-change conflict risk |
| far | - | 0.3 | 0.2 | Low risk at larger distance |

### 5.4 Collision Detection

```python
def collision_detection(ego_traj, agent_traj, config):
    for t in range(len(ego_traj)):
        # Euclidean distance
        dist = sqrt((ego_x - agent_x)² + (ego_y - agent_y)²)

        # Collision threshold (simplified model)
        # Accounts for the sizes of both vehicles
        threshold = (ego_length + agent_length)/4 + (ego_width + agent_width)/4
        # = (4.5 + 4.0)/4 + (2.0 + 2.0)/4 = 3.125 metres

        if dist < threshold:
            has_collision = True
            collision_timestep = t

        min_distance = min(min_distance, dist)

    return has_collision, min_distance, collision_timestep
```

### 5.5 Lane-Change Conflict Detection

```python
def lane_change_conflict_score(ego_traj, agent_traj, config):
    for t in range(len(ego_traj)):
        ego_intent, ego_dir = detect_lane_change_intent(ego_vy)
        agent_intent, agent_dir = detect_lane_change_intent(agent_vy)

        # Both vehicles intend to change lanes, in opposite directions (towards the same lane)
        if ego_intent and agent_intent and ego_dir != agent_dir:
            dist = abs(ego_y - agent_y)
            if dist < adjacent_lane_threshold:
                score += weight_lane_change_conflict

    return score
```

**Illustration**:

```
Lane-change conflict scenario:

    lane 1    |  lane 0  |  lane -1
            |         |
    agent → | ← ego   |
            |         |
            ↓ lane-change conflict ↓
```

---

## 6. Main Function Workflow

### 6.1 Computation Flowchart

```
                    ┌─────────────────┐
                    │   Input data    │
                    │ ego/agent traj. │
                    └────────┬────────┘
                             ↓
              ┌──────────────┴──────────────┐
              │ Iterate over each future frame │
              └──────────────┬──────────────┘
                             ↓
    ┌────────────────────────┼────────────────────────┐
    ↓                        ↓                        ↓
┌─────────┐            ┌─────────┐            ┌─────────┐
│ TTC comp.│           │ Lane cls.│           │ LC detect│
└────┬────┘            └────┬────┘            └────┬────┘
     └───────────────────────┼───────────────────────┘
                             ↓
                    ┌─────────────────┐
                    │ TTC danger score │
                    │ (lane-aware ver.)│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Collision detect │
                    └────────┬────────┘
                             ↓
              ┌──────────────┴──────────────┐
              │ Compute other component scores │
              │ velocity/acceleration/...  │
              └──────────────┬──────────────┘
                             ↓
                    ┌─────────────────┐
                    │ Weighted sum total │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Danger level cls. │
                    └─────────────────┘
```

### 6.2 Code Structure

```python
def compute_danger_score(ego_history, ego_future, agent_history, agent_future, config, dt, verbose):
    # 1. Initialisation
    component_scores = {each component: 0.0}
    ttc_info = {TTC details}
    lane_info = {lane information}

    # 2. Iterate over the future trajectory
    for t in range(len(ego_future)):
        # Compute the TTC and lane information for this frame
        ttc_result = compute_ttc_at_timestep(...)

        # Accumulate TTC scores
        component_scores['ttc_score'] += ttc_danger_score(...)
        component_scores['lane_aware_ttc_score'] += compute_lane_aware_ttc_danger(...)

        # Record lane relationships
        lane_info['lane_relationships'].append(ttc_result['lane_relationship'])

    # 3. Collision detection
    has_collision, min_distance, collision_t = collision_detection(...)

    # 4. Compute other component scores
    component_scores['velocity_score'] = relative_velocity_danger_score(...)
    component_scores['acceleration_score'] = acceleration_danger_score(...)
    component_scores['lane_change_conflict_score'] = lane_change_conflict_score(...)

    # 5. Compute the total score
    total_score = sum(component_scores.values())

    # 6. Danger level classification
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

### 6.3 Danger Level Classification

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

## 7. Diffusion Guidance

### 7.1 Gradient Computation Principle

Used for classifier guidance, steering trajectory generation towards a "more dangerous" direction during diffusion sampling.

**Mathematical principle**:

$$x_{t-1} = x_t - \epsilon_\theta(x_t, t) + \lambda \cdot \nabla_{x_t} D(x_t)$$

where:
- $\epsilon_\theta$ is the noise prediction network
- $D(x_t)$ is the danger score function
- $\lambda$ is the guidance strength
- $\nabla_{x_t} D(x_t)$ is the gradient of the danger score with respect to the trajectory

### 7.2 Numerical Gradient Implementation

```python
def compute_danger_gradient(ego_future, agent_future, ego_history, agent_history, config, epsilon=0.1):
    # Base score
    base_score = compute_danger_score(ego_history, ego_future, agent_history, agent_future)

    # Compute the gradient at each position of the ego trajectory
    for t in range(len(ego_future)):
        for dim in range(2):  # x, y
            # Forward difference
            ego_perturbed = ego_future.copy()
            ego_perturbed[t, dim] += epsilon

            new_score = compute_danger_score(ego_history, ego_perturbed, ...)

            gradient[t, dim] = (new_score - base_score) / epsilon

    return ego_gradient, agent_gradient
```

### 7.3 Batch Computation

```python
def batch_compute_danger_scores(ego_futures, agent_futures, ego_history, agent_history, config):
    """
    Used for reward-weighted regression:
    1. Generate multiple candidate trajectories
    2. Compute the danger score of each trajectory
    3. Update the model with weighting by high-scoring trajectories
    """
    scores = np.zeros(batch_size)
    for i in range(batch_size):
        scores[i] = compute_danger_score(..., ego_futures[i], agent_futures[i], ...)
    return scores
```

---

## 8. Test Scenario Analysis

### 8.1 Scenario 1: Approaching in the Same Lane

**Setup**:
```
Ego: moving forward at 10 m/s
Agent: moving forward at 8 m/s (20 metres ahead, same lane)
```

**Results**:
```
Longitudinal TTC: 7.00 seconds
Lateral TTC: inf
Danger score: 0.38
Danger level: LOW
Lane relationship: same_lane
```

**Analysis**:
- The TTC is large (7 seconds), so the danger is low
- Same lane, so the focus is on longitudinal TTC
- The score comes mainly from lane_aware_ttc_score

### 8.2 Scenario 2: Collision Trajectory

**Setup**:
```
The agent is slower; the final distance is only 0.17 metres
```

**Results**:
```
Danger score: 212.49
Danger level: CRITICAL
Collision detected: True
Minimum distance: 0.17m
Collision timestep: 20
```

**Analysis**:
- The collision triggers collision_score (100 points)
- min_distance triggers distance_score (100 points)
- Longitudinal TTC close to 0 triggers a high TTC score

### 8.3 Scenario 3: Adjacent Lane without Lane Change

**Setup**:
```
Agent in the left lane (Y=3.5 metres)
Lateral velocity = 0 (no lane-change intent)
```

**Results**:
```
Danger score: 0.00
Danger level: SAFE
Lane relationship: adjacent_left
```

**Analysis**:
- Adjacent lane, no lane-change intent
- lane_aware_ttc_score = 0 (correct behaviour)
- Vehicles in different lanes do not affect each other

### 8.4 Scenario 4: Lane-Change Conflict

**Setup**:
```
Agent changes lanes to the right from the left lane (Y=3.5 metres)
Lateral velocity = -0.8 m/s
```

**Results**:
```
Lateral TTC: 0.11 seconds
Danger score: 0.67
Danger level: LOW
Main lane relationship: same_lane
```

**Analysis**:
- Lane-change intent is detected
- The lateral TTC is very small (0.11 seconds)
- The agent gradually enters the ego lane

---

## 9. Usage Examples

### 9.1 Basic Usage

```python
from danger_score import compute_danger_score, DangerScoreConfig

# Prepare data
ego_history = np.array(...)  # (41, 5)
ego_future = np.array(...)   # (30, 5)
agent_history = np.array(...) # (41, 5)
agent_future = np.array(...)  # (30, 5)

# Compute the danger score
config = DangerScoreConfig()
result = compute_danger_score(
    ego_history, ego_future,
    agent_history, agent_future,
    config=config,
    dt=0.1,
    verbose=True
)

print(f"Danger score: {result['total_score']}")
print(f"Danger level: {result['danger_level']}")
```

### 9.2 Custom Configuration

```python
# Adjust thresholds and weights
config = DangerScoreConfig(
    ttc_critical=0.5,           # Stricter critical TTC
    lane_change_vy_threshold=0.3,  # More sensitive lane-change detection
    weight_collision=20.0,      # Higher collision weight
    weight_ttc_same_lane=2.0    # Higher same-lane TTC weight
)
```

### 9.3 Diffusion Guidance

```python
from danger_score import compute_danger_gradient

# Inside the diffusion sampling loop
for t in reversed(range(T)):
    # Predict noise
    noise_pred = model(x_t, t, condition)

    # Denoise
    x_t_minus_1 = denoise_step(x_t, noise_pred, t)

    # Compute the danger gradient
    ego_grad, agent_grad = compute_danger_gradient(
        x_t_minus_1[:30],    # ego future
        x_t_minus_1[30:],    # agent future
        ego_history,
        agent_history
    )

    # Guide towards a more dangerous direction
    x_t_minus_1[:30, :, :2] += guidance_scale * ego_grad
    x_t_minus_1[30:, :, :2] += guidance_scale * agent_grad
```

---

## 10. Notes and Limitations

### 10.1 Current Limitations

1. **Simplified collision detection**: uses AABB rather than OBB, and does not account for vehicle heading
2. **Fixed lane width**: assumes a lane width of 3.5 metres, which may vary in practice
3. **No map integration**: the NuPlan map API is not used to determine lane membership
4. **Single-agent evaluation**: currently only the danger of one agent pair is evaluated

### 10.2 Directions for Improvement

1. **Integrate the map API**: use the NuPlan map to determine precise lane membership
2. **Joint multi-agent evaluation**: consider the joint danger of multiple agents
3. **Dynamic thresholds**: adjust thresholds automatically according to scenario type
4. **Neural network scoring**: replace the hand-designed scoring function with a learned method

### 10.3 Performance Considerations

- A single score computation takes about 1-5ms (depending on trajectory length)
- Gradient computation requires an additional 2*T*D score computations
- Parallelisation is recommended for batch computation

---

## 11. References

1. TTC computation: Hayward, J.C. (1972). "Near miss determination through use of a scale of danger"
2. Diffusion Guidance: Dhariwal, P. & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis"
3. NuPlan documentation: https://nuplan-devkit.readthedocs.io/

---

## Appendix A: Parameter Tuning Guide

### A.1 TTC Threshold Tuning

| Application scenario | ttc_critical | ttc_warning | ttc_max |
|---------|-------------|-------------|---------|
| Conservative evaluation | 2.0 | 5.0 | 15.0 |
| Standard evaluation | 1.0 | 3.0 | 10.0 |
| Aggressive evaluation | 0.5 | 2.0 | 8.0 |

### A.2 Weight Tuning

| Goal | Suggested adjustment |
|------|---------|
| Increase collision scenarios | weight_collision ↑ |
| Reduce false alarms | weight_ttc_lateral ↓ |
| Emphasise same-lane danger | weight_ttc_same_lane ↑ |
| Increase lane-change scenarios | weight_lane_change_conflict ↑ |

---

## Appendix B: Frequently Asked Questions

**Q1: Why is the score 0 in an adjacent lane without a lane change?**

A: This is by design. A vehicle in an adjacent lane that keeps its lane poses no threat to the ego vehicle, so its danger score should be 0.

**Q2: Why is gradient computation so slow?**

A: The numerical gradient requires the scoring function to be computed many times. Options to consider:
- Use an automatic differentiation framework (PyTorch)
- Reduce the number of evaluated time points
- Use a larger epsilon (at the cost of accuracy)

**Q3: How are multiple agents handled?**

A: The current design evaluates the danger of one agent pair. For multi-agent scenarios, one can:
- Compute the score for each dangerous agent separately and take the maximum or a weighted average
- Extend the function to support joint multi-agent evaluation

---

Document version: 1.0
Last updated: 2026-04-07
