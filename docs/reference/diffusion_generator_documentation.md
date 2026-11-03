# Diffusion Dangerous Scenario Generator

## Document Scope

This document is intended for developers who need to understand the overall design, core modules and training objectives of the Diffusion dangerous scenario generation system. It focuses on the model architecture, the condition encoding scheme and the relationships between components.

## Applicability

- Reading and maintaining the Diffusion generation code
- Understanding `SceneConditionEncoder`, `DangerSceneDiffusion` and the training loss
- Connecting the danger scoring module with the generation module

## Related Files

- `models/scene_condition_encoder.py`
- `models/danger_diffusion.py`
- `utils/danger_score_torch.py`
- `train_diffusion_generator.py`
- `generate_danger_scenes.py`

## Recommended Reading Order

1. Read this document first to understand the model architecture and training objectives
2. Then read `danger_score_documentation.md` to understand where the risk signal comes from
3. Finally read `diffusion_usage_guide.md` and follow the workflow to run training and generation

## Overview

This document describes a dangerous scenario generation system based on a diffusion model (Diffusion Model), used to generate driving scenario data that is more dangerous than the GT (Ground Truth) trajectories.

### Goals

- **Input**: historical trajectories + SVO angle + danger_target
- **Output**: generated future trajectories (more dangerous than GT)
- **Uses**: data augmentation, risk boundary testing, extreme scenario simulation

### Training Objectives

1. **Reconstruct the GT trajectory**: learn basic trajectory generation capability
2. **Danger exceedance**: danger of the generated trajectory > danger of GT

---

## System Architecture

### Overall Structure

```
┌──────────────────────────────────────────────────────────────┐
│          Diffusion Dangerous Scenario Generator              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Input conditions                                            │
│  ├── History encoding (from GNN+Trans encoder, 384-dim)      │
│  ├── SVO angle condition (controls behavioural tendency)     │
│  │   - Competitive (-45°) → more dangerous                   │
│  │   - Cooperative (45°) → safer                             │
│  └── danger_target (target danger level)                     │
│      - danger_target > GT_danger → more dangerous trajectory │
│                                                              │
│  Condition encoder                                           │
│  ├── SceneConditionEncoder                                   │
│  ├── Input: scene_feat(384) + svo_angle + danger             │
│  └── Output: 256-dim condition vector                        │
│                                                              │
│  Diffusion generation                                        │
│  ├── DangerSceneDiffusion (DDPM)                             │
│  ├── 100-step denoising process                              │
│  ├── Transformer-based Denoiser                              │
│  └── Output: trajectory increments (120-dim = 60 steps × dx,dy) │
│                                                              │
│  Evaluation                                                  │
│  ├── compute_danger_score_torch (generated trajectory)       │
│  ├── Comparison with GT                                      │
│  └── Saved as new training samples                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## File Structure

### New Files

| File Path | Function |
|----------|------|
| `utils/danger_score_torch.py` | PyTorch version of the danger computation, supporting GPU and autograd |
| `models/scene_condition_encoder.py` | Condition encoder (scene + SVO + Danger) |
| `models/danger_diffusion.py` | Diffusion generator (DDPM) |
| `train_diffusion_generator.py` | Training script |
| `generate_danger_scenes.py` | Generation script |

### Modified Files

| File Path | Modification |
|----------|----------|
| `models/transformer_predictor.py` | Added the `get_encoder_features()` method |

---

## Core Components in Detail

### 1. danger_score_torch.py

PyTorch version of the danger computation module, used for loss computation in Diffusion training.

#### Main Functions

```python
# Compute the danger index of a single trajectory segment
compute_single_trajectory_danger_torch(ego_traj, agent_traj, config)

# Main danger scoring function
compute_danger_score_torch(ego_history, ego_future, agent_history, agent_future)

# Recover the full trajectory from trajectory increments
recover_trajectory_from_increment(increments, last_pos, last_heading, last_vx, last_vy)

# Compute the trajectory smoothness loss
trajectory_smoothness_loss_torch(increments)
```

#### DangerExceedLoss

Core training loss class:

```python
class DangerExceedLoss(nn.Module):
    """
    Loss components:
    1. reconstruction_loss: MSE(pred_traj, gt_traj)
    2. danger_exceed_loss: ReLU(gt_danger - pred_danger + margin)
    3. smoothness_loss: sum of squared accelerations
    """
    
    def forward(self, pred_increments, gt_increments, ...):
        # Reconstruction loss
        recon_loss = F.mse_loss(pred_increments, gt_increments)
        
        # Danger exceedance loss
        danger_exceed_loss = F.relu(gt_danger + margin - pred_danger)
        
        # Smoothness loss
        smoothness_loss = trajectory_smoothness_loss_torch(pred_increments)
        
        return total_loss, loss_dict
```

### 2. scene_condition_encoder.py

Fuses multiple conditions into a single 256-dimensional condition vector.

#### SceneConditionEncoder

```python
class SceneConditionEncoder(nn.Module):
    """
    Inputs:
    - scene_feat: [batch, 384] from the GNN+Trans encoder
    - svo_angle: [batch, 1] SVO angle (degrees)
    - inherent_danger: [batch, 1] historical risk
    - danger_target: [batch, 1] target risk
    
    Output:
    - condition: [batch, 256] condition vector
    """
    
    def __init__(self, config):
        # Scene encoder: 384 -> 256
        self.scene_encoder = nn.Sequential(...)
        
        # SVO angle encoder: 1 -> 64
        self.svo_encoder = nn.Sequential(...)
        
        # Danger encoder: 2 -> 64
        self.danger_encoder = nn.Sequential(...)
        
        # Condition fusion layer
        self.condition_fusion = nn.Linear(384, 256)
```

#### SimpleSceneEncoder

When the GNN+Trans encoder is unavailable, features are extracted directly from the historical trajectories:

```python
class SimpleSceneEncoder(nn.Module):
    """
    Inputs:
    - ego_history: [batch, T_hist, 5]
    - agent_history: [batch, T_hist, 5]
    
    Output:
    - scene_feat: [batch, 384]
    """
```

### 3. danger_diffusion.py

A denoising diffusion model based on DDPM.

#### DiffusionConfig

```python
@dataclass
class DiffusionConfig:
    condition_dim: int = 256        # Condition vector dimension
    trajectory_dim: int = 120       # Trajectory dimension (60 steps × 2)
    num_diffusion_steps: int = 100  # Number of denoising steps
    beta_schedule: str = 'linear'   # Beta schedule type
    denoiser_num_layers: int = 4    # Number of Transformer layers
    condition_injection: str = 'cross_attention'  # Condition injection method
```

#### DangerSceneDiffusion

```python
class DangerSceneDiffusion(nn.Module):
    """
    Training procedure:
    1. Randomly sample a timestep t
    2. Add noise: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
    3. Predict the noise: noise_pred = denoiser(x_t, condition, t)
    4. Compute the loss: MSE(noise_pred, noise)
    
    Generation procedure:
    1. Initialise pure noise x_T
    2. Denoise step by step: from T to 1
    3. Obtain x_0 (the generated trajectory increments)
    """
    
    def add_noise(self, x_0, t):
        """Add noise (forward diffusion)"""
        
    def denoise_step(self, x_t, condition, t):
        """Single denoising step"""
        
    def generate(self, condition, num_samples=1):
        """Generate trajectories from noise"""
```

#### TrajectoryDenoiser

Transformer-based denoising network:

```python
class TrajectoryDenoiser(nn.Module):
    """
    Supports three condition injection methods:
    1. cross_attention: fuse the condition via cross-attention
    2. concat: concatenate the condition to the input
    3. addition: fuse by direct addition
    """
```

---

## SVO Angle and Behavioural Tendency

### SVO Angle Ranges

| Angle Range | Type | Behavioural Tendency | Generation Effect |
|----------|------|----------|----------|
| -45° ~ -15° | competitive | Competitive | More dangerous trajectories |
| -15° ~ 22.5° | individualistic | Individualistic | Medium risk |
| 22.5° ~ 57.5° | prosocial | Cooperative | Relatively safe trajectories |
| 57.5° ~ 135° | altruistic | Altruistic | Safest trajectories |

### Unit Circle Model

```
      y (collective weight)
      ↑
      │    altruistic (135°)
      │      ●
      │
      │  prosocial (45°)
      │    ●
      │
──────┼──────────────→ x (individual weight)
      │
      │  individualistic (0°)
      │    ●
      │
      │  competitive (-45°)
      │      ●
```

On the unit circle:
- x = cos(θ): individual weight
- y = sin(θ): collective weight
- θ = arctan2(y, x): SVO angle

---

## Data Flow

### Training Data Flow

```
danger_pair_*.pkl
    │
    ├── Data extraction
    │   ├── ego_history, agent_history (historical trajectories)
    │   ├── ego_future_gt, agent_future_gt (GT future trajectories)
    │   ├── svo_angle (computed)
    │   ├── inherent_danger (historical risk)
    │   ├── gt_danger (GT future risk)
    │   └── danger_target = gt_danger + margin
    │
    ├── SceneEncoder
    │   └── scene_feat = encoder(ego_history, agent_history)
    │
    ├── ConditionEncoder
    │   └── condition = encoder(scene_feat, svo, danger)
    │
    ├── Diffusion training
    │   ├── add_noise(gt_increments, t)
    │   ├── denoiser(noisy_traj, condition, t)
    │   └── compute_loss(pred, gt, danger)
    │
    └── Optimisation
        └── backward(loss)
```

### Generation Data Flow

```
Generating new dangerous scenarios at inference:

1. Load historical trajectory data
   ├── ego_history, agent_history
   ├── Set svo_angle (or random)
   ├── Compute inherent_danger
   └── Set danger_target (higher than GT)

2. SceneEncoder encoding
   scene_feat = encoder(ego_history, agent_history)

3. ConditionEncoder encoding
   condition = encoder(scene_feat, svo, danger)

4. Diffusion generation
   pred_increments = diffusion.generate(condition)

5. Trajectory recovery
   pred_trajectory = recover_from_increment(pred_increments)

6. Evaluate the generated result
   pred_danger = compute_danger_score(pred_trajectory)
   assert pred_danger > gt_danger

7. Save as a new training sample
   save_generated_sample(pred_trajectory, ...)
```

---

## Usage

### Training

```bash
# Basic training
python train_diffusion_generator.py \
    --data_dir danger_pair_scenarios \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4

# Debug mode (limited number of samples)
python train_diffusion_generator.py \
    --data_dir danger_pair_scenarios \
    --epochs 10 \
    --max_samples 50
```

### Generation

```bash
# Batch generation
python generate_danger_scenes.py \
    --model_path diffusion_training/best_model.pt \
    --data_dir danger_pair_scenarios \
    --output_dir generated_danger_scenes \
    --num_samples 3

# Single-scenario test
python generate_danger_scenes.py \
    --mode single \
    --model_path diffusion_training/best_model.pt \
    --pkl_path danger_pair_scenarios/danger_pair_features_xxx.pkl \
    --svo_angles -30 0 45 90 \
    --danger_margin 3.0
```

---

## Verification

### 1. Training Verification

Observe the downward trend of the loss in the training log:

```
Epoch 1: loss = 0.5234
Epoch 10: loss = 0.2156
Epoch 50: loss = 0.0843
Epoch 100: loss = 0.0234
```

Key metric:
- `danger_margin_met`: danger exceedance success rate

### 2. Generation Verification

Check whether the generated trajectories are more dangerous than GT:

```python
# Load the trained model
generator = DangerSceneGenerator('best_model.pt')

# Generate a dangerous scenario
result = generator.generate_from_pkl(
    pkl_path,
    svo_angle_override=-30,  # competitive
    danger_target_override=gt_danger + 3.0
)

# Check whether danger is exceeded
print(f"Generated danger: {result['generated_danger']:.2f}")
print(f"GT danger: {result['gt_danger']:.2f}")
print(f"Exceeded: {result['danger_exceeded']}")
```

### 3. Data Augmentation Effect

Train on the generated dangerous scenarios mixed with the original data:

```python
# Original data: N scenarios
original_data = load_danger_pair_data('danger_pair_scenarios')

# Generated data: M more dangerous scenarios
generated_data = load_generated_data('generated_danger_scenes')

# Mixed training
mixed_data = original_data + generated_data
train_prediction_model(mixed_data)
```

---

## Configuration Parameters

### DiffusionConfig

| Parameter | Default | Description |
|------|--------|------|
| condition_dim | 256 | Condition vector dimension |
| trajectory_dim | 120 | Trajectory dimension (60 steps × 2) |
| num_diffusion_steps | 100 | Number of denoising steps |
| beta_start | 1e-4 | Beta start value |
| beta_end | 0.02 | Beta end value |
| beta_schedule | 'linear' | Beta schedule type |
| denoiser_num_layers | 4 | Number of Transformer layers |
| denoiser_num_heads | 8 | Number of Transformer heads |

### ConditionEncoderConfig

| Parameter | Default | Description |
|------|--------|------|
| scene_feat_dim | 384 | Scene feature dimension |
| condition_dim | 256 | Output condition dimension |
| svo_embed_dim | 64 | SVO embedding dimension |
| danger_embed_dim | 64 | Danger embedding dimension |

### DangerExceedLoss Weights

| Parameter | Default | Description |
|------|--------|------|
| recon_weight | 1.0 | Reconstruction loss weight |
| danger_weight | 2.0 | Danger exceedance loss weight |
| smoothness_weight | 0.3 | Smoothness loss weight |
| danger_margin | 0.5 | Danger exceedance threshold |

---

## Notes

### 1. Data Format Requirements

The input pkl files must contain the following fields:
- `ego_history`: (T_hist, 5) - [x, y, heading, vx, vy]
- `key_agent_history`: (T_hist, 5)
- `ego_future_gt`: (T_future, 2) or (1, T_future, 2)
- `key_agent_future`: (T_future, 3) or None

### 2. SVO Angle Range

- Valid range: [-45°, 135°]
- Values outside the range are clamped

### 3. Danger Value Scale

- Danger values are recommended to lie in the range 0-50
- Normalisation maps values to [0, 1]

### 4. GPU Memory

- batch_size=8 requires about 4GB of GPU memory
- batch_size can be adjusted according to the GPU

---

## Integration with the Existing System

### GNN+Transformer Encoder

The modified `transformer_predictor.py` provides a new method:

```python
class TransformerPredictor(nn.Module):
    def get_encoder_features(self, data) -> torch.Tensor:
        """
        Extract encoder features for Diffusion
        
        Returns:
            [batch, 384] encoder feature vector
            = ego_feat(128) + context_feat(128) + gnn_feat(128)
        """
```

### Danger Score System

Compatible with the existing `danger_score.py` system:
- numpy version: used for preprocessing and evaluation
- PyTorch version: used for training loss computation

---

## Future Extensions

### 1. Multi-Agent Generation

`MultiAgentConditionEncoder` supports multi-agent condition encoding:

```python
encoder = MultiAgentConditionEncoder(max_agents=5)
condition = encoder(scene_feat, svo_angles, dangers, agent_mask)
```

### 2. Adaptive Danger Target

The target can be adjusted automatically based on the historical risk:

```python
danger_target = inherent_danger * 1.5  # based on historical risk
```

### 3. Condition-Guided Sampling

Use classifier-free guidance to strengthen conditional control:

```python
# Mix unconditional and conditional generation
trajectory = (1 + guidance_scale) * conditional_gen - guidance_scale * unconditional_gen
```

---

## References

1. **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
2. **Improved DDPM**: Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021
3. **SVO Theory**: Murphy & Ackermann, "Social Value Orientation", 2014

---

## Changelog

- 2026-04-10: First version completed, including the full training and generation system
