# Safe-Sim Integration

**Paper**: Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries (ECCV 2024)  
**Repo**: https://github.com/jxmmy7777/safe-sim

---

## Integration difficulty assessment: high (requires substantial adaptation work)

### Key obstacles

| Obstacle | Details | Work required |
|------|------|---------|
| **Incompatible input format** | Safe-Sim requires a rasterized HD map (224×224 image) + trajectory; our pkl contains only trajectories, no map | Implement an HD map tile → raster conversion module |
| **trajdata dependency** | Uses NVIDIA's modified `trajdata` fork; nuPlan support exists only as config, with no actual implementation | Implement a `nuPlan → AgentBatch` adapter |
| **Closed-loop design** | No standalone inference; depends on the whole simulation environment via `rollout_episodes()` | Write an inference wrapper that skips the environment loop |
| **Checkpoint domain gap** | The public checkpoint is trained only on nuScenes; there are no public weights for nuPlan | Retrain or fine-tune on nuPlan data |
| **Environment dependency conflicts** | `pytorch-lightning==1.9.0` + `python=3.8` + trajdata fork | Separate conda environment, isolated from the existing project |

### Conclusion

Running Safe-Sim inference directly on nuPlan pkl data is **not feasible**; a full re-adaptation is required.
Recommendations for the paper comparison:
- **Option A (recommended)**: cite the public nuScenes numbers from the Safe-Sim paper directly, noting that the domain differs
- **Option B**: complete the adaptation steps below and reproduce on nuPlan (effort ~1-2 weeks)

---

## Option B integration steps

### Step 1: Install the environment

```bash
# Separate conda environment
conda create -n safe_sim python=3.8
conda activate safe_sim

# Clone the official repo
git clone https://github.com/jxmmy7777/safe-sim.git /tmp/safe-sim
cd /tmp/safe-sim

# Install the modified trajdata (the fork must be used)
pip install -e trajdata/

# Install the main dependencies
pip install -r requirements.txt
```

### Step 2: Download the Checkpoint

```bash
# nuScenes pretrained checkpoint on Hugging Face (see the paper's README)
# There are currently no public weights for a nuPlan checkpoint
```

### Step 3: Map Rasterizer (largest workload)

The Safe-Sim model input `obs_dict["image"]` is a rasterized map of shape `[B, C, 224, 224]`.
Required:
1. Read the nuPlan HD map (`maps/*.db` files)
2. Generate a 224×224 map image centred on the current ego position (channels for roads, lanes, boundaries, etc.)
3. Corresponding file: the `rasterize_map()` function in `safe_sim/adapter.py` (to be implemented)

### Step 4: Data adapter

Convert our pkl format into the `obs_dict` required by Safe-Sim (see `adapter.py`).

### Step 5: Inference Wrapper

Bypass the `rollout_episodes()` closed loop, execute only a single forward pass, and obtain the generated trajectories.

---

## File structure

```
safe_sim/
├── README.md           # This file
├── __init__.py
├── adapter.py          # pkl → obs_dict format conversion (core of Steps 3-4)
└── inference.py        # Single forward pass wrapper (Step 5)
```
