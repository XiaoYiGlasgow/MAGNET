# GNN_Trans: Graph Neural Network Transformer for Trajectory Prediction

A deep learning model combining Graph Neural Networks and Transformer architecture for autonomous vehicle trajectory prediction using the nuPlan dataset.

## Prerequisites

### 1. Install nuplan-devkit

Before training the GNN_Trans model, you need to install the nuplan-devkit project:

```bash
# Clone nuplan-devkit repository
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit

# Create conda environment
conda create -n nuplan python=3.9
conda activate nuplan

# Install dependencies
pip install -e .
pip install torch torchvision torchaudio
pip install torch-geometric
```

### 2. Data Setup

Download the nuPlan dataset and configure the data paths:

```bash
export NUPLAN_DATA_ROOT="/path/to/nuplan/data"
export NUPLAN_MAPS_ROOT="/path/to/nuplan/maps"
```

## Installation

1. Clone this repository inside the nuplan-devkit folder:
```bash
cd nuplan-devkit
git clone https://github.com/XiaoYiGlasgow/GNN_Trans.git
```

2. Install additional dependencies:
```bash
cd GNN_Trans
pip install pandas matplotlib seaborn scipy
```

## Usage

### Training

```bash
cd GNN_Trans
python train.py
```

### Evaluation

```bash
python evaluate.py
```

### Visualization

Generate comprehensive evaluation reports:
```bash
python visualize_evaluation.py
```

### nuBoard Visualization

Visualize predictions in nuBoard:
```bash
cd Nuboard
python run_gnn_trans_simulation.py
python launch_gnn_trans_nuboard.py
```

## Model Architecture

The GNN_Trans model combines:
- **Graph Neural Network**: Processes spatial relationships between vehicles
- **Transformer**: Captures temporal dependencies in trajectories
- **Multi-head Attention**: Focuses on relevant trajectory features

## Performance

Current model performance:
- **ADE**: 0.57 meters (Average Displacement Error)
- **FDE**: 1.26 meters (Final Displacement Error)
- **MSE Loss**: 0.002

## Project Structure

```
GNN_Trans/
├── models/                 # Model architectures
├── utils/                  # Data processing utilities
├── Nuboard/               # nuBoard visualization tools
├── GNN_Trans_params/      # Trained model weights
├── processed_data/        # Processed training data
├── train.py              # Training script
├── evaluate.py           # Evaluation script
└── visualize_evaluation.py # Visualization tools
```

## License

This project is for research and educational purposes.
