# GNN_Trans - Graph Neural Network with Transformer for Trajectory Prediction

This project implements a trajectory prediction model based on Graph Neural Networks (GNN) and Transformer architecture, designed for autonomous driving scenarios.

## Project Structure

- `train.py` - Model training script
- `evaluate.py` - Model evaluation script
- `inference.py` - Model inference script
- `config/` - Configuration files directory
- `models/` - Model definitions
- `planner/` - Planner-related code
- `utils/` - Utility functions
- `simple_demo.py` - Simple demonstration script

## Quick Start

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Train the model
```bash
python train.py
```

3. Evaluate the model
```bash
python evaluate.py
```

## Configuration

Main configuration files are located in the `config/` directory. You can adjust model parameters and training settings as needed.

## Demo

Run the demonstration script to see the model performance:
```bash
python simple_demo.py
```
