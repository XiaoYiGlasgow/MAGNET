# Diffusion Danger Model - Data Processing

## Directory Structure

```
diffusion_danger_model/
├── config.py              # Configuration file
├── process_pittsburgh.py  # Pittsburgh data processing script
├── processed_data/        # Output data directory
│   ├── features_*.pkl     # Processed feature files
│   └── processing_report_*.md  # Automatically generated processing report
└── utils/                 # Utility functions (if needed)
```

## Configuration

| Parameter | Value | Description |
|------|---|------|
| PAST_TIME_HORIZON | 4 s | History time window |
| FUTURE_TIME_HORIZON | 6 s | Future time window |
| NUM_PAST_POSES | 40 frames | Number of history trajectory frames |
| NUM_FUTURE_POSES | 60 frames | Number of future trajectory frames |
| NUM_AGENTS | 30 | Maximum number of surrounding vehicles |
| QUERY_RADIUS | 50m | Map lane extraction range |
| MIN_DISPLACEMENT | 10m | Minimum displacement (excludes stationary scenes) |

## Usage

### 1. Process a Small Amount of Data (Test)

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/diffusion_danger_model
python process_pittsburgh.py --max 100
```

### 2. Process All Data

```bash
python process_pittsburgh.py
```

### 3. Custom Output Directory

```bash
python process_pittsburgh.py --output /path/to/output --max 10000
```

## Output Data Format

Each `features_*.pkl` file contains:

| Field | Shape | Description |
|------|------|------|
| POLYLINE_FEATURES | (N, 12) | 12-dimensional feature vector |
| GROUND_TRUTH | (1, 60, 2) | 60-frame future trajectory increments |
| TRAJ_ID_TO_INDICES | dict | Mapping from trajectory ID to indices |
| EGO_LAST_POS | [x, y] | Ego absolute position |

## Automatically Generated Processing Report

After processing finishes, a `processing_report_*.md` report file is generated automatically, containing:

- **Processing configuration**: time windows, vehicle parameters, filtering configuration
- **Processing result statistics**: success/failure counts, processing speed, time taken
- **Distribution of surrounding vehicle counts**: vehicle count statistics and shares
- **Data format validation**: field shapes and feature descriptions
- **Notes**: ID meanings, caveats

### Viewing the Report

```bash
# View the most recently generated report
cat processed_data/processing_report_*.md | head -100
```

## Notes on Vehicle Counts

### ID Meanings

| ID | Type | Counted in statistics |
|----|------|------------|
| ID=0 | Ego (own vehicle) | ❌ Not counted |
| ID>0 + type='agent' | Surrounding vehicle | ✓ Counted |
| ID>0 + type='lane' | Lane | ❌ Not counted |

### Caveats

1. **The vehicle count depends on the NuPlan data itself**, not on the result of danger filtering
2. **Scenes with 0 vehicles**: the NuPlan data genuinely has no surrounding vehicles at that moment
3. **Subsequent processing**: `ttc_filter.py` must be used for TTC danger scene filtering

## Subsequent Processing

After processing finishes, use `ttc_filter.py` for TTC filtering:
- Extract danger scenes from processed_data
- Generate danger_pair_scenarios for Diffusion training