# Diffusion Danger Scene Generator - Project Summary Report

**Generated**: 2026-04-14

---

## Project Overview

This project prepares training data for the Diffusion danger scene generator by reprocessing the NuPlan Pittsburgh dataset with a new time-window configuration (4 seconds of history + 6 seconds of future) and applying TTC (Time To Collision) danger scene filtering.

---

## 1. Data Processing Stage

### 1.1 Processing Configuration

| Parameter | Value | Description |
|------|---|------|
| Data source | Pittsburgh | NuPlan training dataset |
| History time window | 4 s (40 frames) | Differs from the original configuration (2 s) |
| Future time window | 6 s (60 frames) | Differs from the original configuration (8 s) |
| Frame rate | 10Hz | Sampling frequency |
| Maximum surrounding vehicles | 30 vehicles | NUM_AGENTS upper limit |
| Map extraction radius | 50 m | QUERY_RADIUS |
| Minimum displacement filter | 10 m | Excludes stationary scenes |

### 1.2 Processing Results

| Metric | Result |
|------|------|
| Successfully processed | **803 scenes** |
| Failed | 0 |
| Processing speed | 258.1 scenes/minute |
| Total time | 3.1 minutes |

### 1.3 Distribution of Surrounding Vehicle Counts

| Vehicles | Scenes | Share |
|--------|--------|------|
| 0 vehicles | 163 | 20.3% |
| 1 vehicle | 147 | 18.3% |
| 2 vehicles | 128 | 15.9% |
| 3 vehicles | 105 | 13.1% |
| 4-5 vehicles | 118 | 14.7% |
| 6-10 vehicles | 115 | 14.3% |
| 11-16 vehicles | 31 | 3.9% |

**Summary statistics**:
- Mean number of surrounding vehicles: 3.06
- Minimum: 0 vehicles, maximum: 16 vehicles
- Scenes with >=3 vehicles: 365 (45.5%)
- Scenes with >=5 vehicles: 196 (24.4%)

### 1.4 Output Data Format

Each `features_*.pkl` file contains:

| Field | Shape | Description |
|------|------|------|
| POLYLINE_FEATURES | (N, 12) | 12-dimensional feature vector |
| GROUND_TRUTH | (1, 60, 2) | 60-frame future trajectory increments |
| TRAJ_ID_TO_INDICES | dict | Mapping from trajectory ID to indices |
| EGO_LAST_POS | [x, y] | Ego absolute position |

**12-dimensional feature vector description**:
| Dimension | Field | Description |
|------|------|------|
| 0-1 | x_start, y_start | Segment start coordinates |
| 2-3 | x_end, y_end | Segment end coordinates |
| 4 | heading_start | Start heading |
| 5 | heading_end | End heading |
| 6 | normalized_timestamp | Normalised timestamp |
| 7-8 | v_start, v_end | Start/end speed |
| 9 | subgraph_ID | Subgraph ID |
| 10-11 | start_time, end_time | Relative time |

---

## 2. TTC Danger Scene Filtering Stage

### 2.1 TTC Filtering Configuration

| Parameter | Value | Description |
|------|---|------|
| Longitudinal TTC threshold | 5.0 s | Above this value there is no longitudinal collision risk |
| Lateral TTC threshold | 2.0 s | Above this value there is no lateral collision risk |
| Consider vehicle size | False | Computed with a point model |
| Size buffer | 0.5 m | Safety margin |

### 2.2 TTC Filtering Results

| Metric | Result |
|------|------|
| Input scenes | 803 |
| Scenes with dangerous vehicles | **564** |
| Filtering success rate | 70.2% |
| Total original vehicles | 2460 |
| Dangerous vehicles (kept) | **1813** |
| Vehicle retention rate | 73.7% |
| danger_pairs saved | **564** |

### 2.3 Distribution of Selection Reasons

| Selection reason | Scenes | Share |
|---------|--------|------|
| Front_agent_min_lon_ttc | 229 | 40.6% |
| Dual_risk_front_agent | 194 | 34.4% |
| Rear_agent_min_ttc | 141 | 25.0% |

**Notes**:
- `Front_agent_min_lon_ttc`: scenes where the front vehicle has the minimum longitudinal TTC
- `Dual_risk_front_agent`: the front vehicle has both longitudinal and lateral risk
- `Rear_agent_min_ttc`: scenes where the rear vehicle has the minimum TTC

### 2.4 Distribution of Dangerous Vehicle Counts

| Dangerous vehicles | Scenes | Share |
|-----------|--------|------|
| 1 vehicle | 157 | 27.8% |
| 2 vehicles | 146 | 25.9% |
| 3 vehicles | 95 | 16.8% |
| 4-5 vehicles | 83 | 14.7% |
| 6-10 vehicles | 64 | 11.3% |
| 11-15 vehicles | 19 | 3.4% |

### 2.5 danger_pair Data Format

Each `danger_pair_features_*.pkl` file contains:

| Field | Shape | Description |
|------|------|------|
| ego_history | (40, 5) | 4-second history trajectory |
| key_agent_history | (40, 5) | History trajectory of the key dangerous agent |
| ego_future_gt | (60, 5) | 6-second future trajectory ground truth |
| key_agent_future | (60, 3) or None | Agent future trajectory |
| key_agent_lon_ttc | array | Longitudinal TTC sequence |
| key_agent_lat_ttc | array | Lateral TTC sequence |
| boundary_agents | list | List of other dangerous agents |
| time_window_config | dict | Time-window configuration |

---

## 3. File Structure

```
diffusion_danger_model/
├── config.py                          # Configuration file
├── process_pittsburgh.py              # Batch processing script
├── processed_data/                    # Raw processed data
│   ├── features_*.pkl                 # 803 scene files
│   └── processing_report_*.md         # Processing report
├── danger_pair_scenarios/             # Danger scenes after TTC filtering
│   ├── danger_pair_features_*.pkl     # 564 danger scene files
│   └── ttc_filter_report_*.md         # TTC filtering report
└── ttc_filter_results/                # TTC filtering logs
    └── ttc_filter_batch_*.txt
```

---

## 4. Key Findings

### 4.1 Data Characteristics

1. **Time-window change**: changed from the original 2 s history + 8 s future to 4 s history + 6 s future, which is better suited to short-term danger scene prediction
2. **Vehicle distribution**: about 20% of scenes have no surrounding vehicles, and 45% of scenes have 3 or more vehicles
3. **Danger scene proportion**: 70% of scenes contain at least one dangerous vehicle (TTC below threshold)

### 4.2 Distribution of Risk Types

- **Mainly longitudinal risk**: 40.6% of scenes are triggered by the minimum longitudinal TTC of the front vehicle
- **Dual risk**: in 34.4% of scenes the front vehicle has both longitudinal and lateral risk
- **Rear risk**: 25% of scenes are triggered by the rear vehicle

---

## 5. Recommendations for Subsequent Use

### 5.1 Diffusion Model Training

The 564 filtered danger_pair samples can be used directly for training:

```bash
# Example training command
python train_diffusion_generator.py \
    --data_dir diffusion_danger_model/danger_pair_scenarios \
    --epochs 100 \
    --batch_size 32
```

### 5.2 Data Augmentation

The dataset can be extended in the following ways:
- Widen the TTC threshold range (currently only high-risk scenes are kept)
- Process other NuPlan datasets (e.g. Boston)
- Adjust MIN_DISPLACEMENT to obtain more low-speed scenes

---

## 6. Code Fix Record

The following issues were fixed during the project:

| Issue | File | Solution |
|------|------|----------|
| Wrong function definition order | ttc_filter.py | Moved `generate_ttc_report` before `main()` |
| f-string syntax error | Multiple places | Fixed the `print(f'{'='*60}')` formatting problem |
| Wrong scene type name | config.py | Corrected `near_multiple_vehicle` to `near_multiple_vehicles` |

---

**End of report**

---

## Appendix: Related File Paths

- Data processing script: [process_pittsburgh.py](diffusion_danger_model/process_pittsburgh.py)
- Configuration file: [config.py](diffusion_danger_model/config.py)
- TTC filtering script: [ttc_filter.py](GNN_Trans/utils/ttc_filter.py)
- Processing report: [processing_report_20260414_162359.md](diffusion_danger_model/processed_data/processing_report_20260414_162359.md)
- TTC report: [ttc_filter_report_20260414_162458.md](diffusion_danger_model/danger_pair_scenarios/ttc_filter_report_20260414_162458.md)