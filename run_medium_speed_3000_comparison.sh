#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans"
PYTHON_BIN="/home/team/anaconda3/envs/nuplan/bin/python"
DATA_DIR="${ROOT_DIR}/data/processed_data_by_type/processed_medium_magnitude_speed"
RESULTS_DIR="${ROOT_DIR}/experiments/analysis/comparisons"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULTS_DIR}/medium_speed_compare_${TIMESTAMP}"
MANIFEST_PATH="${RUN_DIR}/medium_speed_manifest_3000.json"
MASTER_LOG="${RUN_DIR}/master.log"

mkdir -p "${RUN_DIR}"

exec >> "${MASTER_LOG}" 2>&1

echo "[INFO] Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Run dir: ${RUN_DIR}"
echo "[INFO] Data dir: ${DATA_DIR}"

cd "${ROOT_DIR}"

echo "[INFO] Generating 3000-sample manifest..."
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp "${PYTHON_BIN}" generate_medium_speed_manifest.py \
  --data_dir "${DATA_DIR}" \
  --sample_count 3000 \
  --seed 42 \
  --output_json "${MANIFEST_PATH}"

echo "[INFO] Manifest ready: ${MANIFEST_PATH}"

echo "[INFO] Launching GNN-Trans evaluation on GPU 2"
CUDA_VISIBLE_DEVICES=2 MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp "${PYTHON_BIN}" run_gnn_manifest_eval.py \
  --model_path "${ROOT_DIR}/results_type_medium_magnitude_speed_nearby_only_adjacency/training_20251002_124458_ds18000_bs8_lr5e-04/models/best_model_nearby_only.pth" \
  --data_dir "${DATA_DIR}" \
  --manifest "${MANIFEST_PATH}" \
  --adjacency_config nearby_only \
  --device cuda:0 \
  --output_json "${RUN_DIR}/gnn_nearby_only_3000.json" \
  > "${RUN_DIR}/gnn_eval.log" 2>&1 &
GNN_PID=$!

echo "[INFO] Launching LSTM evaluation on GPU 0"
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp "${PYTHON_BIN}" run_lstm_pkl_eval.py \
  --model_path "${ROOT_DIR}/experiments/baselines/lstm/trained_lstm_model.pth" \
  --data_dir "${DATA_DIR}" \
  --manifest "${MANIFEST_PATH}" \
  --num_samples 3000 \
  --device cuda:0 \
  --output_json "${RUN_DIR}/lstm_3000.json" \
  > "${RUN_DIR}/lstm_eval.log" 2>&1 &
LSTM_PID=$!

echo "[INFO] Launching IDM evaluation on CPU"
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp "${PYTHON_BIN}" run_idm_pkl_eval.py \
  --data_dir "${DATA_DIR}" \
  --manifest "${MANIFEST_PATH}" \
  --num_samples 3000 \
  --output_json "${RUN_DIR}/idm_3000.json" \
  > "${RUN_DIR}/idm_eval.log" 2>&1 &
IDM_PID=$!

echo "[INFO] GNN PID: ${GNN_PID}"
echo "[INFO] LSTM PID: ${LSTM_PID}"
echo "[INFO] IDM PID: ${IDM_PID}"

wait "${GNN_PID}"
echo "[INFO] GNN evaluation finished"
wait "${LSTM_PID}"
echo "[INFO] LSTM evaluation finished"
wait "${IDM_PID}"
echo "[INFO] IDM evaluation finished"

echo "[INFO] All comparisons finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Outputs:"
echo "  - ${RUN_DIR}/gnn_nearby_only_3000.json"
echo "  - ${RUN_DIR}/lstm_3000.json"
echo "  - ${RUN_DIR}/idm_3000.json"
