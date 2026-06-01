#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans"
PYTHON_BIN="/home/team/anaconda3/envs/nuplan/bin/python"
DATA_DIR="${ROOT_DIR}/data/processed_data_by_type/processed_medium_magnitude_speed"
RESULTS_DIR="${ROOT_DIR}/experiments/baselines/lstm"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${RESULTS_DIR}/lstm_train_eval_${RUN_TAG}.log"
MODEL_PATH="${RESULTS_DIR}/trained_lstm_model.pth"
EVAL100_JSON="${RESULTS_DIR}/lstm_eval_100_${RUN_TAG}.json"
EVAL500_JSON="${RESULTS_DIR}/lstm_eval_500_${RUN_TAG}.json"

mkdir -p "${RESULTS_DIR}"

exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[INFO] Run tag: ${RUN_TAG}"
echo "[INFO] Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Data dir: ${DATA_DIR}"
echo "[INFO] Training script: ${ROOT_DIR}/train_lstm.py"
echo "[INFO] Model path: ${MODEL_PATH}"
echo "[INFO] Log path: ${LOG_PATH}"

cd "${ROOT_DIR}"

MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp "${PYTHON_BIN}" train_lstm.py

echo "[INFO] Training finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Starting 100-sample evaluation"
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp "${PYTHON_BIN}" run_lstm_pkl_eval.py \
  --model_path "${MODEL_PATH}" \
  --data_dir "${DATA_DIR}" \
  --num_samples 100 \
  --device cpu \
  --output_json "${EVAL100_JSON}"

echo "[INFO] Starting 500-sample evaluation"
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp "${PYTHON_BIN}" run_lstm_pkl_eval.py \
  --model_path "${MODEL_PATH}" \
  --data_dir "${DATA_DIR}" \
  --num_samples 500 \
  --device cpu \
  --output_json "${EVAL500_JSON}"

echo "[INFO] All tasks finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Evaluation 100 json: ${EVAL100_JSON}"
echo "[INFO] Evaluation 500 json: ${EVAL500_JSON}"
