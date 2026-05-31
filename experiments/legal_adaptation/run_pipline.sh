#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="outputs/legal_pile_gpt2_adaptation"
TRAIN_DATA_CONFIG="configs/data/legal_pile_train.yaml"
VAL_DATA_CONFIG="configs/data/legal_pile_val.yaml"
TRAINING_CONFIG="configs/training/causal_lm.yaml"
EVAL_CONFIG="configs/evaluation/legal_eval.yaml"

MODELS=(
  "configs/models/gpt2.yaml"
)

ADAPTERS=(
  "configs/adapters/layer_adapter_attention.yaml"
)

data_prepare=false
train=true
evaluate=true

# data preparation
if [ "$data_prepare" = true ]; then
    echo "Preparing training dataset..."
    echo "Output prefix: ${EXPERIMENT_ROOT}/prepared_data/train/"
    python scripts/prepare_dataset.py \
        --config $TRAIN_DATA_CONFIG \
        --output_prefix ${EXPERIMENT_ROOT}/prepared_data/train/
    
    echo "Preparing validation dataset..."
    echo "Output prefix: ${EXPERIMENT_ROOT}/prepared_data/val"
    python scripts/prepare_dataset.py \
        --config $VAL_DATA_CONFIG \
        --output_prefix ${EXPERIMENT_ROOT}/prepared_data/val/
fi

BINARY_TOKENIZED_TRAIN_PATH="${EXPERIMENT_ROOT}/prepared_data/train/data.bin" 
BINARY_TOKENIZED_VAL_PATH="${EXPERIMENT_ROOT}/prepared_data/val/data.bin" 

SEEDS=(1)
MAX_PARALLEL=4  # adjust to how many jobs fit in your VRAM

run_experiment() {
  local MODEL_CONFIG="$1"
  local ADAPTER_CONFIG="$2"
  local SEED="$3"

  local MODEL_NAME
  MODEL_NAME=$(basename "$MODEL_CONFIG" .yaml)
  local ADAPTER_NAME
  ADAPTER_NAME=$(basename "$ADAPTER_CONFIG" .yaml)

  local RUN_NAME="${MODEL_NAME}_${ADAPTER_NAME}_seed${SEED}"
  local RUN_DIR="${EXPERIMENT_ROOT}/${RUN_NAME}"

  local CHECKPOINT_PATH="${RUN_DIR}/train/checkpoint"
  local TRAIN_HISTORY_PATH="${RUN_DIR}/train/history.jsonl"
  local EVAL_OUTPUT="${RUN_DIR}/eval/results.json"
  local TRAIN_STATUS_PATH="${RUN_DIR}/train/checkpoint.training_status.json"

  mkdir -p "${RUN_DIR}/train"
  mkdir -p "${RUN_DIR}/eval"

  local LOG_FILE="${RUN_DIR}/run.log"
  exec > >(tee -a "${LOG_FILE}") 2>&1

  echo "=================================================="
  echo "Running: ${RUN_NAME}"
  echo "Checkpoint: ${CHECKPOINT_PATH}"
  echo "Eval output: ${EVAL_OUTPUT}"
  echo "Train history: ${TRAIN_HISTORY_PATH}"
  echo "Train status: ${TRAIN_STATUS_PATH}"
  echo "=================================================="

  if [ "$train" = true ]; then
    echo "Training model for ${RUN_NAME}..."
    python scripts/train.py \
        --model_config "${MODEL_CONFIG}" \
        --adapter_config "${ADAPTER_CONFIG}" \
        --training_config "${TRAINING_CONFIG}" \
        --model_path "${CHECKPOINT_PATH}" \
        --train_bin_path "${BINARY_TOKENIZED_TRAIN_PATH}" \
        --history_path "${TRAIN_HISTORY_PATH}" \
        --experiment_name "${RUN_NAME}" \
        --seed "${SEED}" \
        --continue-training "${TRAIN_STATUS_PATH}"
  else
    echo "Skipping training for ${RUN_NAME}..."
  fi

  if [ "$evaluate" = true ]; then
    echo "Evaluating model for ${RUN_NAME}..."
    python scripts/eval.py \
      --config "${EVAL_CONFIG}" \
      --model_name_or_path "${CHECKPOINT_PATH}" \
      --output_file "${EVAL_OUTPUT}" \
      --seed "${SEED}"
  else
    echo "Skipping evaluation for ${RUN_NAME}..."
  fi
}

job_count=0

for MODEL_CONFIG in "${MODELS[@]}"; do
  for ADAPTER_CONFIG in "${ADAPTERS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        run_experiment "$MODEL_CONFIG" "$ADAPTER_CONFIG" "$SEED" &
        (( ++job_count ))
        if (( job_count >= MAX_PARALLEL )); then
          wait -n 2>/dev/null || wait
          (( --job_count ))
        fi
    done
  done
done

wait