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
train=false
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

for MODEL_CONFIG in "${MODELS[@]}"; do
  MODEL_NAME=$(basename "$MODEL_CONFIG" .yaml)

  for ADAPTER_CONFIG in "${ADAPTERS[@]}"; do
    ADAPTER_NAME=$(basename "$ADAPTER_CONFIG" .yaml)

    for SEED in "${SEEDS[@]}"; do

      RUN_NAME="${MODEL_NAME}_${ADAPTER_NAME}_seed${SEED}"
      RUN_DIR="${EXPERIMENT_ROOT}/${RUN_NAME}"

      CHECKPOINT_PATH="${RUN_DIR}/train/checkpoint"
      TRAIN_HISTORY_PATH="${RUN_DIR}/train/history.jsonl"
      EVAL_OUTPUT="${RUN_DIR}/eval/results.json"

      mkdir -p "${RUN_DIR}/train"
      mkdir -p "${RUN_DIR}/eval"

      echo "=================================================="
      echo "Running: ${RUN_NAME}"
      echo "Checkpoint: ${CHECKPOINT_PATH}"
      echo "Eval output: ${EVAL_OUTPUT}"
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
            --seed "${SEED}"
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

    done
  done
done