#!/usr/bin/env bash
# Contrastive/ranking causal-LM training on LEDGAR.
#
# Run from the project root:
#   bash legal_reasoning/ledgar_contrastive/train.sh
#
# LEDGAR has 100 classes.  At training time --num_negatives incorrect labels
# are sampled per example (default 15 → 16-way ranking).
#
# Memory note: each example holds (num_negatives + 1) sequences.
# Start with --per_device_train_batch_size 1; increase gradient_accumulation_steps
# to achieve the desired effective batch size.

set -euo pipefail

input_models_dir=/Users/dsb322/Lab/llm-adapter/models
output_models_dir=/Users/dsb322/Lab/llm-adapter/models/legal_reasoners

# Model aliases (same names as trainer.sh / run_all_evaluations.sh)
gpt2_no_reg=gpt2_pile-of-law_layer_adapter_nal-1_ntl0_npt0_freezeLMheads1_kq32_vnone_nh4_t2.0_aggReppre_mlp_aggQueryfinal_hidden_aggStrategyattention_lowrank_variational0_shiftReg0_lr1e-4.pt
gpt2_sh_reg=gpt2_pile-of-law_layer_adapter_nal-1_ntl0_npt0_freezeLMheads1_kq32_vnone_nh4_t2.0_aggReppre_mlp_aggQueryfinal_hidden_aggStrategyattention_lowrank_variational0_shiftReg1_lr1e-4.pt
gpt2_sh_var=gpt2_pile-of-law_layer_adapter_nal-1_ntl0_npt0_freezeLMheads1_kq32_vnone_nh4_t2.0_aggReppre_mlp_aggQueryfinal_hidden_aggStrategyattention_lowrank_variational1_shiftReg0_lr1e-4.pt
gpt2_sh_varreg=gpt2_pile-of-law_layer_adapter_nal-1_ntl0_npt0_freezeLMheads1_kq32_vnone_nh4_t2.0_aggReppre_mlp_aggQueryfinal_hidden_aggStrategyattention_lowrank_variational1_shiftReg1_lr1e-4.pt

# ---------------------------------------------------------------------------
# Helper: run one model
# ---------------------------------------------------------------------------
run_model() {
    local model_name="$1"
    local model_path="$2"
    local ranking_loss="${3:-softmax}"
    local aux_ce_alpha="${4:-0.0}"
    local num_negatives="${5:-15}"

    local out_name
    out_name="ledgar_contrastive_$(basename "$model_name")_${ranking_loss}_neg${num_negatives}_alpha${aux_ce_alpha}"
    local output_dir="${output_models_dir}/${out_name}"

    echo "========================================================"
    echo "Model      : $model_name"
    echo "Ranking    : $ranking_loss  (negatives=${num_negatives}, aux_ce_alpha=${aux_ce_alpha})"
    echo "Output dir : $output_dir"
    echo "========================================================"

    python -m legal_reasoning.ledgar_contrastive.train \
        --model_name_or_path "$model_path" \
        --output_dir "$output_dir" \
        --ranking_loss "$ranking_loss" \
        --aux_ce_alpha "$aux_ce_alpha" \
        --num_negatives "$num_negatives" \
        --max_length 512 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 16 \
        --learning_rate 2e-5 \
        --warmup_ratio 0.05 \
        --weight_decay 0.01 \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 2 \
        --device mps \
        --seed 42
}

# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

# Baseline GPT-2
run_model "gpt2"          "gpt2"                                     softmax  0.0  15

# Best adapter — softmax with and without auxiliary CE
run_model "gpt2_no_reg"   "${input_models_dir}/${gpt2_no_reg}"      softmax  0.0  15
run_model "gpt2_no_reg"   "${input_models_dir}/${gpt2_no_reg}"      softmax  0.1  15

# Other adapters — softmax (primary comparison)
run_model "gpt2_sh_reg"   "${input_models_dir}/${gpt2_sh_reg}"      softmax  0.0  15
run_model "gpt2_sh_var"   "${input_models_dir}/${gpt2_sh_var}"      softmax  0.0  15
run_model "gpt2_sh_varreg" "${input_models_dir}/${gpt2_sh_varreg}"  softmax  0.0  15
