#!/usr/bin/env bash
# Contrastive/ranking causal-LM training on UNFAIR-ToS.
#
# Run from the project root:
#   bash legal_reasoning/unfair_tos_contrastive/train.sh
#
# UNFAIR-ToS has 8 unfairness categories.  Each clause is expanded into
# per-(clause, category) binary ranking instances (Yes vs. No).
# --num_negative_categories caps inactive-category instances per clause
# (default 4) to reduce the ~7:1 inactive/active imbalance.
#
# Memory note: each instance holds only 2 sequences (Yes and No), so
# per_device_train_batch_size can be larger than for LEDGAR/CaseHOLD.

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
    local ranking_loss="${3:-softmax}"    # softmax | hinge | logistic
    local aux_ce_alpha="${4:-0.0}"
    local num_neg="${5:-4}"               # inactive categories per clause

    local out_name
    out_name="unfair_tos_contrastive_$(basename "$model_name")_${ranking_loss}_neg${num_neg}_alpha${aux_ce_alpha}"
    local output_dir="${output_models_dir}/${out_name}"

    echo "========================================================"
    echo "Model      : $model_name"
    echo "Ranking    : $ranking_loss  (neg_categories=${num_neg}, aux_ce_alpha=${aux_ce_alpha})"
    echo "Output dir : $output_dir"
    echo "========================================================"

    python -m legal_reasoning.unfair_tos_contrastive.train \
        --model_name_or_path "$model_path" \
        --output_dir "$output_dir" \
        --ranking_loss "$ranking_loss" \
        --aux_ce_alpha "$aux_ce_alpha" \
        --num_negative_categories "$num_neg" \
        --max_length 512 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 \
        --warmup_ratio 0.05 \
        --weight_decay 0.01 \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 2 \
        --dataloader_num_workers 0 \
        --device mps \
        --seed 42
        # Add --fp16 (CUDA) or --bf16 (Ampere+ / Apple Silicon MPS) for mixed-precision speedup
}

# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

# Baseline GPT-2
#run_model "gpt2"          "gpt2"                                     softmax  0.0  4

# Best adapter — softmax with and without auxiliary CE
run_model "gpt2_no_reg"   "${input_models_dir}/${gpt2_no_reg}"      softmax  0.0  4
run_model "gpt2_no_reg"   "${input_models_dir}/${gpt2_no_reg}"      softmax  0.1  4

# Other adapters — softmax (primary comparison)
run_model "gpt2_sh_reg"   "${input_models_dir}/${gpt2_sh_reg}"      softmax  0.0  4
run_model "gpt2_sh_var"   "${input_models_dir}/${gpt2_sh_var}"      softmax  0.0  4
run_model "gpt2_sh_varreg" "${input_models_dir}/${gpt2_sh_varreg}"  softmax  0.0  4
