#!/bin/bash


model_prefix=/Users/dsb322/Lab/llm-adapter/models

gpt2_no_reg=gpt2_pile-of-law_layer_adapter_nal-1_ntl0_npt0_freezeLMheads1_kq32_vnone_nh4_t2.0_aggReppre_mlp_aggQueryfinal_hidden_aggStrategyattention_lowrank_variational0_shiftReg0_lr1e-4.pt
gpt2_sh_reg=gpt2_pile-of-law_layer_adapter_nal-1_ntl0_npt0_freezeLMheads1_kq32_vnone_nh4_t2.0_aggReppre_mlp_aggQueryfinal_hidden_aggStrategyattention_lowrank_variational0_shiftReg1_lr1e-4.pt
gpt2_sh_var=gpt2_pile-of-law_layer_adapter_nal-1_ntl0_npt0_freezeLMheads1_kq32_vnone_nh4_t2.0_aggReppre_mlp_aggQueryfinal_hidden_aggStrategyattention_lowrank_variational1_shiftReg0_lr1e-4.pt

gpt2_lora=gpt2-large_pile-of-law_lora_r4_alpha16_ntl0_freezeLMheads1_lr1e-4.pt-best

models=(
    #"$gpt2_no_reg"
    #"$gpt2_sh_reg"
    "$gpt2_sh_var"
    #"$gpt2_lora"
)

# Run all evaluations
for model_name in "${models[@]}"; do
    python run_all_evaluations.py \
        --model_name_or_path ${model_prefix}/${model_name} \
        --device mps \
        --legal_prompts_path evaluations/style/legal_prompts/ \
        --general_prompts_path evaluations/style/general_prompts/ \
        --style_min_new_tokens 20 \
        --style_max_new_tokens 150 \
        --max_examples 1000 \
        --output_file ${model_prefix}/${model_name}_eval.json \
        --skip perplexity
done