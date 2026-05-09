#!/bin/bash


model_prefix=/Users/dsb322/Lab/llm-adapter/models

gpt2=gpt2

gpt2_no_reg=gpt2_pile-of-law_layer_adapter_nal-1_ntl0_npt0_freezeLMheads1_kq32_vnone_nh4_t2.0_aggReppre_mlp_aggQueryfinal_hidden_aggStrategyattention_lowrank_variational0_shiftReg0_lr1e-4.pt
gpt2_sh_reg=gpt2_pile-of-law_layer_adapter_nal-1_ntl0_npt0_freezeLMheads1_kq32_vnone_nh4_t2.0_aggReppre_mlp_aggQueryfinal_hidden_aggStrategyattention_lowrank_variational0_shiftReg1_lr1e-4.pt
gpt2_sh_var=gpt2_pile-of-law_layer_adapter_nal-1_ntl0_npt0_freezeLMheads1_kq32_vnone_nh4_t2.0_aggReppre_mlp_aggQueryfinal_hidden_aggStrategyattention_lowrank_variational1_shiftReg0_lr1e-4.pt
gpt2_sh_varreg=gpt2_pile-of-law_layer_adapter_nal-1_ntl0_npt0_freezeLMheads1_kq32_vnone_nh4_t2.0_aggReppre_mlp_aggQueryfinal_hidden_aggStrategyattention_lowrank_variational1_shiftReg1_lr1e-4.pt

gpt2_lora=gpt2-large_pile-of-law_lora_r4_alpha16_ntl0_freezeLMheads1_lr1e-4.pt-best

style_classifier_model=${model_prefix}/legal_style_clf.joblib

models=(
    "$gpt2"
    "$gpt2_no_reg"
    "$gpt2_sh_reg"
    "$gpt2_sh_var"
    "$gpt2_sh_varreg"
    "$gpt2_lora"
)

# Run all evaluations
output_files=()
for model_name in "${models[@]}"; do
    CMD="python run_all_evaluations.py --device mps"
    if [ "$model_name" == "gpt2" ]; then
            CMD+=" --model_name_or_path ${model_name}"
    else
            CMD+=" --model_name_or_path ${model_prefix}/${model_name}"
    fi
    CMD+=" --style_min_new_tokens 20 --style_max_new_tokens 150 --style_clf_path ${style_classifier_model}"
    CMD+=" --max_examples 1000"
    CMD+=" --skip perplexity"
    CMD+=" --output_file ${model_prefix}/${model_name}_eval.json"
    CMD+=" --legal_prompts_path evaluations/style/legal_prompts/"
    CMD+=" --general_prompts_path evaluations/style/general_prompts/"
    # if the classification models are not already trained and saved, this will train them and save to the specified path
    if [ ! -f "$style_classifier_model" ]; then
        CMD+=" --train_style_clf --style_clf_max_samples 10000 --style_clf_output_model ${style_classifier_model}"
    fi

    echo "Running evaluation for $model_name..."
    echo "Command: $CMD"
    eval $CMD
    output_files+=("${model_prefix}/${model_name}_eval.json")
done

# Combine and display all results in a single table
echo ""
echo "Combining results..."
python combine_eval_results.py "${output_files[@]}"