input_models_dir=/Users/dsb322/Lab/llm-adapter/models

gpt2=gpt2

gpt2_no_reg=gpt2_pile-of-law_layer_adapter_nal-1_ntl0_npt0_freezeLMheads1_kq32_vnone_nh4_t2.0_aggReppre_mlp_aggQueryfinal_hidden_aggStrategyattention_lowrank_variational0_shiftReg0_lr1e-4.pt
gpt2_sh_reg=gpt2_pile-of-law_layer_adapter_nal-1_ntl0_npt0_freezeLMheads1_kq32_vnone_nh4_t2.0_aggReppre_mlp_aggQueryfinal_hidden_aggStrategyattention_lowrank_variational0_shiftReg1_lr1e-4.pt
gpt2_sh_var=gpt2_pile-of-law_layer_adapter_nal-1_ntl0_npt0_freezeLMheads1_kq32_vnone_nh4_t2.0_aggReppre_mlp_aggQueryfinal_hidden_aggStrategyattention_lowrank_variational1_shiftReg0_lr1e-4.pt
gpt2_sh_varreg=gpt2_pile-of-law_layer_adapter_nal-1_ntl0_npt0_freezeLMheads1_kq32_vnone_nh4_t2.0_aggReppre_mlp_aggQueryfinal_hidden_aggStrategyattention_lowrank_variational1_shiftReg1_lr1e-4.pt

gpt2_lora=gpt2-large_pile-of-law_lora_r4_alpha16_ntl0_freezeLMheads1_lr1e-4.pt-best



output_models_dir=/Users/dsb322/Lab/llm-adapter/models/legal_reasoners

python -m casehold_continuation.train \
    --model_name_or_path ${input_models_dir}/$gpt2_no_reg \
    --output_dir ${output_models_dir}/casehold_continuation \
    --max_length 1024 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --max_train_examples 10000 \
    --device mps \