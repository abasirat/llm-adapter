#!/bin/bash

# Activate Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the specific environment
conda activate torchEnv

echo "Conda environment activated: $(conda env list | grep '*' | awk '{print $1}')"

wandb login YOUR-WANDB-KEY

model_prefix=./models/
data_prefix=./data/
llm=gpt2
adapter_type=lora #layer_adapter # none or lora

train_data=dr_articles.txt
model_name=${llm}_$(basename $train_data .txt)_${adapter_type}.pt

path_to_train_data=${data_prefix}/${train_data}
path_to_model=${model_prefix}/${model_name}
num_tailor_layers=1
load_model=1
num_epochs=1
chkpt=""

proj_name="language_adapt"

# extract the first five letters
first_five=${train_data:0:5}

# extract the last three letters
last_five=${train_data: -5}
experiment_description=${llm}_${first_five}-${adapter_type}

python language_adapter.py \
    $llm \
    $path_to_train_data \
    $path_to_model \
    $num_tailor_layers \
    $adapter_type \
    "$chkpt" \
    $num_epochs \
    $proj_name \
    $experiment_description

conda deactivate