#!/bin/bash

# Activate Conda
#source ~/miniconda3/etc/profile.d/conda.sh

# Activate the specific environment
conda activate YOUR_CONDA_ENV

echo "Conda environment activated: $(conda env list | grep '*' | awk '{print $1}')"

wandb login YOUR_WANDB_API_KEY
echo "wandb [Ok]"

model_prefix=path/to/models/
data_prefix=/path/to/data/
llm=gpt2
adapter_type=layer_adapter # layer_adapter, none, or lora

train_token_bin=gpt2_wechsel_wikimedia-wikipedia_en-da_tokenized_train.bin
model_name=${llm}_wikimedia-wikipedia_en-da_${adapter_type}.pt
tokenizer_name=gpt2_wechsel_wikimedia-wikipedia_en-da_tokenizer

path_to_train_data=${data_prefix}/${train_token_bin}
path_to_model=${model_prefix}/${model_name}
path_to_tokenizer=${model_prefix}/${tokenizer_name}
num_tailor_layers=0
num_epochs=1
chkpt=""

proj_name="language_adapt"

# extract the first five letters
first_five=${train_token_bin:0:5}
experiment_description=${llm}_${first_five}-${adapter_type}

chunk_size=$((1024 * 1024))
batch_size=4

script=language_adapter.py

# Train with file-based data source
python $script \
    --model_name $llm \
    --model_path $path_to_model \
    --tokenizer_path $path_to_tokenizer \
    --num_tailor_layers $num_tailor_layers \
    --adapter_type $adapter_type \
    --num_epochs 1 \
    --proj_name $proj_name \
    --experiment_description $experiment_description \
    --token_bin $path_to_train_data \
    --batch_size $batch_size
 
 conda deactivate
