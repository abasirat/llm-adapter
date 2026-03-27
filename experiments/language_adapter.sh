#!/bin/bash

# Activate Conda
#source ~/miniconda3/etc/profile.d/conda.sh

# Activate the specific environment
conda activate YOUR_CONDA_ENV

echo "Conda environment activated: $(conda env list | grep '*' | awk '{print $1}')"

wandb login 5c7bc6a74f9ce4c1ff724611877a4937c29a475b
echo "wandb [Ok]"

model_prefix=path/to/models/
data_prefix=/path/to/data/
llm=gpt2
adapter_type=layer_adapter # layer_adapter, none, or lora

# Option 1: Train from file
<<<<<<< HEAD
train_data=dr_articles.txt
model_name=${llm}_$(basename $train_data .txt)_${adapter_type}.pt
tokenizer_name=${llm}_$(basename $train_data .txt)_tokenizer
=======
train_token_bin=gpt2_wechsel_wikimedia-wikipedia_en-da_tokenized_train.bin
model_name=${llm}_wikimedia-wikipedia_en-da_${adapter_type}.pt
tokenizer_name=gpt2_wechsel_wikimedia-wikipedia_en-da_tokenizer
>>>>>>> 9f79c03 (enable pre-tokenization. fix the issue with the tqdm progress when using pre-tokenized binary training data.  wechsel is integrated to the pre-tokenization)

path_to_train_data=${data_prefix}/${train_token_bin}
path_to_model=${model_prefix}/${model_name}
path_to_tokenizer=${model_prefix}/${tokenizer_name}
<<<<<<< HEAD

=======
>>>>>>> 9f79c03 (enable pre-tokenization. fix the issue with the tqdm progress when using pre-tokenized binary training data.  wechsel is integrated to the pre-tokenization)
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
<<<<<<< HEAD
#python $script \
#    $llm \
#    $path_to_model \
#    $num_tailor_layers \
#    $adapter_type \
#    "$chkpt" \
#    $num_epochs \
#    $proj_name \
#    $experiment_description \
#    $src_lang \
#    $tgt_lang \
#    $dictionary \
#    --train_data $path_to_train_data \
#    --chunk_size $chunk_size \
#    --batch_size $batch_size

# Option 2: Train from HuggingFace dataset (OSCAR - language-specific)
# Uses dataset_config to load language-specific data
echo "Begin processing ..."
python $script \
     $llm \
     $path_to_model \
     $path_to_tokenizer \
     $num_tailor_layers \
     $adapter_type \
     "$chkpt" \
     $num_epochs \
     $proj_name \
     $experiment_description \
     $src_lang \
     $tgt_lang \
     $dictionary \
     --dataset_name "wikimedia/wikipedia" \
     --dataset_config "20231101.${tgt_lang}" \
     --dataset_split train \
     --text_column text \
     --chunk_size $chunk_size \
     --batch_size $batch_size

# Option 3: Train from multilingual dataset with language filtering
# Uncomment to use with datasets that have language information
#python $script \
#     $llm \
#     $path_to_model \
#     $num_tailor_layers \
#     $adapter_type \
#     "$chkpt" \
#     $num_epochs \
#     $proj_name \
#     $experiment_description \
#     $src_lang \
#     $tgt_lang \
#     $dictionary \
#     --dataset_name wikitext-103 \
#     --dataset_split train \
#     --text_column text \
#     --language_filter $tgt_lang \
#     --chunk_size $chunk_size \
#     --batch_size $batch_size

conda deactivate
=======
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
>>>>>>> 9f79c03 (enable pre-tokenization. fix the issue with the tqdm progress when using pre-tokenized binary training data.  wechsel is integrated to the pre-tokenization)
