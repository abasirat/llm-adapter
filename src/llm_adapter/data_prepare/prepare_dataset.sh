#!/bin/bash

#input_file=dr_articles.txt

dataset_name=wikimedia/wikipedia
dataset_config=20231101.da
split=train

model_name=gpt2
tokenizer_name=gpt2
tokenizer_save_path=../models/gpt2_wechsel_wikimedia-wikipedia_en-da_tokenizer
output_prefix=gpt2_wechsel_wikimedia-wikipedia_en-da_tokenized

script=prepare_dataset.py
python $script --dataset_name $dataset_name \
    --dataset_config $dataset_config \
    --split $split \
    --output_prefix $output_prefix \
    --tokenizer_name $tokenizer_name \
    --model_name $model_name \
    --wechsel_src_lang en \
    --wechsel_tgt_lang da \
    --wechsel_dictionary danish \
    --save_tokenizer $tokenizer_save_path
