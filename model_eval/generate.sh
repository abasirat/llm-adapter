#!/bin/bash

conda activate YOUR_ENV_NAME

model_path=/path/to/your/model.pt
tokenizer_path=/path/to/your/tokenizer

prompt_file=/path/to/your/prompts.txt
output_file=/path/to/save/generated_outputs.json

python generate.py \
    --model_path $model_path \
    --tokenizer_path $tokenizer_path \
    --prompt_file $prompt_file \
    --output_file $output_file