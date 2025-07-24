#!/bin/bash

cd "$(dirname "$0")/.."

python train.py --dataset_type helpsteer \
                --model_name_or_path /path/to/Llama3-8B-Instruct \
                --helpsteer_path  /path/to/dataset/HelpSteer\
                --pretrained_R ./features/llama3_helpsteer/final_result.npy \
                --indices "[0,1,4,5,8,9,12,13]" \
                --learning_rate 9e-4 \
                --mask_prior_config '{"shared_rank": [0,1], "sub_rank1": [2,3], "sub_rank2": [4,5], "sub_rank3": [6,7]}'\
                --target_layer 9 \
                --output_dir ./output/helpsteer
