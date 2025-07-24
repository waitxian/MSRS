#!/bin/bash

cd "$(dirname "$0")/.."

python train.py --dataset_type truthful_and_bbq \
                --model_name_or_path /path/to/Llama3-8B-Instruct \
                --truthfulqa_path /path/to/dataset/TruthfulQA \
                --bbq_path /path/to/dataset/bbq \
                --pretrained_R /./features/llama3_truthful_bbq/final_result.npy \
                --indices "[0,1,2,3,4,5,8,9]" \
                --mask_prior_config '{"shared_rank": [0,3], "sub_rank1": [4,5], "sub_rank2": [6,7]}'\
                --target_layer 15 \
                --output_dir ./output/truthful_and_bbq
