#!/bin/bash

cd "$(dirname "$0")/.."

python train.py --dataset_type alpaca_and_refusal \
                --model_name_or_path /path/to/Llama3-8B-Instruct \
                --alpaca_path  /path/to/dataset/alpaca\
		--refusal_path  /path/to/dataset/refusal\
                --pretrained_R ./features/llama3_alpaca_refusal/final_result.npy \
                --indices "[0,1,4,5,6,8,9,10]" \
                --learning_rate 3e-5 \
                --mask_prior_config '{"shared_rank": [0,1], "sub_rank1": [2,4], "sub_rank2": [5,7]}'\
                --target_layer 15 \
                --output_dir ./output/alpaca_and_refusal
