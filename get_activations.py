import os
import torch
import argparse
import transformers
import numpy as np
import json
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from pyreft import (
    get_reft_model,
    ReftConfig,
    ReftDataCollator,
    ReftSupervisedDataset
)
from SvdNULLIntervention import ReftSvdNULLTrainerForCausalLM
from utils.utils import CustomTrainingArguments
from data_process import (
    preprocess_truthfulqa_for_reft,
    preprocess_bbq_for_reft,
    preprocess_refusal_for_reft,
    preprocess_alpaca_for_reft,
    preprocess_helpsteer_for_reft1,
    preprocess_helpsteer_for_reft2,
    preprocess_helpsteer_for_reft3
)
import random
import gc
from utils.utils import get_llama_activations_bau

# Preprocessing functions (Data processing)
def preprocess_and_get_prompts(dataset, tokenizer, dataset_type):
    """
    Generalized preprocessing for different datasets.
    It processes the data and returns tokenized prompts.
    """
    all_prompts = []
    if dataset_type == "truthful_and_bbq":
        # Preprocess TruthfulQA and BBQ dataset for prompts.
        for example in dataset:
            prompt = f"Please respond to the following statement, and do not output any unnecessary content:  {example['instruction']} Response: {example['output']}"
            all_prompts.append(prompt)

    elif dataset_type == "alpaca_and_refusal":
        # Preprocess Alpaca and Refusal dataset for prompts.
        for example in dataset:
            prompt = f"Please follow the instruction to give the response:###Instruction: {example['instructiion']} ###Response: {example['output']}"
            all_prompts.append(prompt)

    elif dataset_type == "helpsteer":
        # Preprocess Helpsteer dataset for different help, coherence, verbosity categories
        for example in dataset:
            prompt = f"Helpfulness: {example['helpfulness']} Coherence: {example['coherence']} Verbosity: {example['verbosity']}"
            all_prompts.append(prompt)
    
    return all_prompts


def load_and_process_datasets(args, tokenizer):
    """
    Load and preprocess datasets based on user input.
    This function returns a dictionary of prompts, with dataset names as keys.
    """
    # Initialize the dictionary to hold prompts for each dataset
    prompts_dict = {}

    if args.dataset_type == "truthful_and_bbq":
        # Load and process TruthfulQA and BBQ dataset
        truthful_dataset = load_dataset(args.truthfulqa_path)
        truthfulqa_dataset = truthful_dataset.map(preprocess_truthfulqa_for_reft, batched=True, remove_columns=truthful_dataset["train"].column_names)
        
        bbq_dataset_raw = load_dataset(args.bbq_path)
        bbq_dataset = bbq_dataset_raw.map(preprocess_bbq_for_reft)
        
        # Balancing the datasets
        total_samples = len(truthfulqa_dataset["train"])
        truthfulqa_dataset = truthfulqa_dataset["train"]
        subsets = list(bbq_dataset.keys())
        samples_per_subset = total_samples // len(subsets)
        
        sampled_splits = {}
        for subset in subsets:
            ds = bbq_dataset[subset]
            n = min(len(ds), samples_per_subset)  
            sampled_ds = ds.shuffle(seed=42).select(range(n))
            sampled_splits[subset] = sampled_ds
        bbq_dataset = concatenate_datasets(list(sampled_splits.values()))
        
        # Process prompts for both datasets and store in the dictionary
        prompts_dict["truthful"] = preprocess_and_get_prompts(truthfulqa_dataset, tokenizer, "truthful_and_bbq")
        prompts_dict["bbq"] = preprocess_and_get_prompts(bbq_dataset, tokenizer, "truthful_and_bbq")

    elif args.dataset_type == "alpaca_and_refusal":
        # Load and process Alpaca and Refusal dataset
        alpaca_raw = load_dataset(args.alpaca_path)
        alpaca_dataset = alpaca_raw.map(preprocess_alpaca_for_reft)
        
        refusal_raw = load_dataset(args.refusal_path)
        refusal_dataset = refusal_raw.map(preprocess_refusal_for_reft)
        refusal_dataset = refusal_dataset['train']

        # Ensure equal dataset length by truncating the larger one to the size of the smaller one
        shuffled = alpaca_dataset['train'].shuffle(seed=42)
        alpaca_dataset = shuffled.select(range(len(refusal_dataset)))

        # Process prompts for both datasets and store in the dictionary
        prompts_dict["alpaca"] = preprocess_and_get_prompts(alpaca_dataset, tokenizer, "alpaca_and_refusal")
        prompts_dict["refusal"] = preprocess_and_get_prompts(refusal_dataset, tokenizer, "alpaca_and_refusal")
        
    elif args.dataset_type == "helpsteer":
        # Load and process Helpsteer dataset
        raw_dataset = load_dataset(args.helpsteer_path)
        train_ds = raw_dataset['train']
        
        # Filter based on specific attributes
        filtered_ds = train_ds.filter(lambda ex: len(ex["prompt"]) < 800 and len(ex["response"]) < 512)
        help_data = filtered_ds.filter(lambda ex: ex["helpfulness"] > 2)
        coher_data = filtered_ds.filter(lambda ex: ex["coherence"] > 3)
        verb_data = filtered_ds.filter(lambda ex: ex["verbosity"] < 2)
        
        data_list = [help_data, coher_data, verb_data]
        sampled_prompts = []
        
        for i, raw_dataset in enumerate(data_list):
            if i == 0:
                dataset = raw_dataset.map(preprocess_helpsteer_for_reft1)
            elif i == 1:
                dataset = raw_dataset.map(preprocess_helpsteer_for_reft2)
            else:
                dataset = raw_dataset.map(preprocess_helpsteer_for_reft3)

            sampled_prompts.append(dataset)
        
        # Store the processed prompts in the dictionary
        prompts_dict["helpfulness"] = preprocess_and_get_prompts(sampled_prompts[0], tokenizer, "helpsteer")
        prompts_dict["coherence"] = preprocess_and_get_prompts(sampled_prompts[1] , tokenizer, "helpsteer")
        prompts_dict["verbosity"] = preprocess_and_get_prompts(sampled_prompts[2], tokenizer, "helpsteer")
        
    else:
        raise ValueError("Invalid dataset name")

    return prompts_dict


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformers.set_seed(42)
    
    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, model_max_length=512, padding_side="right", use_fast=False)
    #tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token = tokenizer.eos_token
    # Load and preprocess dataset
    prompts_dict = load_and_process_datasets(args, tokenizer)

    # Process each dataset and get activations
    all_layer_wise_activations = []
    for dataset_name, prompts in prompts_dict.items():
        print(f"Processing dataset: {dataset_name}")
        
        for prompt in tqdm(prompts):
            # Get activations for the prompt
            prompt = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).input_ids
            layer_wise_activations = get_llama_activations_bau(model, prompt, device)
            layer_wise_activations_wanted = layer_wise_activations[:, -1, :].copy()
            del layer_wise_activations
            all_layer_wise_activations.append(layer_wise_activations_wanted)

        # Save the activations for each dataset
        print(f"Saving layer-wise activations for {dataset_name}")
        np.save(f'features/{args.model_name}_{dataset_name}_layer_wise.npy', all_layer_wise_activations)
        all_layer_wise_activations.clear()  # Reset the list for the next dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get activations!")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["truthful_and_bbq", "alpaca_and_refusal", "helpsteer"])
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--truthfulqa_path", type=str, required=False)
    parser.add_argument("--bbq_path", type=str, required=False)
    parser.add_argument("--alpaca_path", type=str, required=False)
    parser.add_argument("--refusal_path", type=str, required=False)
    parser.add_argument("--helpsteer_path", type=str, required=False)

    args = parser.parse_args()
    main(args)