import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import argparse
import transformers
from datasets import load_dataset, concatenate_datasets
from pyreft import (
    get_reft_model,
    ReftConfig,
    ReftDataCollator,
    ReftSupervisedDataset
)
from SvdNULLIntervention import ReftSvdNULLTrainerForCausalLM
from data_process import preprocess_truthfulqa_for_reft,preprocess_bbq_for_reft,preprocess_refusal_for_reft, preprocess_alpaca_for_reft, preprocess_helpsteer_for_reft1, preprocess_helpsteer_for_reft2, preprocess_helpsteer_for_reft3
from utils.utils import CustomTrainingArguments
import json
import numpy as np
from SvdNULLIntervention import SubloreftIntervention

def main(args):

    mask_prior_config = json.loads(args.mask_prior_config)
    indices = json.loads(args.indices)
    # Set up device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformers.set_seed(42)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, model_max_length=512, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token

    # Prepare the dataset
    if args.dataset_type == "truthful_and_bbq":
        truthful_dataset = load_dataset(args.truthfulqa_path)
        truthfulqa_dataset = truthful_dataset.map(preprocess_truthfulqa_for_reft,batched=True,remove_columns=truthful_dataset["train"].column_names)
        
        bbq_dataset_raw = load_dataset(args.bbq_path)
        
        bbq_dataset = bbq_dataset_raw.map(preprocess_bbq_for_reft)
        
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
        
        dataset = concatenate_datasets([truthfulqa_dataset, bbq_dataset])

    elif args.dataset_type == "alpaca_and_refusal":
        alpaca_raw = load_dataset(args.alpaca_path)
        alpaca_dataset = alpaca_raw.map(preprocess_alpaca_for_reft)
        
        refusal_raw = load_dataset(args.refusal_path)
        refusal_dataset = refusal_raw.map(preprocess_refusal_for_reft)
        refusal_dataset = refusal_dataset['train']

        shuffled = alpaca_dataset['train'].shuffle(seed=42)
        alpaca_dataset = shuffled.select(range(len(refusal_dataset)))

        dataset = concatenate_datasets([alpaca_dataset, refusal_dataset])
        

    elif args.dataset_type == "helpsteer":
        raw_dataset = load_dataset(args.helpsteer_path)
        train_ds = raw_dataset['train']
        filtered_ds = train_ds.filter(
        lambda ex: len(ex["prompt"]) < 800 and len(ex["response"]) < 512
        )
        help_data = filtered_ds.filter(lambda ex: ex["helpfulness"] > 2)
        coher_data = filtered_ds.filter(lambda ex: ex["coherence"] > 3)
        verb_data = filtered_ds.filter(lambda ex: ex["verbosity"] < 2)
        data_list = [help_data,coher_data,verb_data]

        sampled_prompts = []
        for i in range(len(data_list)):
            raw_dataset = data_list[i]
            if i==0:
                dataset = raw_dataset.map(preprocess_helpsteer_for_reft1)
            elif i==1:
                dataset = raw_dataset.map(preprocess_helpsteer_for_reft2)
            else:
                dataset = raw_dataset.map(preprocess_helpsteer_for_reft3)

            sampled_prompts.append(dataset)
        dataset = concatenate_datasets([sampled_prompts[0],sampled_prompts[1],sampled_prompts[2]])
        
    pretrained_R_matrix = np.load(args.pretrained_R)
    pretrained_R_matrix = pretrained_R_matrix[:, indices, :]
    # Create ReFT Model Configuration
    reft_config = ReftConfig(
        representations={
            "layer": args.target_layer,
            "component": "block_output",
            "intervention": SubloreftIntervention(
                layer_no=args.target_layer,
                pretrained_R=pretrained_R_matrix,
                mask_prior_config=mask_prior_config,
                embed_dim=model.config.hidden_size,
                low_rank_dimension=args.rank
            )
        }
    )

    # Initialize ReFT Model
    reft_model = get_reft_model(model, reft_config)
    reft_model.print_trainable_parameters()

    # Create Training Dataset
    train_dataset = ReftSupervisedDataset(
        "Subloreft", None, tokenizer, dataset=dataset,
        **{"num_interventions": 1, "position": "l1", "share_weights": False},
        input_field="input", instruction_field="instruction", output_field="output",
        no_stop=True
    )

    # Data collator
    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest"
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)

    # Training arguments
    training_args = CustomTrainingArguments(
        num_train_epochs=4.0, output_dir=args.output_dir, learning_rate=args.learning_rate, report_to=[],
        per_device_train_batch_size=args.batchsize, logging_steps=200, reg_lambda=0.5, align_lambda=0.8, save_steps=1000
    )

    # Train
    trainer = ReftSvdNULLTrainerForCausalLM(
        model=reft_model, tokenizer=tokenizer, args=training_args, 
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MSRS!")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["truthful_and_bbq", "alpaca_and_refusal", "helpsteer"])
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--truthfulqa_path", type=str, required=False)
    parser.add_argument("--bbq_path", type=str, required=False)
    parser.add_argument("--alpaca_path", type=str, required=False)
    parser.add_argument("--refusal_path", type=str, required=False)
    parser.add_argument("--helpsteer_path", type=str, required=False)
    parser.add_argument("--pretrained_R", type=str, required=True)
    parser.add_argument("--mask_prior_config", type=str, required=True)
    parser.add_argument("--indices", type=str, required=True)
    parser.add_argument("--target_layer", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--batchsize", type=int, default=2)


    args = parser.parse_args()
    main(args)
