* **Environments**

```
conda create -n test python=3.10
git clone https://github.com/waitxian/MSRS.git
pip install -r requirements.txt
pip install pyreft einops 
pip install git+https://github.com/davidbau/baukit
```

* **Step 1: Obtain Model Activations**

  Run the following command to extract the activations for a given dataset:

```
#Example for TruthfulQA and BBQ datasets
python get_activations.py --dataset_type truthful_and_bbq --model_name /path/to/your/model --truthfulqa_path /path/to/TruthfulQA --bbq_path /path/to/bbq

# Example for Alpaca and Refusal datasets
python get_activations.py --dataset_type alpaca_and_refusal --model_name /path/to/your/model --alpaca_path /path/to/Alpaca --refusal_path /path/to/Refusal

# Example for HelpSteer dataset
python get_activations.py --dataset_type helpsteer --model_name /path/to/your/model --helpsteer_path /path/to/HelpSteer
```

* **Step 2: Perform SVD Decomposition on Activations**

  After you have obtained the activations, run the following command to perform SVD on the activations:

```
# Example for TruthfulQA and BBQ activations
python get_svd_rank.py --dataset_type truthful_and_bbq --model_name /path/to/your/model --truthfulqa_path /path/to/{args.model_name}_truthful_layer_wise.npy --bbq_path /path/to/{args.model_name}_bbq_layer_wise.npy

# Example for Alpaca and Refusal activations
python get_svd_rank.py --dataset_type alpaca_and_refusal --model_name /path/to/your/model --alpaca_path /path/to/{args.model_name}_alpaca_layer_wise.npy --refusal_path {args.model_name}_refusal_layer_wise.npy

# Example for HelpSteer activations
python get_svd_rank.py --dataset_type helpsteer --model_name /path/to/your/model --helpful_path /path/to/{model_name}_helpfulness_layer_wise.npy --coher_path /path/to/{model_name}_coherence_layer_wise.npy --verb_path /path/to/{model_name}_verbosity_layer_wise.npy
```

* **Step 3: Run MSRS**

  In this step, we use three different scripts depending on the dataset you are working with:

`truthful_and_bbq.sh`: For fine-tuning with the TruthfulQA and BBQ datasets.

`helpsteer.sh`: For fine-tuning with the HelpSteer dataset.

`alpaca_refusal.sh`: For fine-tuning with the Alpaca and Refusal datasets.