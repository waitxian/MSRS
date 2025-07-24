import os
import torch
import numpy as np
import json
import argparse
from datasets import load_dataset, concatenate_datasets
from scipy.linalg import svd
import gc
from tqdm import tqdm

# Dictionary to store results
svd_s_dict = {}
svd_Vh_dict = {}
svd_U_dict = {}

def svd_decomposition(layer_no, X, dataset_name, n_components=4):
    """
    Perform SVD decomposition on the matrix X and return the top n_components singular vectors.
    
    Parameters:
    -----------
    layer_no : int
        The layer number for which SVD is performed.
    X : numpy.ndarray
        The input matrix (activations or residuals).
    dataset_name : str
        The name of the dataset being processed (used for saving results).
    n_components : int
        The number of components to retain from the SVD decomposition.
        
    Returns:
    --------
    top_n_right_singular_vectors : numpy.ndarray
        The top n_components right singular vectors (Vh).
    """
    
    # Remove NaNs and Infs
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        X = X[np.isfinite(X).all(axis=1)]  # Remove NaNs and Infs
    
    X = X.astype(np.float32)
    # Perform SVD decomposition
    U, s, Vh = np.linalg.svd(X, full_matrices=True)

    # Save SVD results in dictionaries (global)
    key = f'Name:{dataset_name},L:{layer_no}'
    svd_s_dict[key] = s
    svd_Vh_dict[key] = Vh
    svd_U_dict[key] = U
    
    # Get top n singular values and vectors
    sorted_indices = np.argsort(s)[::-1]
    top_n_singular_values = s[sorted_indices[:n_components]]
    top_n_right_singular_vectors = Vh[sorted_indices[:n_components], :]

    return top_n_right_singular_vectors

def process_activations_and_svd(dataset_paths, n_components=4):
    activations_list = []
    dataset_names = []

    for name, path in dataset_paths.items():
        if os.path.isdir(path):
            raise ValueError(f"Expected .npy file path, got directory: {path}")
        activations = np.load(path, allow_pickle=True)
        activations_list.append(activations)
        dataset_names.append(name)

    num_layers = activations_list[0].shape[1]
    hidden_dim = activations_list[0].shape[2]

    for i, act in enumerate(activations_list):
        if act.shape[1] != num_layers or act.shape[2] != hidden_dim:
            raise ValueError(f"Inconsistent layer/hidden size in dataset {dataset_names[i]}.")

    combined_activations = np.concatenate(activations_list, axis=0)

    print("Processing shared subspace SVD for each layer:")
    for layer_no in tqdm(range(1, num_layers), desc="Shared SVD"):
        X = combined_activations[:, layer_no, :]
        Vh_combined = svd_decomposition(layer_no, X, 'combined', n_components=n_components)
        #svd_Vh_dict[f'Name:combined,L:{layer_no}'] = Vh_combined
        gc.collect()

def process_residuals_and_svd(dataset_paths, n_components=4):
    activations_dict = {}
    for dataset_name, path in dataset_paths.items():
        activations_dict[dataset_name] = np.load(path, allow_pickle=True)

    num_layers = activations_dict[list(dataset_paths.keys())[0]].shape[1]
    all_layerwise_results = []

    print("Processing residual SVD for each layer:")
    for layer_no in tqdm(range(1, num_layers), desc="Residual SVD"):
        combined_activations = []
        Vh_shared = svd_Vh_dict.get(f'Name:combined,L:{layer_no}')
        combined_activations.append(Vh_shared[:n_components, :])
        for dataset_name, activations in activations_dict.items():
            X = activations[:, layer_no, :]

            if Vh_shared is None:
                raise ValueError(f"Missing shared subspace for layer {layer_no}.")

            X_projected = project_onto_shared_subspace(X, Vh_shared)
            X_residual = X - X_projected @ Vh_shared.T

            Vh_res = svd_decomposition(layer_no, X_residual, dataset_name, n_components=n_components)
            
            combined_activations.append(Vh_res)

        combined_layer_result = np.concatenate(combined_activations, axis=0)
        all_layerwise_results.append(combined_layer_result)
        gc.collect()

    final_result = np.stack(all_layerwise_results, axis=0)
    return final_result



def project_onto_shared_subspace(X, Vh_shared):
    return X @ Vh_shared


def main(args):
    """
    Main function to load activations, perform SVD, and save results.
    """
    # Initialize paths to load activation files based on dataset type
    dataset_paths = {
        "truthfulqa": args.truthfulqa_path,
        "bbq": args.bbq_path,
        "alpaca": args.alpaca_path,
        "refusal": args.refusal_path,
        "helpful": args.helpful_path,
        "coherence":args.coher_path,
        "verbosity":args.verb_path
    }

    # Filter out None paths
    dataset_paths = {k: v for k, v in dataset_paths.items() if v is not None}
    process_activations_and_svd(dataset_paths, n_components=4)
    # Process activations and perform SVD decomposition
    final_result = process_residuals_and_svd(dataset_paths, n_components=4)

    # Save the final result with the dataset type in the filename
    np.save(f'features/{args.model_name}_{args.dataset_type}_final_result.npy', final_result)
    print(f"Final result saved as 'features/{args.model_name}_{args.dataset_type}_final_result.npy'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process activations and perform SVD!")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["truthful_and_bbq", "alpaca_and_refusal", "helpsteer"])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--truthfulqa_path", type=str, required=False)
    parser.add_argument("--bbq_path", type=str, required=False)
    parser.add_argument("--alpaca_path", type=str, required=False)
    parser.add_argument("--refusal_path", type=str, required=False)
    parser.add_argument("--helpful_path", type=str, required=False)
    parser.add_argument("--coher_path", type=str, required=False)
    parser.add_argument("--verb_path", type=str, required=False)

    args = parser.parse_args()
    main(args)
