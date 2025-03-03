from model import GPTConfig44_Complete, GPT
from data_utils import initialize_loaders
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

def load_model_and_extract_mlp_weights(config):
    """
    Load a trained model and extract MLP layer weights
    
    Args:
        model_path: Path to the saved model checkpoint (optional)
        device: Device to load the model onto
    
    Returns:
        model: The loaded model
        mlp_weights: Dictionary containing MLP weights from each layer
    """

    # Load the checkpoint
    checkpoint = torch.load(config.filename, weights_only = False)
    
    # Create the model architecture
    model = GPT(config).to(device)
    
    # Load the weights
    model.load_state_dict(checkpoint['model'])
    model.eval()  # Set to evaluation mode
    
    # Extract MLP weights from each layer
    mlp_weights = {}
    for i, block in enumerate(model.transformer.h):
        print(f"Layer {i}")
        layer_weights = {
            'c_fc': block.mlp.c_fc.weight.detach().cpu().numpy().T,  # Expansion weights
            'c_proj': block.mlp.c_proj.weight.detach().cpu().numpy().T  # Projection weights
        }
        if block.mlp.c_fc.bias is not None:
            layer_weights['c_fc_bias'] = block.mlp.c_fc.bias.detach().cpu().numpy()
        if block.mlp.c_proj.bias is not None:
            layer_weights['c_proj_bias'] = block.mlp.c_proj.bias.detach().cpu().numpy()
        
        mlp_weights[f'layer_{i}'] = layer_weights
    
    return model, mlp_weights

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import os

def visualize_mlp_weights(mlp_weights, save_dir):
    """
    Comprehensive visualization of MLP weight matrices
    
    Args:
        mlp_weights: Dictionary containing MLP weights
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for layer_name, layer_weights in mlp_weights.items():
        # Create a subdirectory for each layer
        layer_dir = os.path.join(save_dir, layer_name)
        os.makedirs(layer_dir, exist_ok=True)
        
        for weight_name, weights in layer_weights.items():
            # 1. Direct Heatmap Visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(weights, cmap='coolwarm', center=0)
            plt.title(f"{layer_name} - {weight_name} Weight Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(layer_dir, f"{weight_name}_heatmap.png"))
            plt.close()
            
            # 2. Row-wise and Column-wise Norms
            plt.figure(figsize=(12, 6))
            
            # Row norms (input dimensions)
            plt.subplot(1, 2, 1)
            row_norms = np.linalg.norm(weights, axis=1)
            plt.bar(range(len(row_norms)), row_norms)
            plt.title(f"Row Norms ({weights.shape[0]} input dimensions)")
            plt.xlabel("Input Dimension")
            plt.ylabel("L2 Norm")
            
            # Column norms (output dimensions)
            plt.subplot(1, 2, 2)
            col_norms = np.linalg.norm(weights, axis=0)
            plt.bar(range(len(col_norms)), col_norms)
            plt.title(f"Column Norms ({weights.shape[1]} output dimensions)")
            plt.xlabel("Output Dimension")
            plt.ylabel("L2 Norm")
            
            plt.tight_layout()
            plt.savefig(os.path.join(layer_dir, f"{weight_name}_norms.png"))
            plt.close()
            
            # 3. Singular Value Analysis
            u, s, vh = np.linalg.svd(weights, full_matrices=False)
            
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.plot(s, '-o', markersize=4)
            plt.title("Singular Values")
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(np.cumsum(s) / np.sum(s), '-o', markersize=4)
            plt.title("Cumulative Explained Variance")
            plt.xlabel("Number of Singular Values")
            plt.ylabel("Explained Variance Ratio")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(layer_dir, f"{weight_name}_svd.png"))
            plt.close()
            
            # 4. Weight Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(weights.flatten(), kde=True, bins=50)
            plt.title(f"{layer_name} - {weight_name} Weight Distribution")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(os.path.join(layer_dir, f"{weight_name}_distribution.png"))
            plt.close()
            
            # 5. PCA Visualization of Weights (for high-dimensional visualization)
            if weights.shape[0] > 2:  # Only if we have more than 2 dimensions
                # PCA on rows (input dimensions)
                pca = PCA(n_components=2)
                weights_pca = pca.fit_transform(weights)
                
                plt.figure(figsize=(10, 8))
                plt.scatter(weights_pca[:, 0], weights_pca[:, 1], alpha=0.7)
                plt.title(f"PCA of {weight_name} Rows (Input Dimensions)")
                plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
                plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
                plt.grid(True)
                plt.savefig(os.path.join(layer_dir, f"{weight_name}_pca_rows.png"))
                plt.close()
            
            if weights.shape[1] > 2:  # Only if we have more than 2 dimensions
                # PCA on columns (output dimensions)
                pca = PCA(n_components=2)
                weights_pca = pca.fit_transform(weights.T)
                
                plt.figure(figsize=(10, 8))
                plt.scatter(weights_pca[:, 0], weights_pca[:, 1], alpha=0.7)
                plt.title(f"PCA of {weight_name} Columns (Output Dimensions)")
                plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
                plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
                plt.grid(True)
                plt.savefig(os.path.join(layer_dir, f"{weight_name}_pca_cols.png"))
                plt.close()

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

def analyze_concept_neurons(model, data_loader, target_neurons):
    """
    Perform dataset traversal to find inputs that maximally and minimally activate specific neurons in layer 3.
    
    Parameters:
        model: The GPT model
        data_loader: DataLoader yielding batches of input sequences compatible with the model
        target_neurons: List of neuron indices to analyze (those with highest norms)
        device: Device to run the model on
    
    Returns:
        top_k_inputs: Dictionary mapping neuron indices to their top 10 activating inputs
        bottom_k_inputs: Dictionary mapping neuron indices to their bottom 10 activating inputs
    """
    model.eval()
    
    # Initialize tracking structures
    neuron_activations = {neuron: [] for neuron in target_neurons}
    top_k_inputs = {neuron: [] for neuron in target_neurons}  # Store top 10 activating inputs for each neuron
    bottom_k_inputs = {neuron: [] for neuron in target_neurons}  # Store bottom 10 activating inputs for each neuron
    
    # Process each batch in the data loader
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}/{len(data_loader)}")
        
        batch = batch.to(device)
        batch_size = batch.shape[0]
        
        # Forward pass with capture_layer=3 to get layer 3 activations
        with torch.no_grad():
            _, _, _, layer_3_activations, _ = model(batch, capture_layer=3)
            
            # For each sequence in the batch, extract final position's activations
            for seq_idx in range(batch_size):
                # Extract final position's activations for this sequence
                # Shape: [hidden_size]
                final_pos_activations = layer_3_activations[seq_idx, -1, :]
                
                # Check each target neuron's activation
                for neuron in target_neurons:
                    activation = final_pos_activations[neuron].item()
                    # Store (activation, batch_idx, seq_idx) to track which input caused this activation
                    neuron_activations[neuron].append((activation, batch_idx, seq_idx, batch[seq_idx].clone().cpu()))
    
    # Sort activations and keep top-k and bottom-k for each neuron
    for neuron in target_neurons:
        # Sort by activation value (descending for top-k)
        sorted_activations = sorted(neuron_activations[neuron], key=lambda x: x[0], reverse=True)
        
        # Store top 10 activations
        top_k = sorted_activations[:10]
        top_k_inputs[neuron] = [(act, input_seq) for act, _, _, input_seq in top_k]
        
        # Store bottom 10 activations
        bottom_k = sorted_activations[-10:]  # Get last 10 elements (lowest activations)
        bottom_k_inputs[neuron] = [(act, input_seq) for act, _, _, input_seq in bottom_k]
    
    return top_k_inputs, bottom_k_inputs

if __name__ == "__main__":
    # config = GPTConfig44_Complete()
    # print("Extracting mlp weights")
    # model, mlp_weights = load_model_and_extract_mlp_weights(config)
    # mlp_weights_save_path = f"{PATH_PREFIX}/mlp_triples_card_randomization_tuple_randomization_layers_4_heads_4.pt"
    # torch.save(mlp_weights, mlp_weights_save_path)

    # mlp_weights = torch.load(f"{PATH_PREFIX}/mlp_triples_card_randomization_tuple_randomization_layers_4_heads_4.pt")
    # visualize_mlp_weights(mlp_weights, f"COMPLETE_FIGS/mlp")

    config = GPTConfig44_Complete()
    # Load the checkpoint
    checkpoint = torch.load(config.filename, weights_only = False)
    
    # Create the model architecture
    model = GPT(config).to(device)
    
    # Load the weights
    model.load_state_dict(checkpoint['model'])
    model.eval()  # Set to evaluation mode

    dataset = torch.load(config.dataset_path)
    _, val_loader = initialize_loaders(config, dataset)

    target_neurons = [5, 13, 20, 36, 60]
    top_k_inputs, bottom_k_inputs = analyze_concept_neurons(model, val_loader, target_neurons)

    # Save top_k_inputs and bottom_k_inputs as pickle files
    with open(f"top_k_inputs.pkl", "wb") as f:
        pickle.dump(top_k_inputs, f)

    with open(f"bottom_k_inputs.pkl", "wb") as f:
        pickle.dump(bottom_k_inputs, f)

    print("Top-k Inputs:")
    for neuron, inputs in top_k_inputs.items():
        print(f"Neuron {neuron}:")
        for activation, input_seq in inputs:
            print(f"Activation: {activation}")
            print(f"Input Sequence: {input_seq}")
            print()
    
    breakpoint()


