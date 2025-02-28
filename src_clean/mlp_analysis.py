from model import GPTConfig44_Complete, GPT
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


if __name__ == "__main__":
    config = GPTConfig44_Complete()
    print("Extracting mlp weights")
    model, mlp_weights = load_model_and_extract_mlp_weights(config)

    # mlp_weights = torch.load(f"{PATH_PREFIX}/mlp_triples_card_randomization_tuple_randomization_layers_4_heads_4.pt")
    breakpoint()
    # visualize_mlp_weights(mlp_weights, f"COMPLETE_FIGS/mlp")
