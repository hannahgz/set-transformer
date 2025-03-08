import torch
from model import GPT, GPTConfig44_Complete
from data_utils import load_tokenizer
import random
import numpy as np
import os
from ablation import generate_heatmap_from_kl_matrix
from classify import plot_similarity_heatmap
import seaborn as sns
import matplotlib.pyplot as plt
from classify import load_linear_probe_from_config, load_continuous_to_original_from_config, LinearProbeBindingCardAttrConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

annot_font_size = 14
title_font_size = 16
label_font_size = 14

def embedding_ablation_kl_fig():
    ablate_type = "noise"
    matrix_path = f"results/ablation_study"
    os.makedirs(matrix_path, exist_ok=True)

    # Load the KL divergence matrix
    loaded_kl_matrix = np.load(os.path.join(
        matrix_path, f"avg_kl_divergence_matrix_ablate_type_{ablate_type}.npy"))

    fig = generate_heatmap_from_kl_matrix(
        loaded_kl_matrix, range(40), range(1, 5))
    fig_save_path = f"COMPLETE_FIGS/paper/ablation_study"
    os.makedirs(fig_save_path, exist_ok=True)
    fig.savefig(os.path.join(
        fig_save_path, f"avg_embedding_ablation_heatmap_ablate_type_{ablate_type}.png"), bbox_inches="tight")

def avg_combined_cosine_similarity_probe_embedding_heatmap():
    cards = ['A', 'B', 'C', 'D', 'E']
    num_layers = 4
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # Placeholder to collect data for shared color scale
    all_matrices = []

    # Load similarity matrices and determine global min/max for color scale
    for capture_layer in range(num_layers):
        similarity_matrix = np.load(f"{PATH_PREFIX}/complete/classify/avg_cosine_similarity_matrix_layer{capture_layer}.npy")
        all_matrices.append(similarity_matrix)
    
    global_min = np.min(all_matrices)
    global_max = np.max(all_matrices)

    # Flatten the axes array for easier iteration
    axes_flat = axes.flatten()

    # Plot each heatmap in a subplot
    for capture_layer, ax in enumerate(axes_flat[:num_layers]):
        sns.heatmap(
            all_matrices[capture_layer],
            xticklabels=cards,
            yticklabels=cards,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu',
            center=0,
            square=True,
            cbar=False,  # Disable individual colorbars
            vmin=global_min,  # Ensure consistent color scale across all heatmaps
            vmax=global_max,
            ax=ax,
            annot_kws={"fontsize": annot_font_size}
        )
        ax.set_title(f'Layer {capture_layer + 1}', fontsize=title_font_size)
        ax.set_xlabel('Probe Cards', fontsize=label_font_size)
        ax.set_ylabel('Attribute Embedding Cards', fontsize=label_font_size)
        ax.tick_params(axis='y', rotation=0, labelsize=label_font_size)  # Rotate y-tick labels to horizontal
        ax.tick_params(axis='x', labelsize=label_font_size)

    # Hide unused subplots
    for ax in axes_flat[num_layers:]:
        ax.set_visible(False)

    # Add a single colorbar for the whole figure
    cbar = fig.colorbar(axes_flat[0].collections[0], ax=axes, location='right', shrink=0.8, pad=0.02)
    cbar.set_label('Cosine Similarity', fontsize=title_font_size)
    cbar.ax.tick_params(labelsize=label_font_size)  # Set font size for colorbar tick labels
    # Remove the color bar's black outline
    cbar.outline.set_visible(False)
    
    # Add overarching title
    fig.suptitle('Cosine Similarities Between Linear Probe Weights and Model Embeddings', fontsize=title_font_size)
    
    # Save or show the combined figure
    save_path = f"COMPLETE_FIGS/paper/cosine_sim"
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"avg_combined_cosine_similarity_probe_embedding_heatmap.png"), bbox_inches="tight")

def combined_probe_weight_cosine_sim(model_config, probe_config):

    from sklearn.metrics.pairwise import cosine_similarity

    all_matrices = []
    for capture_layer in range(4):
        continuous_to_original = load_continuous_to_original_from_config(
            probe_config, capture_layer=capture_layer)
        tokenizer = load_tokenizer(model_config.tokenizer_path)

        cards = []
        for i in range(probe_config.output_dim):
            cards.append(tokenizer.decode([continuous_to_original[i]])[0])
        
        # Load probe and get weights
        probe = load_linear_probe_from_config(probe_config, capture_layer)
        probe_weights = probe.fc.weight.data.detach()  # Shape: [5, 64]
        
        # Convert to numpy array if it's a torch tensor
        if isinstance(probe_weights, torch.Tensor):
            probe_weights = probe_weights.numpy()
        
        # Calculate cosine similarity matrix
        cosine_sim_matrix = cosine_similarity(probe_weights)
        
        # Get sorting indices for alphabetical order
        sorted_indices = sorted(range(len(cards)), key=lambda k: cards[k])
        
        # Reorder the similarity matrix and cards
        cosine_sim_matrix = cosine_sim_matrix[sorted_indices][:, sorted_indices]
        all_matrices.append(cosine_sim_matrix)

    sorted_cards = sorted(cards)

    num_layers = 4  # Adjust the number of layers to plot (can change based on your use case)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    global_min = -1  # Fixed min for cosine similarity
    global_max = 1   # Fixed max for cosine similarity

    # Flatten the axes array for easier iteration
    axes_flat = axes.flatten()

    # Plot each heatmap in a subplot
    for capture_layer_idx, ax in enumerate(axes_flat[:num_layers]):
        sns.heatmap(
            all_matrices[capture_layer_idx],
            annot=True,
            cmap='RdBu_r',  # Red-Blue diverging colormap
            vmin=global_min,  # Standardized color scale min
            vmax=global_max,   # Standardized color scale max
            center=0,  # Center the colormap at 0
            square=True,  # Make cells square
            fmt='.2f',  # Format annotations to 2 decimal places
            xticklabels=sorted_cards,
            yticklabels=sorted_cards,
            cbar=False,  # Disable individual colorbars
            ax=ax,
            annot_kws={"fontsize": annot_font_size}
        )
        ax.set_title(f'Layer {capture_layer_idx + 1}', fontsize=title_font_size)
        ax.set_xlabel('Card for Neuron', fontsize=label_font_size)
        ax.set_ylabel('Card for Neuron', fontsize=label_font_size)
        ax.tick_params(axis='y', rotation=0, labelsize=label_font_size)  # Rotate y-tick labels to horizontal
        ax.tick_params(axis='x', labelsize=label_font_size)

    # Add a single colorbar for the whole figure
    cbar = fig.colorbar(axes_flat[0].collections[0], ax=axes, location='right', shrink=0.8, pad=0.02)
    cbar.set_label('Cosine Similarity', fontsize=title_font_size)
    cbar.ax.tick_params(labelsize=label_font_size)  # Set font size for colorbar tick labels
    
    # Remove the color bar's black outline
    cbar.outline.set_visible(False)
    
    # Add overarching title
    fig.suptitle(f'Cosine Similarities Between Probe Weight Vectors', fontsize=title_font_size)
    
    # Save or show the combined figure
    save_path = f"COMPLETE_FIGS/paper/cosine_sim"
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"combined_probe_weight_cosine_sim_layer{capture_layer}.png"), bbox_inches="tight")

    return fig


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # embedding_ablation_kl_fig()
    avg_combined_cosine_similarity_probe_embedding_heatmap()
    combined_probe_weight_cosine_sim(GPTConfig44_Complete(), LinearProbeBindingCardAttrConfig())
    
    # config = GPTConfig44_Complete()
    # checkpoint = torch.load(config.filename, weights_only=False)

    # # Create the model architecture
    # model = GPT(config).to(device)

    # # Load the weights
    # model.load_state_dict(checkpoint['model'])
    # model.eval()  # Set to evaluation mode

    # tokenizer = load_tokenizer(config.tokenizer_path)
