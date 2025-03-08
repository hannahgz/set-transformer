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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

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
    
# def avg_combined_cosine_similarity_probe_embedding_heatmap():
#     cards = ['A', 'B', 'C', 'D', 'E']
#     num_layers = 4
#     fig, axes = plt.subplots(1, num_layers, figsize=(16, 5), sharex=True, sharey=True, constrained_layout=True)

#     # Placeholder to collect data for shared color scale
#     all_matrices = []

#     # Load similarity matrices and determine global min/max for color scale
#     for capture_layer in range(num_layers):
#         similarity_matrix = np.load(f"{PATH_PREFIX}/complete/classify/avg_cosine_similarity_matrix_layer{capture_layer}.npy")
#         all_matrices.append(similarity_matrix)
    
#     global_min = np.min(all_matrices)
#     global_max = np.max(all_matrices)

#     # Plot each heatmap in a subplot
#     for capture_layer, ax in enumerate(axes):
#         sns.heatmap(
#             all_matrices[capture_layer],
#             xticklabels=cards,
#             yticklabels=cards,
#             annot=True,
#             fmt='.3f',
#             cmap='RdYlBu',
#             center=0,
#             square=True,
#             cbar=False,  # Disable individual colorbars
#             vmin=global_min,  # Ensure consistent color scale across all heatmaps
#             vmax=global_max,
#             ax=ax,
#             annot_kws={"fontsize": 14}
#         )
#         ax.set_title(f'Layer {capture_layer + 1}', fontsize=18)
#         ax.set_xlabel('Probe Cards', fontsize=18)
#         ax.set_ylabel('Attribute Embedding Cards', fontsize=18)

#     # Add a single colorbar for the whole figure
#     cbar = fig.colorbar(axes[0].collections[0], ax=axes, location='right', shrink=0.8, pad=0.02)
#     cbar.set_label('Cosine Similarity', fontsize=18)
    
#     # Add overarching title
#     fig.suptitle('Cosine Similarities Between Linear Probe Weights and Model Embeddings', fontsize=20)
    
#     # Save or show the combined figure
#     save_path = f"COMPLETE_FIGS/paper/cosine_sim"
#     os.makedirs(save_path, exist_ok=True)
#     fig.savefig(os.path.join(save_path, f"avg_combined_cosine_similarity_probe_embedding_heatmap.png"), bbox_inches="tight")

def avg_combined_cosine_similarity_probe_embedding_heatmap():
    cards = ['A', 'B', 'C', 'D', 'E']
    num_layers = 4
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True, constrained_layout=True)

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
            annot_kws={"fontsize": 14}
        )
        ax.set_title(f'Layer {capture_layer + 1}', fontsize=18)
        ax.set_xlabel('Probe Cards', fontsize=14)
        ax.set_ylabel('Attribute Embedding Cards', fontsize=14)
        ax.tick_params(axis='y', rotation=0, labelsize=14)  # Rotate y-tick labels to horizontal
        ax.tick_params(axis='x', labelsize=14)

    # Hide unused subplots
    for ax in axes_flat[num_layers:]:
        ax.set_visible(False)

    # Add a single colorbar for the whole figure
    cbar = fig.colorbar(axes_flat[0].collections[0], ax=axes, location='right', shrink=0.8, pad=0.02)
    cbar.set_label('Cosine Similarity', fontsize=18)
    cbar.ax.tick_params(labelsize=14)  # Set font size for colorbar tick labels
    # Remove the color bar's black outline
    cbar.outline.set_visible(False)
    
    # Add overarching title
    fig.suptitle('Cosine Similarities Between Linear Probe Weights and Model Embeddings', fontsize=22)
    
    # Save or show the combined figure
    save_path = f"COMPLETE_FIGS/paper/cosine_sim"
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"avg_combined_cosine_similarity_probe_embedding_heatmap.png"), bbox_inches="tight")


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # embedding_ablation_kl_fig()
    avg_combined_cosine_similarity_probe_embedding_heatmap()

    # config = GPTConfig44_Complete()
    # checkpoint = torch.load(config.filename, weights_only=False)

    # # Create the model architecture
    # model = GPT(config).to(device)

    # # Load the weights
    # model.load_state_dict(checkpoint['model'])
    # model.eval()  # Set to evaluation mode

    # tokenizer = load_tokenizer(config.tokenizer_path)
