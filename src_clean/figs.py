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
from data_utils import initialize_loaders, split_data
import wandb
import re

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
        similarity_matrix = np.load(
            f"{PATH_PREFIX}/complete/classify/avg_cosine_similarity_matrix_layer{capture_layer}.npy")
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
        # Rotate y-tick labels to horizontal
        ax.tick_params(axis='y', rotation=0, labelsize=label_font_size)
        ax.tick_params(axis='x', labelsize=label_font_size)

    # Hide unused subplots
    for ax in axes_flat[num_layers:]:
        ax.set_visible(False)

    # Add a single colorbar for the whole figure
    cbar = fig.colorbar(axes_flat[0].collections[0],
                        ax=axes, location='right', shrink=0.8, pad=0.02)
    cbar.set_label('Cosine Similarity', fontsize=title_font_size)
    # Set font size for colorbar tick labels
    cbar.ax.tick_params(labelsize=label_font_size)
    # Remove the color bar's black outline
    cbar.outline.set_visible(False)

    # Add overarching title
    fig.suptitle('Cosine Similarities Between Linear Probe Weights and Model Embeddings',
                 fontsize=title_font_size)

    # Save or show the combined figure
    save_path = f"COMPLETE_FIGS/paper/cosine_sim"
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(
        save_path, f"avg_combined_cosine_similarity_probe_embedding_heatmap.png"), bbox_inches="tight")


def combined_probe_weight_cosine_sim(model_config, probe_config, center=False):

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

        if center:
            centroid = np.mean(probe_weights, axis=0)
    
            # Center weights by subtracting the centroid
            probe_weights -= centroid

        # Calculate cosine similarity matrix
        cosine_sim_matrix = cosine_similarity(probe_weights)

        # Get sorting indices for alphabetical order
        sorted_indices = sorted(range(len(cards)), key=lambda k: cards[k])

        # Reorder the similarity matrix and cards
        cosine_sim_matrix = cosine_sim_matrix[sorted_indices][:,
                                                              sorted_indices]
        all_matrices.append(cosine_sim_matrix)

    sorted_cards = sorted(cards)

    # Adjust the number of layers to plot (can change based on your use case)
    num_layers = 4
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
        ax.set_title(f'Layer {capture_layer_idx + 1}',
                     fontsize=title_font_size)
        ax.set_xlabel('Probe Cards', fontsize=label_font_size)
        ax.set_ylabel('Probe Cards', fontsize=label_font_size)
        # Rotate y-tick labels to horizontal
        ax.tick_params(axis='y', rotation=0, labelsize=label_font_size)
        ax.tick_params(axis='x', labelsize=label_font_size)

    # Add a single colorbar for the whole figure
    cbar = fig.colorbar(axes_flat[0].collections[0],
                        ax=axes, location='right', shrink=0.8, pad=0.02)
    cbar.set_label('Cosine Similarity', fontsize=title_font_size)
    # Set font size for colorbar tick labels
    cbar.ax.tick_params(labelsize=label_font_size)

    # Remove the color bar's black outline
    cbar.outline.set_visible(False)

    # Add overarching title
    if center:
        fig.suptitle(f'Centered Cosine Similarities Between Linear Probe Weight Vectors',
                 fontsize=title_font_size)
    else:
        fig.suptitle(f'Cosine Similarities Between Linear Probe Weight Vectors',
                    fontsize=title_font_size)

    # Save or show the combined figure
    save_path = f"COMPLETE_FIGS/paper/cosine_sim"
    os.makedirs(save_path, exist_ok=True)

    if center:
        fig.savefig(os.path.join(
            save_path, f"centered_combined_probe_weight_cosine_sim.png"), bbox_inches="tight")
    else:
        fig.savefig(os.path.join(
            save_path, f"combined_probe_weight_cosine_sim.png"), bbox_inches="tight")

    return fig

def confirm_probe_dims(dataset_name, capture_layer):
    config = GPTConfig44_Complete()
    dataset = torch.load(config.dataset_path)
    train_loader, val_loader = initialize_loaders(config, dataset)

    input_embeddings_path = f"{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}/input_embeddings.pt"
    mapped_target_tokens_path = f"{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}/continuous_target_tokens.pt"

    X = torch.load(input_embeddings_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.load(mapped_target_tokens_path).to(device)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    breakpoint()

    print(f"Size of input_embeddings: {X.size()}")
    print(f"Size of mapped_target_tokens: {y.size()}")
    print(f"Size of X_train: {X_train.size()}")
    print(f"Size of X_val: {X_val.size()}")
    print(f"Size of X_test: {X_test.size()}")
    print(f"Size of y_train: {y_train.size()}")
    print(f"Size of y_val: {y_val.size()}")
    print(f"Size of y_test: {y_test.size()}")


def fetch_wandb_data(entity, project_name, run_names):
    """
    Fetch run data from W&B for specified runs.
    
    Args:
        entity: W&B entity name
        project_name: W&B project name
        run_names: List of run names to fetch
        
    Returns:
        Dictionary of run data with metrics
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Dictionary to store run data
    run_data = {}
    
    # Fetch data for each run
    for run_name in run_names:
        print(f"Fetching data for run: {run_name}")
        runs = api.runs(f"{entity}/{project_name}", {"display_name": run_name})
        if runs:
            run = runs[0]
            # For each run, fetch train and val loss
            train_key = "train_loss"
            val_key = "val_loss"
            history = run.history(keys=[train_key, val_key])
            # breakpoint()
            
            run_data[run_name] = {
                "steps": history["_step"].to_numpy(),
                "train_loss": history[train_key].to_numpy() if train_key in history.columns else None,
                "val_loss": history[val_key].to_numpy() if val_key in history.columns else None
            }
            print(f"  Found data with {len(history)} steps")
        else:
            print(f"  No run found with name: {run_name}")
    
    # Check if we have data to visualize
    if not run_data:
        raise ValueError("No data found for any of the specified runs")
        
    breakpoint()
    return run_data

def create_loss_figure(run_data, model_type, layers):
    """
    Create a figure with loss plots for a specific model type across multiple layers.
    
    Args:
        run_data: Dictionary containing the run data
        model_type: 'attr_from_card' or 'card_from_attr'
        layers: List of layer numbers
        layer_names: List of layer names for plot titles
        
    Returns:
        Matplotlib figure
    """
    # Define colors
    train_color = '#1f77b4'  # blue for all training curves
    val_color = '#ff7f0e'    # orange for all validation curves
    
    # Create figure
    fig, axes = plt.subplots(1, len(layers), figsize=(5*len(layers), 5))
    if model_type == "attr_from_card":
        model_title = "Attribute From Card"
    else:
        model_title = "Card From Attribute"

    fig.suptitle(f'{model_title} Loss', fontsize=title_font_size)
    
    # First pass: determine global min and max values across all layers
    global_min = float('inf')
    global_max = float('-inf')
    
    for layer in layers:
        run_name = f"{model_type}_linear_layer{layer}"
        if run_name in run_data and run_data[run_name]["train_loss"] is not None:
            train_loss = run_data[run_name]["train_loss"]
            val_loss = run_data[run_name]["val_loss"]
            
            global_min = min(global_min, np.min(train_loss), np.min(val_loss))
            global_max = max(global_max, np.max(train_loss), np.max(val_loss))
    
    # Add padding to the global min/max
    padding = (global_max - global_min) * 0.1
    y_min = global_min - padding
    y_max = global_max + padding
    
    # Second pass: create the plots with standardized y-axis
    for i, layer in enumerate(layers):
        ax = axes[i]
        
        # Get run for this layer
        run_name = f"{model_type}_linear_layer{layer}"
        
        if run_name in run_data and run_data[run_name]["train_loss"] is not None:
            # Plot losses
            steps = run_data[run_name]["steps"]
            train_loss = run_data[run_name]["train_loss"]
            val_loss = run_data[run_name]["val_loss"]
            
            ax.plot(steps, train_loss, color=train_color, label='Training Loss')
            ax.plot(steps, val_loss, color=val_color, label='Validation Loss')
            
            # Set standardized y-axis limits
            ax.set_ylim(y_min, y_max)
        else:
            ax.text(0.5, 0.5, f"No data for {run_name}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
        
        ax.set_title(f'Layer {i+1}')
        ax.set_xlabel('Epochs', fontsize=label_font_size)
        # ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only add y-label to the first subplot
        if i == 0:
            ax.set_ylabel('Loss', rotation=0, fontsize = label_font_size)
            
        # Add legend to the last subplot
        if i == len(layers) - 1:
            ax.legend(loc='upper right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
    return fig

def run_probe_weight_loss_fig():
    entity = "hazhou-harvard"
    project_name = "full-complete-classify-card"
    
    # List of run names to include in the visualization
    run_names = [
        "attr_from_card_linear_layer3",
        "attr_from_card_linear_layer2",
        "attr_from_card_linear_layer1",
        "attr_from_card_linear_layer0",
        "card_from_attr_linear_layer3",
        "card_from_attr_linear_layer1",
        "card_from_attr_linear_layer0",
        "card_from_attr_linear_layer2"
    ]
    
    # Layer numbers and names
    layers = [0, 1, 2, 3]
    
    # Fetch data from W&B
    run_data = fetch_wandb_data(entity, project_name, run_names)
    
    fig_save_dir = "COMPLETE_FIGS/paper/probe_weight_loss"
    os.makedirs(fig_save_dir, exist_ok=True)

    # Create figure for attr_from_card
    model_type = "attr_from_card"
    attr_fig = create_loss_figure(run_data, model_type, layers)
    attr_fig.savefig(os.path.join(fig_save_dir, f'{model_type}_losses.png'), dpi=300, bbox_inches='tight')
    
    model_type = "card_from_attr"
    attr_fig = create_loss_figure(run_data, model_type, layers)
    attr_fig.savefig(os.path.join(fig_save_dir, f'{model_type}_losses.png'), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    run_probe_weight_loss_fig()

    # embedding_ablation_kl_fig()
    # avg_combined_cosine_similarity_probe_embedding_heatmap()
    # combined_probe_weight_cosine_sim(
    #     GPTConfig44_Complete(), 
    #     LinearProbeBindingCardAttrConfig())
    # combined_probe_weight_cosine_sim(
    #     GPTConfig44_Complete(),
    #     LinearProbeBindingCardAttrConfig(),
    #     center=True)

    # confirm_probe_dims("card_from_attr", 0)
    # confirm_probe_dims("attr_from_card", 0)

    # config = GPTConfig44_Complete()
    # checkpoint = torch.load(config.filename, weights_only=False)

    # # Create the model architecture
    # model = GPT(config).to(device)

    # # Load the weights
    # model.load_state_dict(checkpoint['model'])
    # model.eval()  # Set to evaluation mode

    # tokenizer = load_tokenizer(config.tokenizer_path)
