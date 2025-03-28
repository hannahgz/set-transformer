import pickle
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
import pandas as pd

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
        model_title = "Card-Attribute Binding Linear Probe: Attribute From Card Loss Curves"
    else:
        model_title = "Card-Attribute Binding Linear Probe: Card From Attribute Loss Curves"

    fig.suptitle(f'{model_title}', fontsize=title_font_size)

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

            ax.plot(steps, train_loss, color=train_color,
                    label='Training Loss')
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
            ax.set_ylabel('Loss', rotation=0,
                          fontsize=label_font_size, labelpad=20)

        # Add legend to the last subplot
        if i == len(layers) - 1:
            ax.legend(loc='upper right')

    plt.tight_layout(rect=[0.05, 0, 1, 0.99])  # Make room for the title
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
    attr_fig.savefig(os.path.join(
        fig_save_dir, f'{model_type}_losses.png'), dpi=300, bbox_inches='tight')

    model_type = "card_from_attr"
    attr_fig = create_loss_figure(run_data, model_type, layers)
    attr_fig.savefig(os.path.join(
        fig_save_dir, f'{model_type}_losses.png'), dpi=300, bbox_inches='tight')


def fetch_wandb_test_accuracy_data(entity, project_name, run_names):
    """
    Fetch final test accuracy data from W&B for specified runs.

    Args:
        entity: W&B entity name
        project_name: W&B project name
        run_names: List of run names to fetch

    Returns:
        Dictionary of model names mapped to their final test accuracy
    """
    # Initialize wandb API
    api = wandb.Api()

    # Dictionary to store accuracy data
    accuracy_data = {}

    # Fetch data for each run
    for run_name in run_names:
        print(f"Fetching data for run: {run_name}")
        runs = api.runs(f"{entity}/{project_name}", {"display_name": run_name})

        if runs:
            run = runs[0]
            # For each run, fetch final test accuracy
            summary = run.summary

            # The field name for test accuracy may vary, try different possibilities
            test_accuracy = None
            for field in ["final_test_accuracy", "test_accuracy", "accuracy/test"]:
                if field in summary._json_dict:
                    test_accuracy = summary._json_dict[field]
                    break

            if test_accuracy is not None:
                accuracy_data[run_name] = test_accuracy
                print(f"  Found test accuracy: {test_accuracy}")
            else:
                print(f"  No test accuracy found for run: {run_name}")
        else:
            print(f"  No run found with name: {run_name}")

    return accuracy_data


def create_final_test_accuracy_chart(data, output_path="final_test_accuracy_chart.png"):
    """
    Creates a grouped bar chart showing final test accuracy for models,
    split into two groups (attr_from_card and card_from_attr).

    Args:
        data: Dictionary with keys as model names and values as their test accuracies
             Example format: {
                'attr_from_card_linear_layer0': 0.85,
                'attr_from_card_linear_layer1': 0.87,
                ...
                'card_from_attr_linear_layer0': 0.92,
                ...
             }
        output_path: Path to save the figure

    Returns:
        The matplotlib figure object
    """
    # Extract model types and layers from the data keys
    models = []
    accuracies = []
    types = []
    layers = []

    for model_name, accuracy in data.items():
        if 'attr_from_card' in model_name:
            model_type = 'Attribute from Card'
            # Extract layer number from the model name
            layer = int(model_name.split('layer')[-1])
        elif 'card_from_attr' in model_name:
            model_type = 'Card from Attribute'
            layer = int(model_name.split('layer')[-1])
        else:
            continue  # Skip if not matching our expected pattern

        models.append(model_name)
        accuracies.append(accuracy)
        types.append(model_type)
        layers.append(layer)

    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracies,
        'Type': types,
        'Layer': layers
    })

    # Sort by layer number to ensure sequential order
    df = df.sort_values(by=['Type', 'Layer'])

    # Set up the figure
    plt.figure(figsize=(12, 4))

    # Set seaborn style
    sns.set_style("white")

    # Create grouped bar chart
    ax = sns.barplot(
        x='Layer',
        y='Accuracy',
        hue='Type',
        data=df,
        palette=['#6A0572', '#36D1DC']
        # palette=['#1f77b4', '#ff7f0e']  # Blue for attr_from_card, Orange for card_from_attr
    )

    # Customize the plot
    plt.title('Card-Attribute Binding Linear Probe: Test Accuracy',
              fontsize=title_font_size)
    plt.xlabel('Layer', fontsize=label_font_size)
    plt.ylabel('Accuracy', fontsize=label_font_size)
    plt.xticks(fontsize=label_font_size)
    plt.yticks(fontsize=label_font_size)

    # Set the y-axis to start at 0
    plt.ylim(0, max(df['Accuracy']) * 1.1)  # Add 10% padding at the top

    # Adjust x-axis labels to show actual layer numbers (0, 1, 2, 3)
    num_layers = len(df['Layer'].unique())
    # plt.xticks(range(num_layers), sorted(df['Layer'].unique()))
    plt.xticks(range(num_layers), [
               layer + 1 for layer in sorted(df['Layer'].unique())])
    ax.tick_params(axis='y', which='both', length=5, width=1)
    plt.tick_params(axis='y', which='both', left=True, right=False)

    # Add value annotations on top of bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        # Only add text if the height is significant
        if height > 0.001:  # Skip annotations for very small or zero values
            ax.text(
                p.get_x() + p.get_width() / 2.,
                height + 0.01,
                f'{height:.3f}',
                ha='center',
                fontsize=annot_font_size
            )

    # Adjust legend
    plt.legend(title='Model Type', fontsize=label_font_size,
               title_fontsize=label_font_size, loc='upper left')

    # Tight layout to ensure everything fits
    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return plt.gcf()

# Main function to run the full pipeline


def create_test_accuracy_visualization():
    """
    Main function to fetch data from wandb and create the test accuracy visualization.
    Uses the same wandb project and entity as in run_probe_weight_loss_fig().
    """
    # Define wandb parameters (same as in run_probe_weight_loss_fig)
    entity = "hazhou-harvard"
    project_name = "full-complete-classify-card"

    # List of run names to include in the visualization
    run_names = [
        "attr_from_card_linear_layer0",
        "attr_from_card_linear_layer1",
        "attr_from_card_linear_layer2",
        "attr_from_card_linear_layer3",
        "card_from_attr_linear_layer0",
        "card_from_attr_linear_layer1",
        "card_from_attr_linear_layer2",
        "card_from_attr_linear_layer3"
    ]

    # Create output directory
    fig_save_dir = "COMPLETE_FIGS/paper/probe_weight_loss"
    os.makedirs(fig_save_dir, exist_ok=True)
    output_path = os.path.join(fig_save_dir, "final_test_accuracy_chart.png")

    # Fetch data from wandb
    print("Fetching test accuracy data from wandb...")
    accuracy_data = fetch_wandb_test_accuracy_data(
        entity, project_name, run_names)

    # Check if we have data to visualize
    if not accuracy_data:
        print("No accuracy data found")

    # Create and save the chart
    print(f"Creating test accuracy chart and saving to {output_path}")
    fig = create_final_test_accuracy_chart(accuracy_data, output_path)
    plt.show()
    print("Done!")
    return fig


def plot_consolidated_attribute_metrics(layers, tokenizer_path, loss_range=[0, 0.5], acc_range=[0.7, 1], project_name="binary-probe-training-all-attr", entity="hazhou-harvard"):
    """
    Create four figures:
    1. Training losses across all layers
    2. Validation losses across all layers
    3. Training accuracies across all layers
    4. Validation accuracies across all layers

    Args:
        layers (list): List of layer numbers to plot
        tokenizer_path (str): Path to the tokenizer
        project_name (str): Name of the W&B project
        entity (str): Your W&B username
    """
    # Initialize wandb
    api = wandb.Api()

    # Get all runs from your project
    runs = api.runs(f"{entity}/{project_name}")

    # Define the desired attribute order
    shapes = ["oval", "squiggle", "diamond"]
    colors = ["green", "blue", "pink"]
    numbers = ["one", "two", "three"]
    shadings = ["solid", "striped", "open"]
    desired_order = shapes + colors + numbers + shadings

    # Create four figures with (1, 4) subplots
    fig_train_loss, axes_train_loss = plt.subplots(1, 4, figsize=(24, 5))
    fig_val_loss, axes_val_loss = plt.subplots(1, 4, figsize=(24, 5))
    fig_train_acc, axes_train_acc = plt.subplots(1, 4, figsize=(24, 5))
    fig_val_acc, axes_val_acc = plt.subplots(1, 4, figsize=(24, 5))

    # Set style
    sns.set_style("white")

    tokenizer = load_tokenizer(tokenizer_path)

    # Process each layer
    for layer_idx, target_layer in enumerate(layers):
        # Filter and sort runs for current layer
        layer_runs = []
        run_order_mapping = {}

        for run in runs:
            match = re.search(r'layer(\d+)$', run.name)
            if match and int(match.group(1)) == target_layer:
                attr_match = re.search(r'attr_(\d+)_', run.name)
                if attr_match:
                    attr_id = int(attr_match.group(1))
                    attr_name = tokenizer.id_to_token[attr_id]
                    try:
                        order_index = desired_order.index(attr_name)
                        run_order_mapping[run] = order_index
                        layer_runs.append(run)
                    except ValueError:
                        continue

        # Sort runs based on desired order
        layer_runs.sort(key=lambda x: run_order_mapping[x])

        # Color map for different attributes
        num_runs = len(layer_runs)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_runs))

        # Plot for each run
        for run, color in zip(layer_runs, colors):
            # Extract attribute ID and get label
            attr_match = re.search(r'attr_(\d+)_', run.name)
            attr_id = attr_match.group(1) if attr_match else 'unknown'
            label = tokenizer.id_to_token[int(attr_id)]

            # Convert run history to pandas DataFrame
            history = pd.DataFrame(run.history())

            # Plot training loss
            axes_train_loss[layer_idx].plot(history['epoch'], history['train_loss'],
                                            label=label, color=color)
            axes_train_loss[layer_idx].set_title(
                f'Layer {target_layer + 1}', fontsize=title_font_size)
            axes_train_loss[layer_idx].set_xlabel(
                'Epoch', fontsize=label_font_size)
            if layer_idx == 0:
                axes_train_loss[layer_idx].set_ylabel(
                    'Loss', fontsize=label_font_size)
            axes_train_loss[layer_idx].set_ylim(loss_range[0], loss_range[1])
            axes_train_loss[layer_idx].tick_params(labelsize=annot_font_size)

            # Plot validation loss
            axes_val_loss[layer_idx].plot(history['epoch'], history['val_loss'],
                                          label=label, color=color)
            axes_val_loss[layer_idx].set_title(
                f'Layer {target_layer + 1}', fontsize=title_font_size)
            axes_val_loss[layer_idx].set_xlabel(
                'Epoch', fontsize=label_font_size)
            if layer_idx == 0:
                axes_val_loss[layer_idx].set_ylabel(
                    'Loss', fontsize=label_font_size)
            axes_val_loss[layer_idx].set_ylim(loss_range[0], loss_range[1])
            axes_val_loss[layer_idx].tick_params(labelsize=annot_font_size)

            # Plot training accuracy
            axes_train_acc[layer_idx].plot(history['epoch'], history['train_accuracy'],
                                           label=label, color=color)
            axes_train_acc[layer_idx].set_title(
                f'Layer {target_layer + 1}', fontsize=title_font_size)
            axes_train_acc[layer_idx].set_xlabel(
                'Epoch', fontsize=label_font_size)
            if layer_idx == 0:
                axes_train_acc[layer_idx].set_ylabel(
                    'Accuracy', fontsize=label_font_size)
            axes_train_acc[layer_idx].set_ylim(acc_range[0], acc_range[1])
            axes_train_acc[layer_idx].tick_params(labelsize=annot_font_size)

            # Plot validation accuracy
            axes_val_acc[layer_idx].plot(history['epoch'], history['val_accuracy'],
                                         label=label, color=color)
            axes_val_acc[layer_idx].set_title(
                f'Layer {target_layer + 1}', fontsize=title_font_size)
            axes_val_acc[layer_idx].set_xlabel(
                'Epoch', fontsize=label_font_size)
            if layer_idx == 0:
                axes_val_acc[layer_idx].set_ylabel(
                    'Accuracy', fontsize=label_font_size)
            axes_val_acc[layer_idx].set_ylim(acc_range[0], acc_range[1])
            axes_val_acc[layer_idx].tick_params(labelsize=annot_font_size)

        # Only add legend to the last subplot
        if layer_idx == len(layers) - 1:
            axes_train_loss[layer_idx].legend(bbox_to_anchor=(
                1.05, 1), loc='upper left', fontsize=annot_font_size)
            axes_val_loss[layer_idx].legend(bbox_to_anchor=(
                1.05, 1), loc='upper left', fontsize=annot_font_size)
            axes_train_acc[layer_idx].legend(bbox_to_anchor=(
                1.05, 1), loc='upper left', fontsize=annot_font_size)
            axes_val_acc[layer_idx].legend(bbox_to_anchor=(
                1.05, 1), loc='upper left', fontsize=annot_font_size)

    # Set main titles
    fig_train_loss.suptitle('Training Losses', fontsize=title_font_size)
    fig_val_loss.suptitle('Validation Losses', fontsize=title_font_size)
    fig_train_acc.suptitle('Training Accuracies', fontsize=title_font_size)
    fig_val_acc.suptitle('Validation Accuracies', fontsize=title_font_size)

    # Adjust layouts
    fig_train_loss.tight_layout()
    fig_val_loss.tight_layout()
    fig_train_acc.tight_layout()
    fig_val_acc.tight_layout()

    # Create save directory
    save_path = f"COMPLETE_FIGS/paper/{project_name}"
    os.makedirs(save_path, exist_ok=True)

    # Save figures
    fig_train_loss.savefig(
        f'{save_path}/{project_name}_all_layers_train_loss.png', dpi=300, bbox_inches='tight')
    fig_val_loss.savefig(
        f'{save_path}/{project_name}_all_layers_val_loss.png', dpi=300, bbox_inches='tight')
    fig_train_acc.savefig(
        f'{save_path}/{project_name}_all_layers_train_acc.png', dpi=300, bbox_inches='tight')
    fig_val_acc.savefig(
        f'{save_path}/{project_name}_all_layers_val_acc.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_consolidated_same_card(layers, tokenizer_path, loss_range=[0, 0.8], acc_range=[0.6, 1.05], project_name="acc-binding-id-specific-card", entity="hazhou-harvard"):
    """
    Create four figures:
    1. Training losses across all layers
    2. Validation losses across all layers
    3. Training accuracies across all layers
    4. Validation accuracies across all layers

    Args:
        layers (list): List of layer numbers to plot
        tokenizer_path (str): Path to the tokenizer
        project_name (str): Name of the W&B project
        entity (str): Your W&B username
    """
    # Initialize wandb
    api = wandb.Api()

    # Get all runs from your project
    runs = api.runs(f"{entity}/{project_name}")

    # Create four figures with (1, 4) subplots
    fig_train_loss, axes_train_loss = plt.subplots(1, 4, figsize=(24, 5))
    fig_val_loss, axes_val_loss = plt.subplots(1, 4, figsize=(24, 5))
    fig_train_acc, axes_train_acc = plt.subplots(1, 4, figsize=(24, 5))
    fig_val_acc, axes_val_acc = plt.subplots(1, 4, figsize=(24, 5))

    # Set style
    sns.set_style("white")

    tokenizer = load_tokenizer(tokenizer_path)

    desired_order = ["A", "B", "C", "D", "E"]
    # Process each layer
    for layer_idx, target_layer in enumerate(layers):
        # Filter and sort runs for current layer
        layer_runs = []
        run_order_mapping = {}

        for run in runs:
            match = re.search(r'layer(\d+)', run.name)
            if match and int(match.group(1)) == target_layer:
                card_match = re.search(r'card(\d+)', run.name)
                if card_match:
                    card_id = int(card_match.group(1))
                    card_name = tokenizer.id_to_token[card_id]
                    try:
                        order_index = desired_order.index(card_name)
                        run_order_mapping[run] = order_index
                        layer_runs.append(run)
                    except ValueError:
                        continue

        # Sort runs based on desired order
        layer_runs.sort(key=lambda x: run_order_mapping[x])

        # Color map for different attributes
        num_runs = len(layer_runs)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_runs))

        # Plot for each run
        for run, color in zip(layer_runs, colors):
            # Extract attribute ID and get label
            card_match = re.search(r'card(\d+)', run.name)
            card_id = int(card_match.group(1))
            label = f"Card {tokenizer.id_to_token[card_id]}"

            # Convert run history to pandas DataFrame
            history = pd.DataFrame(run.history())

            # Plot training loss
            axes_train_loss[layer_idx].plot(history['epoch'], history['train_loss'],
                                            label=label, color=color)
            axes_train_loss[layer_idx].set_title(
                f'Layer {target_layer + 1}', fontsize=title_font_size)
            axes_train_loss[layer_idx].set_xlabel(
                'Epoch', fontsize=label_font_size)
            if layer_idx == 0:
                axes_train_loss[layer_idx].set_ylabel(
                    'Loss', fontsize=label_font_size)
            # axes_train_loss[layer_idx].set_ylim(loss_range[0], loss_range[1])
            # axes_train_loss[layer_idx].set_yscale('log')
            axes_train_loss[layer_idx].ticklabel_format(
                useOffset=False, style='plain')
            if layer_idx != 0:
                axes_train_loss[layer_idx].set_ylim(-0.001, 0.03)
            axes_train_loss[layer_idx].tick_params(labelsize=annot_font_size)

            # Plot validation loss
            axes_val_loss[layer_idx].plot(history['epoch'], history['val_loss'],
                                          label=label, color=color)
            axes_val_loss[layer_idx].set_title(
                f'Layer {target_layer + 1}', fontsize=title_font_size)
            axes_val_loss[layer_idx].set_xlabel(
                'Epoch', fontsize=label_font_size)
            if layer_idx == 0:
                axes_val_loss[layer_idx].set_ylabel(
                    'Loss', fontsize=label_font_size)
            # axes_val_loss[layer_idx].set_ylim(loss_range[0], loss_range[1])
            # axes_val_loss[layer_idx].set_yscale('log')
            axes_val_loss[layer_idx].ticklabel_format(
                useOffset=False, style='plain')
            if layer_idx != 0:
                axes_val_loss[layer_idx].set_ylim(-0.001, 0.013)
            axes_val_loss[layer_idx].tick_params(labelsize=annot_font_size)

            # Plot training accuracy
            axes_train_acc[layer_idx].plot(history['epoch'], history['train_accuracy'],
                                           label=label, color=color)
            axes_train_acc[layer_idx].set_title(
                f'Layer {target_layer + 1}', fontsize=title_font_size)
            axes_train_acc[layer_idx].set_xlabel(
                'Epoch', fontsize=label_font_size)
            if layer_idx == 0:
                axes_train_acc[layer_idx].set_ylabel(
                    'Accuracy', fontsize=label_font_size)
            # axes_train_acc[layer_idx].set_ylim(acc_range[0], acc_range[1])
            # axes_train_acc[layer_idx].set_yscale('log')
            axes_train_acc[layer_idx].ticklabel_format(
                useOffset=False, style='plain')
            if layer_idx != 0:
                axes_train_acc[layer_idx].set_ylim(0.992, 1.001)
            axes_train_acc[layer_idx].tick_params(labelsize=annot_font_size)

            # Plot validation accuracy
            axes_val_acc[layer_idx].plot(history['epoch'], history['val_accuracy'],
                                         label=label, color=color)
            axes_val_acc[layer_idx].set_title(
                f'Layer {target_layer + 1}', fontsize=title_font_size)
            axes_val_acc[layer_idx].set_xlabel(
                'Epoch', fontsize=label_font_size)
            if layer_idx == 0:
                axes_val_acc[layer_idx].set_ylabel(
                    'Accuracy', fontsize=label_font_size)
            # axes_val_acc[layer_idx].set_ylim(acc_range[0], acc_range[1])
            # axes_val_acc[layer_idx].set_yscale('log')
            axes_val_acc[layer_idx].ticklabel_format(
                useOffset=False, style='plain')
            if layer_idx != 0:
                axes_val_acc[layer_idx].set_ylim(0.996, 1.001)
            axes_val_acc[layer_idx].tick_params(labelsize=annot_font_size)

        # Only add legend to the last subplot
        if layer_idx == len(layers) - 1:
            axes_train_loss[layer_idx].legend(bbox_to_anchor=(
                1.05, 1), loc='upper left', fontsize=annot_font_size)
            axes_val_loss[layer_idx].legend(bbox_to_anchor=(
                1.05, 1), loc='upper left', fontsize=annot_font_size)
            axes_train_acc[layer_idx].legend(bbox_to_anchor=(
                1.05, 1), loc='upper left', fontsize=annot_font_size)
            axes_val_acc[layer_idx].legend(bbox_to_anchor=(
                1.05, 1), loc='upper left', fontsize=annot_font_size)

    # Set main titles
    fig_train_loss.suptitle('Training Losses', fontsize=title_font_size)
    fig_val_loss.suptitle('Validation Losses', fontsize=title_font_size)
    fig_train_acc.suptitle('Training Accuracies', fontsize=title_font_size)
    fig_val_acc.suptitle('Validation Accuracies', fontsize=title_font_size)

    # Adjust layouts
    fig_train_loss.tight_layout()
    fig_val_loss.tight_layout()
    fig_train_acc.tight_layout()
    fig_val_acc.tight_layout()

    # Create save directory
    save_path = f"COMPLETE_FIGS/paper/{project_name}"
    os.makedirs(save_path, exist_ok=True)

    # Save figures
    fig_train_loss.savefig(
        f'{save_path}/{project_name}_all_layers_train_loss.png', dpi=300, bbox_inches='tight')
    fig_val_loss.savefig(
        f'{save_path}/{project_name}_all_layers_val_loss.png', dpi=300, bbox_inches='tight')
    fig_train_acc.savefig(
        f'{save_path}/{project_name}_all_layers_train_acc.png', dpi=300, bbox_inches='tight')
    fig_val_acc.savefig(
        f'{save_path}/{project_name}_all_layers_val_acc.png', dpi=300, bbox_inches='tight')

    plt.show()


def attr_from_last_attr_dataset_size(parent_folder="all_attr_from_last_attr_binding", capture_layer=0, attribute_id=1):
    dataset_path = f"{PATH_PREFIX}/{parent_folder}/layer{capture_layer}/binary_dataset_{attribute_id}.pt"
    data = torch.load(dataset_path)

    input_embeddings = data['input_embeddings']
    breakpoint()
    print(
        f"Dataset size for attribute {attribute_id}: {len(input_embeddings)}")

    val_split = 0.2
    # Calculate split sizes
    val_size = int(len(input_embeddings) * val_split)
    train_size = len(input_embeddings) - val_size

    print(f"Train size: {train_size}, Val size: {val_size}")


def create_loss_figure_orig_set_model(run_data, layers):
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
    fig, axes = plt.subplots(1, len(layers), figsize=(5*len(layers), 4))
    model_title = "Set Prediction Model Loss Curves"

    fig.suptitle(f'{model_title}', fontsize=title_font_size)

    # First pass: determine global min and max values across all layers
    global_min = float('inf')
    global_max = float('-inf')

    for run_name in run_data:
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
    for i, run_name in enumerate(run_data):
        ax = axes[i]

        if run_data[run_name]["train_loss"] is not None:
            # Plot losses
            steps = run_data[run_name]["steps"]
            train_loss = run_data[run_name]["train_loss"]
            val_loss = run_data[run_name]["val_loss"]

            ax.plot(steps, train_loss, color=train_color,
                    label='Training Loss')
            ax.plot(steps, val_loss, color=val_color, label='Validation Loss')

            # Set standardized y-axis limits
            ax.set_ylim(y_min, y_max)
        else:
            ax.text(0.5, 0.5, f"No data for {run_name}",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)

        ax.set_title(f'{layers[i]} Layers')
        ax.set_xlabel('Steps', fontsize=label_font_size)
        # ax.grid(True, linestyle='--', alpha=0.7)

        # Only add y-label to the first subplot
        if i == 0:
            ax.set_ylabel('Loss', rotation=0,
                          fontsize=label_font_size, labelpad=20)

        # Add legend to the last subplot
        if i == len(layers) - 1:
            ax.legend(loc='upper right')

    plt.tight_layout(rect=[0.05, 0, 1, 0.99])  # Make room for the title
    return fig


def create_loss_figure_orig_set_model_seeded(run_data, seeds=[1, 100, 200, 300, 400]):
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

    # annot_font_size += 2
    # title_font_size += 2
    # label_font_size += 2

    # Define colors
    train_color = '#1f77b4'  # blue for all training curves
    val_color = '#ff7f0e'    # orange for all validation curves

    # Create figure
    fig, axes = plt.subplots(1, len(seeds), figsize=(4*len(seeds), 4))
    model_title = "Set Prediction Model Loss Curves"

    fig.suptitle(f'{model_title}', fontsize=title_font_size + 2)

    # First pass: determine global min and max values across all layers
    global_min = float('inf')
    global_max = float('-inf')

    for run_name in run_data:
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
    for i, run_name in enumerate(run_data):
        ax = axes[i]

        if run_data[run_name]["train_loss"] is not None:
            # Plot losses
            steps = run_data[run_name]["steps"]
            train_loss = run_data[run_name]["train_loss"]
            val_loss = run_data[run_name]["val_loss"]

            ax.plot(steps, train_loss, color=train_color,
                    label='Training Loss')
            ax.plot(steps, val_loss, color=val_color, label='Validation Loss')

            # Set standardized y-axis limits
            ax.set_ylim(y_min, y_max)
        else:
            ax.text(0.5, 0.5, f"No data for {run_name}",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)

        if seeds[i] != 1:
            ax.set_title(f'Seed {seeds[i]}', fontsize=label_font_size + 2)
        else:
            ax.set_title(f'Seed 42 (Original)', fontsize=label_font_size + 2)
        ax.set_xlabel('Steps', fontsize=label_font_size + 2)
        # ax.grid(True, linestyle='--', alpha=0.7)

        ax.tick_params(axis='x', labelsize=label_font_size)
        ax.tick_params(axis='y', labelsize=label_font_size)
        # Only add y-label to the first subplot
        if i == 0:
            ax.set_ylabel('Loss', rotation=0,
                          fontsize=label_font_size + 2, labelpad=20)

        # Add legend to the last subplot
        if i == len(seeds) - 1:
            ax.legend(loc='upper right')

    plt.tight_layout(rect=[0.05, 0, 1, 0.99])  # Make room for the title
    return fig


def fetch_wandb_data_set_model(entity, project_name, run_names):
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

    return run_data


def plot_orig_set_model_loss():
    run_data = fetch_wandb_data_set_model(
        entity="hazhou-harvard",
        project_name="equal-set-prediction-balanced",
        run_names=[
            "triples_card_randomization_tuple_randomization_layers_2_heads_4.pt",
            "triples_card_randomization_tuple_randomization_layers_3_heads_4.pt",
            "triples_card_randomization_tuple_randomization_layers_4_heads_4.pt"]
    )

    fig_save_dir = "COMPLETE_FIGS/paper/set_model"
    os.makedirs(fig_save_dir, exist_ok=True)

    fig = create_loss_figure_orig_set_model(
        run_data,
        layers=[2, 3, 4])

    fig.savefig(os.path.join(
        fig_save_dir, f'set_model_losses.png'), dpi=300, bbox_inches='tight')


def plot_orig_set_model_loss_seeded():
    run_data = fetch_wandb_data_set_model(
        entity="hazhou-harvard",
        project_name="equal-set-prediction-balanced",
        run_names=[
            "triples_card_randomization_tuple_randomization_layers_4_heads_4.pt",
            "seed100/triples_card_randomization_tuple_randomization_layers_4_heads_4.pt",
            "seed200/triples_card_randomization_tuple_randomization_layers_4_heads_4.pt",
            "seed300/triples_card_randomization_tuple_randomization_layers_4_heads_4.pt",
            "seed400/triples_card_randomization_tuple_randomization_layers_4_heads_4.pt",
        ]
    )

    fig_save_dir = "COMPLETE_FIGS/paper/set_model_seeded"
    os.makedirs(fig_save_dir, exist_ok=True)

    fig = create_loss_figure_orig_set_model_seeded(
        run_data)

    fig.savefig(os.path.join(
        fig_save_dir, f'set_model_losses.png'), dpi=300, bbox_inches='tight')


def plot_mlp_layer3_weights():
    mlp_weights = torch.load(
        f"{PATH_PREFIX}/mlp_triples_card_randomization_tuple_randomization_layers_4_heads_4.pt")

    layer_name = "layer_3"
    weight_name = "c_proj"
    weights = mlp_weights[layer_name][weight_name]

    # Use a wider figure with appropriate aspect ratio
    plt.figure(figsize=(15, 6))  # Wider figure

    # Create the heatmap
    ax = sns.heatmap(weights.T, cmap='coolwarm', center=0,
                     cbar_kws={'pad': 0.01})

    plt.title(f"Layer 4: MLP Projection Weight Matrix", fontsize=18)

    # Create tick positions with wider intervals
    x_ticks = np.arange(0, weights.shape[0], 25)
    y_ticks = np.arange(0, weights.shape[1], 10)

    # Set the tick positions
    plt.xticks(x_ticks + 0.5, x_ticks, fontsize=12, rotation=0)
    plt.yticks(y_ticks + 0.5, y_ticks, fontsize=12, rotation=0)

    # Add proper axis labels based on MLP architecture
    plt.xlabel("Hidden Layer Neurons (256)", fontsize=16)
    plt.ylabel("Output Embedding Dimensions (64)", fontsize=16)

    # Adjust layout to minimize whitespace
    plt.tight_layout()

    # Optional: adjust the colorbar position if it's creating whitespace
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    fig_save_dir = "COMPLETE_FIGS/paper/mlp"
    os.makedirs(fig_save_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_save_dir, f"{weight_name}_{layer_name}_heatmap.png"),
                dpi=300,
                bbox_inches='tight')  # 'tight' parameter reduces extra whitespace
    plt.close()

    # for layer_name, layer_weights in mlp_weights.items():
    # for weight_name, weights in layer_weights.items():

    # print(f"Layer: {layer_name}, Weight: {weight_name}")
    # 1. Direct Heatmap Visualization
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(weights, cmap='coolwarm', center=0)
    # plt.title(f"{layer_name} - {weight_name} Weight Matrix")
    # plt.tight_layout()
    # plt.savefig(os.path.join(layer_dir, f"{weight_name}_heatmap.png"))
    # plt.close()


def get_neuron_activation(neuron_type="hidden", curr_layer=3):
    output_dir = f"{PATH_PREFIX}/data/mlp_fixed/{neuron_type}/layer{curr_layer}"

    pkl_filename = os.path.join(
        output_dir, f"neuron_activations_layer{curr_layer}_all_positions.pkl")

    # Load neuron activations
    with open(pkl_filename, "rb") as f:
        neuron_activations = pickle.load(f)

    return neuron_activations


from matplotlib.ticker import FormatStrFormatter

def plot_overlap_histograms(neuron_activations, target_neurons, pos_range=range(36, 40), num_bins=50, figsize=None):
    # Define set filtering types and their colors
    set_types = {
        0: {"name": "No Set", "color": "blue", "alpha": 0.5},
        1: {"name": "One Set", "color": "green", "alpha": 0.5},
        2: {"name": "Two Set", "color": "red", "alpha": 0.5}
    }

    # Calculate rows and columns for subplot grid
    n_neurons = len(target_neurons)
    n_cols = len(pos_range)  # Each column represents a position
    n_rows = n_neurons  # Each row represents a neuron

    # Set default figsize based on number of neurons if not provided
    if figsize is None:
        figsize = (20, 4 * n_neurons)

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle the case of a single neuron (1D array)
    if n_neurons == 1:
        axes = axes.reshape(1, -1)

    # Plot histogram for each neuron and position
    for i, neuron in enumerate(target_neurons):
        for j, pos_idx in enumerate(pos_range):
            print(f"Plotting neuron {neuron} at position {pos_idx}...")
            ax = axes[i, j]
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

            # Plot histogram for each set type at this position
            for set_type in set_types:
                activations = neuron_activations[neuron][pos_idx][set_type]
                if not activations:  # Skip if no data for this set type
                    continue

                # Calculate statistics for this set type
                mean_act = np.mean(activations)
                median_act = np.median(activations)

                set_info = set_types[set_type]

                # Plot histogram with alpha transparency to show overlap
                ax.hist(activations, bins=num_bins, alpha=set_info["alpha"],
                        color=set_info["color"], edgecolor='black',
                        label=f'{set_info["name"]} (n={len(activations)})')

                # Add vertical lines for mean values
                ax.axvline(mean_act, color=set_info["color"], linestyle='dashed', linewidth=2,
                           label=f'{set_info["name"]} Mean: {mean_act:.3f}')

            # Set titles and labels
            # if i == 0:  # Add position labels only on top row
            ax.set_title(f'Position {pos_idx}')
            if j == 0:  # Add neuron labels only on leftmost column
                ax.set_ylabel(f'Neuron {neuron}')

            # Only add legend to the rightmost column
            # if pos_idx == n_cols - 1:
            ax.legend(loc='best', fontsize=8)

            # Remove x-axis labels except for bottom row
            # if i != n_rows - 1:
            #     ax.set_xticklabels([])
            if i == n_rows - 1:
                ax.set_xlabel('Activation Value')

    plt.tight_layout()
    print(
        f"Done! Plotted histograms for {len(target_neurons)} neurons with overlapping set types.")

    return fig


def all_neuron_histogram_activation_breakdown():
    neuron_activations = get_neuron_activation()

    # indices = sorted([185, 25, 93, 36, 166, 89])
    indices = sorted([25, 36, 89, 185])

    fig = plot_overlap_histograms(
        neuron_activations=neuron_activations,
        target_neurons=indices)
    
    fig_save_dir = "COMPLETE_FIGS/paper/mlp"
    os.makedirs(fig_save_dir, exist_ok=True)

    plt.savefig(os.path.join(fig_save_dir, f"hidden_consolidated_selected_neurons.png"),
                dpi=300,
                bbox_inches='tight')

def create_accuracy_comparison_chart():
    """
    Creates a grouped bar chart showing training and validation accuracies grouped by number of layers.
    Loads accuracy values directly from text files.
    
    Args:
        path_prefix: Path prefix where the accuracy text files are stored
        output_path: Path to save the figure
        
    Returns:
        The matplotlib figure object
    """
    import os
    import re
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Define font sizes for consistency
    title_font_size = 16
    label_font_size = 14
    annot_font_size = 12
    
    # Lists to store data
    layers = []
    train_accuracies = []
    val_accuracies = []
    
    # Find all accuracy files in the directory
    train_pattern = re.compile(r"train_accuracy_(\d+)_layers\.txt")
    val_pattern = re.compile(r"val_accuracy_(\d+)_layers\.txt")
    
    # Process training accuracy files
    for filename in os.listdir(PATH_PREFIX):
        train_match = train_pattern.match(filename)
        if train_match:
            layer_num = int(train_match.group(1))
            with open(os.path.join(PATH_PREFIX, filename), 'r') as f:
                content = f.read()
                # Extract accuracy value using regex
                acc_match = re.search(r"Train accuracy: ([\d\.]+)", content)
                if acc_match:
                    accuracy = float(acc_match.group(1))
                    layers.append(layer_num)
                    train_accuracies.append(accuracy)
    
    # Create dictionary to store layer -> train_accuracy mapping
    layer_to_train = {layer: acc for layer, acc in zip(layers, train_accuracies)}
    
    # Reset lists for validation processing
    layers = []
    val_accuracies = []
    
    # Process validation accuracy files
    for filename in os.listdir(PATH_PREFIX):
        val_match = val_pattern.match(filename)
        if val_match:
            layer_num = int(val_match.group(1))
            with open(os.path.join(PATH_PREFIX, filename), 'r') as f:
                content = f.read()
                # Extract accuracy value using regex
                acc_match = re.search(r"Val accuracy: ([\d\.]+)", content)
                if acc_match:
                    accuracy = float(acc_match.group(1))
                    layers.append(layer_num)
                    val_accuracies.append(accuracy)

    print(f"Train accuracies: {train_accuracies}")
    print(f"Val accuracies: {val_accuracies}")
    
    # Create dictionary to store layer -> val_accuracy mapping
    layer_to_val = {layer: acc for layer, acc in zip(layers, val_accuracies)}
    
    # Get unique sorted layer numbers from both train and val
    all_layers = sorted(set(list(layer_to_train.keys()) + list(layer_to_val.keys())))
    
    # Create data for DataFrame
    data = {
        'Layer': [],
        'Accuracy': [],
        'Type': []
    }
    
    # Fill in the data
    for layer in all_layers:
        if layer in layer_to_train:
            data['Layer'].append(layer)
            data['Accuracy'].append(layer_to_train[layer])
            data['Type'].append('Training')
        
        if layer in layer_to_val:
            data['Layer'].append(layer)
            data['Accuracy'].append(layer_to_val[layer])
            data['Type'].append('Validation')
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by layer number to ensure sequential order
    df = df.sort_values(by=['Layer', 'Type'])
    
    # Set up the figure
    plt.figure(figsize=(12, 4))
    
    # Set seaborn style
    sns.set_style("white")
    
    # Create grouped bar chart
    ax = sns.barplot(
        x='Layer',
        y='Accuracy',
        hue='Type',
        data=df,
        palette=['#1f77b4', '#ff7f0e']  # Blue for training, Orange for validation
    )
    
    # Customize the plot
    plt.title('SET Model Accuracy Comparison: Training vs Validation',
              fontsize=title_font_size)
    plt.xlabel('Number of Layers', fontsize=label_font_size)
    plt.ylabel('Accuracy', fontsize=label_font_size)
    plt.xticks(fontsize=label_font_size)
    plt.yticks(fontsize=label_font_size)
    
    # Set the y-axis to start at 0
    # plt.ylim(0, max(df['Accuracy']) * 1.1)  # Add 10% padding at the top
    min_acc = 0.75 * 0.99  # Add 1% padding below
    max_acc = 1 * 1.03  # Add 1% padding above

    # Set custom y-axis limits that focus on the relevant range
    plt.ylim(min_acc, max_acc)

    # Add value annotations on top of bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        # Only add text if the height is significant
        if height > 0.001:  # Skip annotations for very small or zero values
            ax.text(
                p.get_x() + p.get_width() / 2.,
                height + 0.005,
                f'{height:.3f}',
                ha='center',
                fontsize=annot_font_size
            )
    
    # Adjust legend
    plt.legend(title='Accuracy Type', fontsize=label_font_size,
               title_fontsize=label_font_size, loc='upper left')
    
    # # Add grid lines for better readability
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Tight layout to ensure everything fits
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    output_path=f"COMPLETE_FIGS/paper/set_model/accuracy_comparison_chart.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    create_accuracy_comparison_chart()

    # all_neuron_histogram_activation_breakdown()
    # plot_mlp_layer3_weights()
    # sns.set_style("white")
    # plot_consolidated_attribute_metrics(
    #     layers=[0, 1, 2, 3],
    #     tokenizer_path=GPTConfig44_Complete().tokenizer_path,
    #     project_name="attr_from_last_attr_binding_seeded200",
    #     loss_range=[0, 0.65],
    #     acc_range=[0.65, 1],
    # )

    # plot_consolidated_attribute_metrics(
    #     layers=[0, 1, 2, 3],
    #     tokenizer_path=GPTConfig44_Complete().tokenizer_path,
    #     project_name="attr_from_answer200",
    #     loss_range=[0, 0.65],
    #     acc_range=[0.65, 1],
    # )

    # plot_orig_set_model_loss_seeded()

    # plot_orig_set_model_loss()

    # plot_consolidated_same_card(
    #     layers=[0, 1, 2, 3],
    #     tokenizer_path=GPTConfig44_Complete().tokenizer_path,
    # )

    # attr_from_last_attr_dataset_size()
    # sns.set_style("white")
    # plot_consolidated_attribute_metrics(
    #     layers=[0, 1, 2, 3],
    #     tokenizer_path=GPTConfig44_Complete().tokenizer_path
    # )
    # sns.set_style("white")
    # plot_consolidated_attribute_metrics(
    #     layers=[0, 1, 2, 3],
    #     tokenizer_path=GPTConfig44_Complete().tokenizer_path,
    #     project_name="attr_from_answer",
    #     loss_range=[0, 0.63],
    #     acc_range=[0.65, 1],
    # )

    # project = "attr_from_answer"
    # plot_all_layers_metrics(
    #     layers=[0, 1, 2, 3],
    #     tokenizer_path=config.tokenizer_path,
    #     project_name=project,
    #     loss_range=[0, 0.63],
    #     acc_range=[0.65, 1]
    #     )

    # run_probe_weight_loss_fig()
    # create_test_accuracy_visualization()

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
