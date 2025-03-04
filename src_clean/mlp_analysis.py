from model import GPTConfig44_Complete, GPT
from data_utils import initialize_loaders, pretty_print_input
from tokenizer import load_tokenizer
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

def plot_neuron_activation_histograms(model, data_loader, target_neurons, config, set_filtering = None, num_bins=50, figsize=(60, 40), save_pkl=True, output_dir='./'):
    """
    Compute and plot histograms of activations for specified neurons.
    
    Parameters:
        model: The GPT model
        data_loader: DataLoader yielding batches of input sequences compatible with the model
        target_neurons: List of neuron indices to analyze
        device: Device to run the model on
        num_bins: Number of bins for the histogram
        figsize: Figure size as (width, height)
        save_pkl: Whether to save the neuron activations to a pickle file
        output_dir: Directory to save the pickle file in
    
    Returns:
        fig: The matplotlib figure object containing the histograms
        neuron_activations: Dictionary mapping neuron indices to their activation values
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import pickle
    import os
    from datetime import datetime
    
    model.eval()
    
    # Initialize tracking structures for activations
    neuron_activations = {neuron: [] for neuron in target_neurons}
    
    tokenizer = load_tokenizer(config.tokenizer_path)
    no_set_token = tokenizer.token_to_id["*"]
    two_set_token = tokenizer.token_to_id["/"]

    # Process each batch in the data loader
    print(f"Computing activations for {len(target_neurons)} neurons...")
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
                if set_filtering == 0:
                    if no_set_token not in batch[seq_idx]:
                        continue
                elif set_filtering == 1:
                    if no_set_token in batch[seq_idx] or two_set_token in batch[seq_idx]:
                        continue
                elif set_filtering == 2:
                    if two_set_token not in batch[seq_idx]:
                        continue
                final_pos_activations = layer_3_activations[seq_idx, -1, :]
                
                # Record activation for each target neuron
                for neuron in target_neurons:
                    activation = final_pos_activations[neuron].item()
                    neuron_activations[neuron].append(activation)
    
    # Save neuron activations to pickle file if requested
    if save_pkl:
        # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pkl_filename = os.path.join(output_dir, f"neuron_activations_{timestamp}.pkl")
        
        # Save the activations to a pickle file
        with open(pkl_filename, 'wb') as f:
            pickle.dump(neuron_activations, f)
        
        print(f"Saved neuron activations to {pkl_filename}")
    
    # Calculate rows and columns for subplot grid
    n_neurons = len(target_neurons)
    n_cols = min(4, n_neurons)  # At most 3 columns
    n_rows = (n_neurons + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easy indexing if there are multiple subplots
    if n_neurons > 1:
        if n_rows > 1 and n_cols > 1:
            axes = axes.flatten()
        elif n_rows == 1 or n_cols == 1:
            axes = np.array([axes]).flatten()  # Handle case of 1D array
    else:
        axes = [axes]  # Make it iterable when there's only one subplot
    
    print("Plotting histograms...")
    # Plot histogram for each neuron
    for i, neuron in enumerate(target_neurons):
        # Extract activation values
        activations = neuron_activations[neuron]
        
        # Calculate histogram statistics
        mean_act = np.mean(activations)
        median_act = np.median(activations)
        std_act = np.std(activations)
        min_act = np.min(activations)
        max_act = np.max(activations)
        
        # Plot histogram
        axes[i].hist(activations, bins=num_bins, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].axvline(mean_act, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_act:.3f}')
        axes[i].axvline(median_act, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_act:.3f}')
        
        # Set titles and labels
        axes[i].set_title(f'Neuron {neuron} Activations')
        axes[i].set_xlabel('Activation Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        
        # Add text with activation statistics
        text_info = (f'μ = {mean_act:.3f}\nσ = {std_act:.3f}\n'
                     f'Min = {min_act:.3f}\nMax = {max_act:.3f}')
        axes[i].text(0.95, 0.95, text_info, transform=axes[i].transAxes, 
                     fontsize=10, va='top', ha='right', 
                     bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
    
    # Remove any unused subplots
    for i in range(n_neurons, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    print(f"Done! Plotted histograms for {len(target_neurons)} neurons.")
    
    return fig, neuron_activations

def plot_neuron_activation_histograms_overlap(model, data_loader, target_neurons, config, capture_layer, set_filtering=None, num_bins=50, figsize=(60, 40), save_pkl=True, output_dir='./'):
    """
    Compute and plot histograms of activations for specified neurons with overlapping histograms for different set filtering types.
    
    Parameters:
        model: The GPT model
        data_loader: DataLoader yielding batches of input sequences compatible with the model
        target_neurons: List of neuron indices to analyze
        config: Configuration object containing model settings
        set_filtering: If None, will plot all three filtering types overlapping. Otherwise, specify a single value (0, 1, or 2)
        num_bins: Number of bins for the histogram
        figsize: Figure size as (width, height)
        save_pkl: Whether to save the neuron activations to a pickle file
        output_dir: Directory to save the pickle file in
    
    Returns:
        fig: The matplotlib figure object containing the histograms
        neuron_activations: Dictionary mapping neuron indices to their activation values for each set type
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import pickle
    import os
    from datetime import datetime
    
    model.eval()
    
    # Define set filtering types and their colors
    set_types = {
        0: {"name": "No Set", "color": "blue", "alpha": 0.5},
        1: {"name": "One Set", "color": "green", "alpha": 0.5},
        2: {"name": "Two Set", "color": "red", "alpha": 0.5}
    }
    
    # Determine which set types to process
    if set_filtering is None:
        set_types_to_process = [0, 1, 2]  # Process all three types
    else:
        set_types_to_process = [set_filtering]  # Process only the specified type
    
    # Initialize tracking structures for activations for each set type
    neuron_activations = {
        neuron: {set_type: [] for set_type in set_types_to_process} 
        for neuron in target_neurons
    }
    
    tokenizer = load_tokenizer(config.tokenizer_path)
    no_set_token = tokenizer.token_to_id["*"]
    two_set_token = tokenizer.token_to_id["/"]

    # Process each batch in the data loader
    print(f"Computing activations for {len(target_neurons)} neurons...")
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}/{len(data_loader)}")
        
        batch = batch.to(device)
        batch_size = batch.shape[0]
        
        with torch.no_grad():
            _, _, _, layer_activations, _ = model(batch, capture_layer=capture_layer)
            
            # For each sequence in the batch
            for seq_idx in range(batch_size):
                sequence = batch[seq_idx]
                
                # Determine which set type this sequence belongs to
                current_set_type = None
                if no_set_token in sequence:
                    current_set_type = 0  # No Set
                elif two_set_token in sequence:
                    current_set_type = 2  # Two Set
                else:
                    current_set_type = 1  # One Set
                
                # Skip if we're not processing this set type
                if current_set_type not in set_types_to_process:
                    continue
                
                # Extract final position's activations for this sequence
                final_pos_activations = layer_activations[seq_idx, -1, :]
                
                # Record activation for each target neuron
                for neuron in target_neurons:
                    activation = final_pos_activations[neuron].item()
                    neuron_activations[neuron][current_set_type].append(activation)
    
    # # Save neuron activations to pickle file if requested
    # if save_pkl:
    #     # Make sure the output directory exists
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     # Create timestamp for the filename
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     pkl_filename = os.path.join(output_dir, f"neuron_activations_{timestamp}.pkl")
        
    #     # Save the activations to a pickle file
    #     with open(pkl_filename, 'wb') as f:
    #         pickle.dump(neuron_activations, f)
        
    #     print(f"Saved neuron activations to {pkl_filename}")
    
    # Calculate rows and columns for subplot grid
    n_neurons = len(target_neurons)
    n_cols = min(4, n_neurons)  # At most 4 columns
    n_rows = (n_neurons + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easy indexing if there are multiple subplots
    if n_neurons > 1:
        if n_rows > 1 and n_cols > 1:
            axes = axes.flatten()
        elif n_rows == 1 or n_cols == 1:
            axes = np.array([axes]).flatten()  # Handle case of 1D array
    else:
        axes = [axes]  # Make it iterable when there's only one subplot
    
    print("Plotting histograms...")
    # Plot histogram for each neuron
    for i, neuron in enumerate(target_neurons):
        # Store statistics for the combined data
        all_activations = []
        
        # Plot histogram for each set type
        for set_type in set_types_to_process:
            activations = neuron_activations[neuron][set_type]
            if not activations:  # Skip if no data for this set type
                continue
                
            all_activations.extend(activations)
            
            # Calculate histogram statistics for this set type
            mean_act = np.mean(activations)
            median_act = np.median(activations)
            
            set_info = set_types[set_type]
            
            # Plot histogram with alpha transparency to show overlap
            axes[i].hist(activations, bins=num_bins, alpha=set_info["alpha"], 
                         color=set_info["color"], edgecolor='black', 
                         label=f'{set_info["name"]} (n={len(activations)})')
            
            # Add vertical lines for mean values
            axes[i].axvline(mean_act, color=set_info["color"], linestyle='dashed', linewidth=2, 
                           label=f'{set_info["name"]} Mean: {mean_act:.3f}')
        
        # Calculate overall statistics
        if all_activations:
            overall_mean = np.mean(all_activations)
            overall_std = np.std(all_activations)
            min_act = np.min(all_activations)
            max_act = np.max(all_activations)
            
            # Add text with overall activation statistics
            text_info = (f'Overall:\nμ = {overall_mean:.3f}\nσ = {overall_std:.3f}\n'
                         f'Min = {min_act:.3f}\nMax = {max_act:.3f}')
            axes[i].text(0.95, 0.95, text_info, transform=axes[i].transAxes, 
                         fontsize=10, va='top', ha='right', 
                         bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
        
        # Set titles and labels
        axes[i].set_title(f'Neuron {neuron} Activations')
        axes[i].set_xlabel('Activation Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend(loc='upper left', fontsize=8)
    
    # Remove any unused subplots
    for i in range(n_neurons, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    print(f"Done! Plotted histograms for {len(target_neurons)} neurons with overlapping set types.")
    
    return fig, neuron_activations

def plot_neuron_activation_histograms_overlap_preloaded(target_neurons, capture_layer, set_filtering=None, num_bins=50, figsize=(60, 40)):
    """
    Compute and plot histograms of activations for specified neurons with overlapping histograms for different set filtering types.
    
    Parameters:
        model: The GPT model
        data_loader: DataLoader yielding batches of input sequences compatible with the model
        target_neurons: List of neuron indices to analyze
        config: Configuration object containing model settings
        set_filtering: If None, will plot all three filtering types overlapping. Otherwise, specify a single value (0, 1, or 2)
        num_bins: Number of bins for the histogram
        figsize: Figure size as (width, height)
        save_pkl: Whether to save the neuron activations to a pickle file
        output_dir: Directory to save the pickle file in
    
    Returns:
        fig: The matplotlib figure object containing the histograms
        neuron_activations: Dictionary mapping neuron indices to their activation values for each set type
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    
    # Define set filtering types and their colors
    set_types = {
        0: {"name": "No Set", "color": "blue", "alpha": 0.5},
        1: {"name": "One Set", "color": "green", "alpha": 0.5},
        2: {"name": "Two Set", "color": "red", "alpha": 0.5}
    }
    
    # Determine which set types to process
    if set_filtering is None:
        set_types_to_process = [0, 1, 2]  # Process all three types
    else:
        set_types_to_process = [set_filtering]  # Process only the specified type
    
    # Initialize tracking structures for activations for each set type
    neuron_activations = {
        neuron: {set_type: [] for set_type in set_types_to_process} 
        for neuron in target_neurons
    }
    
    # Load neuron activations from pickle file
    pkl_filename = f"neuron_activations_{capture_layer}.pkl"
    with open(pkl_filename, 'rb') as f:
        neuron_activations = pickle.load(f)

    # Calculate rows and columns for subplot grid
    n_neurons = len(target_neurons)
    n_cols = min(4, n_neurons)  # At most 4 columns
    n_rows = (n_neurons + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easy indexing if there are multiple subplots
    if n_neurons > 1:
        if n_rows > 1 and n_cols > 1:
            axes = axes.flatten()
        elif n_rows == 1 or n_cols == 1:
            axes = np.array([axes]).flatten()  # Handle case of 1D array
    else:
        axes = [axes]  # Make it iterable when there's only one subplot
    
    print("Plotting histograms...")
    # Plot histogram for each neuron
    for i, neuron in enumerate(target_neurons):
        # Store statistics for the combined data
        all_activations = []
        
        # Plot histogram for each set type
        for set_type in set_types_to_process:
            activations = neuron_activations[neuron][set_type]
            if not activations:  # Skip if no data for this set type
                continue
                
            all_activations.extend(activations)
            
            # Calculate histogram statistics for this set type
            mean_act = np.mean(activations)
            median_act = np.median(activations)
            
            set_info = set_types[set_type]
            
            # Plot histogram with alpha transparency to show overlap
            axes[i].hist(activations, bins=num_bins, alpha=set_info["alpha"], 
                         color=set_info["color"], edgecolor='black', 
                         label=f'{set_info["name"]} (n={len(activations)})')
            
            # Add vertical lines for mean values
            axes[i].axvline(mean_act, color=set_info["color"], linestyle='dashed', linewidth=2, 
                           label=f'{set_info["name"]} Mean: {mean_act:.3f}')
        
        # Calculate overall statistics
        if all_activations:
            overall_mean = np.mean(all_activations)
            overall_std = np.std(all_activations)
            min_act = np.min(all_activations)
            max_act = np.max(all_activations)
            
            # Add text with overall activation statistics
            text_info = (f'Overall:\nμ = {overall_mean:.3f}\nσ = {overall_std:.3f}\n'
                         f'Min = {min_act:.3f}\nMax = {max_act:.3f}')
            axes[i].text(0.95, 0.95, text_info, transform=axes[i].transAxes, 
                         fontsize=10, va='top', ha='right', 
                         bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
        
        # Set titles and labels
        axes[i].set_title(f'Neuron {neuron} Activations')
        axes[i].set_xlabel('Activation Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend(loc='upper left', fontsize=8)
    
    # Remove any unused subplots
    for i in range(n_neurons, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    print(f"Done! Plotted histograms for {len(target_neurons)} neurons with overlapping set types.")
    
    return fig, neuron_activations

if __name__ == "__main__":
    config = GPTConfig44_Complete()

    # dataset_path = f"{PATH_PREFIX}/base_card_randomization_tuple_randomization_dataset.pth"
    # dataset = torch.load(dataset_path)
    # _, val_loader = initialize_loaders(config, dataset)

    target_neurons = range(64)

    # # Create the model architecture
    # model = GPT(config).to(device)
    # checkpoint = torch.load(config.filename, weights_only = False)
    # # Load the weights
    # model.load_state_dict(checkpoint['model'])
    # model.eval()  # Set to evaluation mode


    for capture_layer in range(4):
        for target_sets in [0, 1, 2]:
            fig, neuron_acts = plot_neuron_activation_histograms_overlap_preloaded(
                target_neurons,
                capture_layer=capture_layer,
                set_filtering=target_sets,
            )
        plt.savefig(f'COMPLETE_FIGS/mlp/layer_{capture_layer}/neuron_activation_histograms_set{target_sets}.png', dpi=300, bbox_inches="tight")

    # for set_filtering in [0, 1, 2]:
    #     fig, neuron_acts = plot_neuron_activation_histograms(
    #         model, 
    #         val_loader, 
    #         target_neurons,
    #         config,
    #         set_filtering = set_filtering,
    #     )
    #     plt.savefig(f'COMPLETE_FIGS/mlp/neuron_activation_histograms_set{set_filtering}.png', dpi=300, bbox_inches="tight")
    # plt.show()

    # print("Extracting mlp weights")
    # model, mlp_weights = load_model_and_extract_mlp_weights(config)
    # mlp_weights_save_path = f"{PATH_PREFIX}/mlp_triples_card_randomization_tuple_randomization_layers_4_heads_4.pt"
    # torch.save(mlp_weights, mlp_weights_save_path)

    # mlp_weights = torch.load(f"{PATH_PREFIX}/mlp_triples_card_randomization_tuple_randomization_layers_4_heads_4.pt")
    # visualize_mlp_weights(mlp_weights, f"COMPLETE_FIGS/mlp")

    # config = GPTConfig44_Complete()
    # # Load the checkpoint
    # checkpoint = torch.load(config.filename, weights_only = False)
    
    # # Create the model architecture
    # model = GPT(config).to(device)
    
    # # Load the weights
    # model.load_state_dict(checkpoint['model'])
    # model.eval()  # Set to evaluation mode

    # dataset = torch.load(config.dataset_path)
    # _, val_loader = initialize_loaders(config, dataset)

    # target_neurons = [5, 13, 20, 36, 60]
    # top_k_inputs, bottom_k_inputs = analyze_concept_neurons(model, val_loader, target_neurons)

    # # Save top_k_inputs and bottom_k_inputs as pickle files
    # with open(f"top_k_inputs.pkl", "wb") as f:
    #     pickle.dump(top_k_inputs, f)

    # with open(f"bottom_k_inputs.pkl", "wb") as f:
    #     pickle.dump(bottom_k_inputs, f)

    # print("Top-k Inputs:")
    # for neuron, inputs in top_k_inputs.items():
    #     print(f"Neuron {neuron}:")
    #     for activation, input_seq in inputs:
    #         print(f"Activation: {activation}")
    #         print(f"Input Sequence: {input_seq}")
    #         print()
    
    # breakpoint()


    # config = GPTConfig44_Complete()
    # # Load top_k_inputs and bottom_k_inputs from pickle files
    # with open("top_k_inputs.pkl", "rb") as f:
    #     top_k_inputs = pickle.load(f)

    # with open("bottom_k_inputs.pkl", "rb") as f:
    #     bottom_k_inputs = pickle.load(f)

    # tokenizer = load_tokenizer("all_tokenizer.pkl")
    # for neuron, inputs in top_k_inputs.items():
    #     print(f"Neuron {neuron}:")
    #     for activation, input_seq in inputs:
    #         input_seq = input_seq.tolist()
    #         print(f"Activation: {activation}")
    #         tokenized_seq = tokenizer.decode(input_seq)
    #         print(pretty_print_input(tokenized_seq))
    #         print(f"Sets: {tokenized_seq[41:]}")
    #         # print()

    # print()
    # for neuron, inputs in bottom_k_inputs.items():
    #     print(f"Neuron {neuron}:")
    #     for activation, input_seq in inputs:
    #         input_seq = input_seq.tolist()
    #         print(f"Activation: {activation}")
    #         tokenized_seq = tokenizer.decode(input_seq)
    #         print(pretty_print_input(tokenized_seq))
    #         print(f"Sets: {tokenized_seq[41:]}")
    #         # print()

    


