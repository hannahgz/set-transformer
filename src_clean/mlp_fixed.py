from model import GPTConfig44_Complete, GPT
from data_utils import initialize_loaders, pretty_print_input
from tokenizer import load_tokenizer
import torch
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

device = "cuda" if torch.cuda.is_available() else "cpu"
PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'


def analyze_mlp_activations(model, input_data, layer_idx=0, position='both'):
    """
    Analyze MLP activations at specified positions.

    Parameters:
        model: The GPT model
        input_data: Input tensor of shape (batch_size, sequence_length)
        layer_idx: Index of the transformer layer to analyze
        position: One of 'hidden', 'output', or 'both'

    Returns:
        Dictionary with activations at requested positions
    """
    # Put model in eval mode
    model.eval()

    # Dictionary to store activations
    activations = {}

    # Hook function for hidden layer (after GELU)
    def hidden_hook(module, input, output):
        # Store a copy of the activations
        activations['hidden'] = output.detach().clone()

    # Hook function for output layer (after second linear layer)
    def output_hook(module, input, output):
        # Store a copy of the activations
        activations['output'] = output.detach().clone()

    # Register hooks based on what we want to capture
    hooks = []

    if position in ['hidden', 'both']:
        # Hook on the GELU to capture its output (the 256-dimensional hidden activations)
        hidden_hook_handle = model.transformer.h[layer_idx].mlp.gelu.register_forward_hook(
            hidden_hook)
        hooks.append(hidden_hook_handle)

    if position in ['output', 'both']:
        # Hook on the c_proj layer to capture its output (the 64-dimensional output)
        output_hook_handle = model.transformer.h[layer_idx].mlp.c_proj.register_forward_hook(
            output_hook)
        hooks.append(output_hook_handle)

    # Forward pass
    with torch.no_grad():
        _ = model(input_data, get_loss=False)

    # Remove all hooks
    for hook in hooks:
        hook.remove()

    return activations


def analyze_mlp_neurons(model, data_loader, layer_idx=0, neuron_indices=None, mode='hidden', position_slice=None):
    """
    Analyze specific neurons in the MLP's hidden or output layer, preserving position information.
    
    Parameters:
        model: The GPT model
        data_loader: DataLoader yielding batches of input data
        layer_idx: Which transformer layer to analyze
        neuron_indices: List of neuron indices to analyze (None for all)
        mode: 'hidden' for 256-dim hidden neurons, 'output' for 64-dim output neurons
        position_slice: Slice object for positions to analyze (default: slice(-1, None) for last position only)
    
    Returns:
        Dictionary mapping neuron indices to their activation distributions, with position information preserved
    """
    model.eval()
    
    # Determine dimensions based on mode
    n_dims = 256 if mode == 'hidden' else 64
    
    # If no specific neurons provided, analyze all
    if neuron_indices is None:
        neuron_indices = list(range(n_dims))
    
    # Set default position slice if none specified
    if position_slice is None:
        position_slice = slice(-49, None)  # Default to the whole sequence
    
    # We need to determine how many positions we're analyzing
    # This will be initialized with the first batch
    num_positions = None
    
    # Initialize dictionary for each neuron, structure will be determined after first batch
    neuron_activations = {idx: None for idx in neuron_indices}
    
    # Process each batch
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}/{len(data_loader)}")
        
        batch = batch.to(device)
        
        # Get activations for this batch
        activations_dict = analyze_mlp_activations(
            model, batch, layer_idx, position='both')
        
        # Extract the activations we're interested in
        act_tensor = activations_dict['hidden'] if mode == 'hidden' else activations_dict['output']
        
        # Get the sliced positions
        positions_tensor = act_tensor[:, position_slice, :]
        
        # Determine number of positions if this is the first batch
        if num_positions is None:
            num_positions = positions_tensor.shape[1]
            # Now initialize the proper structure
            for idx in neuron_indices:
                neuron_activations[idx] = [[] for _ in range(num_positions)]


        # For each neuron of interest, collect its activations by position
        for neuron_idx in neuron_indices:
            # Get activations for this neuron (shape: [batch_size, num_positions])
            neuron_acts = positions_tensor[:, :, neuron_idx].cpu().numpy()
            
            # For each position, extend the corresponding list
            for pos_idx in range(num_positions):
                neuron_activations[neuron_idx][pos_idx].extend(neuron_acts[:, pos_idx].tolist())
    return neuron_activations

def find_peaks_with_edges(hist, height, distance, prominence):
    # First find regular peaks
    peaks, _ = find_peaks(
        hist, height=height, distance=distance, prominence=prominence)
    
    # Check for edge peaks
    # Left edge
    if hist[0] > hist[1] and hist[0] > height:
        peaks = np.append([0], peaks)
        
    # Right edge
    if hist[-1] > hist[-2] and hist[-1] > height:
        peaks = np.append(peaks, [len(hist)-1])
        
    return peaks

def check_peak_threshold(neuron_activations, neuron_idx, pos_idx, num_bins = 50, peak_threshold=2):
    activations = neuron_activations[neuron_idx][pos_idx]

    # plt.figure(figsize=(10, 6))
    hist, bin_edges = np.histogram(activations, bins=num_bins)

    min_peak_height = 0.04
    min_peak_distance = 0.03
    prominence = 0.03
    num_bins = 50

    distance = int(min_peak_distance * num_bins)  # Convert to bin count
    height = min_peak_height * hist.max()  # Minimum height threshold
    prominence = prominence * hist.max()  # Minimum prominence

    # Find peaks in the histogram
    peaks = find_peaks_with_edges(
        hist, 
        height=height, 
        distance=distance,
        prominence=prominence)

    if len(peaks) >= peak_threshold:
        return True, len(peaks)
    
    return False, len(peaks)

def plot_neuron_activations(neuron_activations, neuron_idx, pos_idx, num_bins=50):
    """Plot histogram of activations for a specific neuron"""
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    activations = neuron_activations[neuron_idx][pos_idx]

    plt.figure(figsize=(10, 6))
    hist, bin_edges = np.histogram(activations, bins=num_bins)

    # Find peaks in the histogram
    peaks, _ = find_peaks(hist, height=0.1*hist.max(), distance=num_bins/10)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot the histogram with identified peaks
    fig = plt.figure(figsize=(12, 6))
    plt.bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), alpha=0.7,
            color='skyblue', edgecolor='black')

    # Mark the peaks
    plt.plot(bin_centers[peaks], hist[peaks], "x", color='red', markersize=10)
    for i, peak in enumerate(peaks):
        plt.text(bin_centers[peak], hist[peak], f"Peak {i+1}",
                 fontsize=10, ha='center', va='bottom')

    # title_suffix = f" - {set_type_filter} Set" if set_type_filter is not None else " - All Sets Combined"
    plt.title(
        f'Neuron {neuron_idx} - Activation Histogram with {len(peaks)} Peaks')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)

    # # Mark the peaks
    # plt.plot(bin_centers[peaks], hist[peaks], "x", color='red', markersize=10)

    # plt.title(f'Neuron {neuron_idx} Activation Distribution')
    # plt.xlabel('Activation Value')
    # plt.ylabel('Frequency')
    # plt.grid(alpha=0.3)
    # plt.show()

    # return plt.gcf()
    return fig
    # return plt.gcf(), bin_centers[peaks], hist[peaks]

if __name__ == "__main__":
    # config = GPTConfig44_Complete()
    # # Load the checkpoint
    # checkpoint = torch.load(config.filename, weights_only=False)

    # # Create the model architecture
    # model = GPT(config).to(device)

    # # Load the weights
    # model.load_state_dict(checkpoint['model'])
    # model.eval()  # Set to evaluation mode

    # # dataset = torch.load(config.dataset_path)
    # # _, val_loader = initialize_loaders(config, dataset)

    # dataset_path = f"{PATH_PREFIX}/base_card_randomization_tuple_randomization_dataset.pth"
    # dataset = torch.load(dataset_path)
    # _, val_loader = initialize_loaders(config, dataset)

    # curr_layer = 0
    # neuron_activations = analyze_mlp_neurons(
    #     model=model, 
    #     data_loader=val_loader, 
    #     layer_idx=0, 
    #     neuron_indices=None, 
    #     mode='hidden',
    #     position_slice=slice(-8,None))

    # output_dir = f"{PATH_PREFIX}/data/mlp/layer{curr_layer}"
    # # Make sure the output directory exists
    # os.makedirs(output_dir, exist_ok=True)
    
    # # Create timestamp for the filename
    # pkl_filename = os.path.join(output_dir, f"neuron_activations_layer{curr_layer}.pkl")
    
    # # Save the activations to a pickle file
    # with open(pkl_filename, 'wb') as f:
    #     pickle.dump(neuron_activations, f)
    
    # print(f"Saved neuron activations to {pkl_filename}")

    curr_layer = 0
    output_dir = f"{PATH_PREFIX}/data/mlp/layer{curr_layer}"
    pkl_filename = os.path.join(output_dir, f"neuron_activations_layer{curr_layer}.pkl")
    with open(pkl_filename, 'rb') as f:
        neuron_activations = pickle.load(f)

    # for neuron_idx in range(256):
    for neuron_idx in [21, 71, 88, 97, 111, 118, 130, 148, 161, 164, 191, 214]:
        for pos_idx in range(8):
            at_threshold, num_peaks = check_peak_threshold(neuron_activations, neuron_idx, pos_idx)
            if at_threshold:
                print(f"Neuron {neuron_idx}, position {pos_idx}, num_peaks {num_peaks}")
            # if check_peak_threshold(neuron_activations, neuron_idx, pos_idx):
            #     print(f"Neuron {neuron_idx}, position {pos_idx} has enough peaks")
            fig = plot_neuron_activations(neuron_activations, neuron_idx, pos_idx)
            peaks_dir = f"results/mlp_fixed/peaks/layer{curr_layer}/neuron{neuron_idx}"
            os.makedirs(peaks_dir, exist_ok=True)
            fig.savefig(f"{peaks_dir}/pos{pos_idx}_hist.png")
    # # Plot the histogram of activations for a specific neuron
    # neuron_idx = 0
    # # pos_idx = 0
    # for pos_idx in range(8):
    #     print(f"Plotting histogram for neuron {neuron_idx}, position {pos_idx}")
    #     fig = plot_neuron_activations(neuron_activations, neuron_idx, pos_idx)
    #     peaks_dir = f"results/mlp_fixed/peaks/layer{curr_layer}/neuron{neuron_idx}"
    #     os.makedirs(peaks_dir, exist_ok=True)
    #     fig.savefig(f"{peaks_dir}/pos{pos_idx}_hist.png")


    

