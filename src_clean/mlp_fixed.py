from model import GPTConfig44_Complete, GPT
from data_utils import initialize_loaders, pretty_print_input
from tokenizer import load_tokenizer
import torch
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from mlp_hist_peak import save_top_peak_examples_as_txt, save_peak_figure, save_summary_statistics_from_peak_info

device = "cuda" if torch.cuda.is_available() else "cpu"
PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

min_peak_height = 0.04
min_peak_distance = 0.03
prominence = 0.03
num_bins = 50


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


tokenizer = load_tokenizer(GPTConfig44_Complete().tokenizer_path)
no_set_token = tokenizer.token_to_id["*"]
two_set_token = tokenizer.token_to_id["/"]


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
        and separated by set types (0: No Set, 1: One Set, 2: Two Set)
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
            # Now initialize the proper structure with 3 set types
            for idx in neuron_indices:
                # Initialize as [num_positions][3 set types][activations]
                neuron_activations[idx] = [
                    [[] for _ in range(3)] for _ in range(num_positions)]

        # Determine set type for each sequence in the batch
        batch_set_types = []
        with torch.no_grad():
            batch_size = batch.shape[0]
            for seq_idx in range(batch_size):
                sequence = batch[seq_idx]

                # Determine which set type this sequence belongs to
                if no_set_token in sequence:
                    current_set_type = 0  # No Set
                elif two_set_token in sequence:
                    current_set_type = 2  # Two Set
                else:
                    current_set_type = 1  # One Set
                batch_set_types.append(current_set_type)

        # For each neuron of interest, collect its activations by position and set type
        for neuron_idx in neuron_indices:
            # Get activations for this neuron (shape: [batch_size, num_positions])
            neuron_acts = positions_tensor[:, :, neuron_idx].cpu().numpy()

            # For each example in the batch
            for batch_item_idx, set_type in enumerate(batch_set_types):
                # For each position, add to the corresponding set type list
                for pos_idx in range(num_positions):
                    # Convert to standard Python float to avoid np.float32 representation
                    neuron_activations[neuron_idx][pos_idx][set_type].append(
                        float(neuron_acts[batch_item_idx, pos_idx]))

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


def check_peak_threshold(neuron_activations, neuron_idx, pos_idx, num_bins=50, peak_threshold=2):
    activations = neuron_activations[neuron_idx][pos_idx]

    # plt.figure(figsize=(10, 6))
    hist, bin_edges = np.histogram(activations, bins=num_bins)

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

    distance = int(min_peak_distance * num_bins)  # Convert to bin count
    height = min_peak_height * hist.max()  # Minimum height threshold
    prominence = prominence * hist.max()  # Minimum prominence

    # Find peaks in the histogram
    peaks = find_peaks_with_edges(
        hist,
        height=height,
        distance=distance,
        prominence=prominence)

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


def get_peak_info(neuron_activations, layer, position, neuron, input_examples=None, min_peak_height=0.05,
                  min_peak_distance=0.1, prominence=0.02, num_bins=50,
                  set_type_filter=None):
    """
    Identify peaks in the activation histogram for a specific neuron in a specific layer,
    and provide examples of inputs that produce activations at each peak.

    Parameters:
        layer: Layer number to analyze
        neuron: Neuron index to analyze
        input_examples_file: File containing input examples (if None, will not provide examples)
        min_peak_height: Minimum height of peaks to identify (fraction of max height)
        min_peak_distance: Minimum distance between peaks (fraction of activation range)
        prominence: Minimum prominence of peaks (fraction of activation range)
        num_bins: Number of bins for the histogram
        set_type_filter: Specific set type to analyze (e.g., 'train', 'test', 'val'). 
                         If None, all set types will be combined.

    Returns:
        peaks_info: Information about the peaks, including indices, heights, and examples
    """
    # Load activation data from the pickle file
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # Combine activations from all set types or filter by specified set type
    all_activations = []
    activation_by_type = {}

    if set_type_filter is not None:
        # Only use the specified set type
        activations = neuron_activations[neuron][position][set_type_filter]
        all_activations.extend(activations)
        activation_by_type[set_type_filter] = activations
        print(
            f"Analyzing {len(all_activations)} activations for neuron {neuron} in layer {layer} (set type: {set_type_filter})")
    else:
        # Use all available set types
        # for set_type in neuron_activations[neuron][position]:
        for set_type in range(3):
            # breakpoint()
            activations = neuron_activations[neuron][position][set_type]
            all_activations.extend(activations)
            activation_by_type[set_type] = activations
        print(
            f"Analyzing {len(all_activations)} total activations for neuron {neuron} in layer {layer} (all set types combined)")

    # Calculate histogram
    hist, bin_edges = np.histogram(all_activations, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize histogram for visualization
    hist_normalized = hist / hist.max()

    # Calculate range for peak distance and prominence parameters
    activation_range = max(all_activations) - min(all_activations)
    distance = int(min_peak_distance * num_bins)  # Convert to bin count
    height = min_peak_height * hist.max()  # Minimum height threshold
    prominence = prominence * hist.max()  # Minimum prominence

    # Find peaks in the histogram
    # peaks, peak_properties = find_peaks(
    #     hist, height=height, distance=distance, prominence=prom)

    peaks = find_peaks_with_edges(
        hist, height=height, distance=distance, prominence=prominence)

    print(f"Found {len(peaks)} peaks in the histogram")

    # Create a dictionary to store peak information (without figure yet)
    peaks_info = {
        'neuron': neuron,
        'layer': layer,
        'set_type': set_type_filter if set_type_filter is not None else 'all',
        'bin_centers': bin_centers[peaks],
        'heights': hist[peaks],
        'examples_by_peak': defaultdict(list)
    }

    # Plot the histogram with identified peaks
    fig = plt.figure(figsize=(12, 6))
    plt.bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), alpha=0.7,
            color='skyblue', edgecolor='black')

    # Mark the peaks
    plt.plot(bin_centers[peaks], hist[peaks], "x", color='red', markersize=10)
    for i, peak in enumerate(peaks):
        plt.text(bin_centers[peak], hist[peak], f"Peak {i+1}",
                 fontsize=10, ha='center', va='bottom')

    title_suffix = f" - {set_type_filter} Set" if set_type_filter is not None else " - All Sets Combined"
    plt.title(
        f'Neuron {neuron} in Layer {layer} with Position {position}{title_suffix} - Activation Histogram with {len(peaks)} Peaks')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)

    if input_examples:

        # Get the bin for each activation value
        all_activations_with_examples = []

        # If filtering by set type, only use examples from that set type
        if set_type_filter:
            if set_type_filter in input_examples:
                for i, act_value in enumerate(activation_by_type[set_type_filter]):
                    if i < len(input_examples[set_type_filter]):
                        all_activations_with_examples.append(
                            (act_value, input_examples[set_type_filter][i]))
            else:
                print(
                    f"Warning: No input examples found for set type '{set_type_filter}'")
        # Otherwise use examples from all available set types
        else:
            for cur_set_type in activation_by_type:
                if cur_set_type in input_examples:
                    for i, act_value in enumerate(activation_by_type[cur_set_type]):
                        if i < len(input_examples[cur_set_type]):
                            all_activations_with_examples.append(
                                (act_value, input_examples[cur_set_type][i]))

        # Assign each activation to its nearest peak
        peak_values = bin_centers[peaks]
        for act_value, example in all_activations_with_examples:
            # Find the closest peak
            closest_peak_idx = np.abs(peak_values - act_value).argmin()
            # Store the example with its exact activation value
            peaks_info['examples_by_peak'][closest_peak_idx].append(
                (act_value, example))

        # Sort examples by how close they are to the peak value
        for peak_idx in peaks_info['examples_by_peak']:
            peaks_info['examples_by_peak'][peak_idx].sort(
                key=lambda x: abs(x[0] - peak_values[peak_idx]))
            # Keep only the top few examples closest to the peak
            # peaks_info['examples_by_peak'][peak_idx] = peaks_info['examples_by_peak'][peak_idx][:5]

    else:
        # print("No input examples file provided or file not found")
        raise ValueError("No input examples file provided or file not found")

    peaks_info['figure'] = fig

    plt.show()
    return peaks_info

def plot_overlap_histograms(neuron_activations, target_neurons, num_bins=50, figsize=None):
    # Define set filtering types and their colors
    set_types = {
        0: {"name": "No Set", "color": "blue", "alpha": 0.5},
        1: {"name": "One Set", "color": "green", "alpha": 0.5},
        2: {"name": "Two Set", "color": "red", "alpha": 0.5}
    }

    # Calculate rows and columns for subplot grid
    n_neurons = len(target_neurons)
    n_cols = 8  # Each column represents a position
    n_rows = n_neurons  # Each row represents a neuron
    
    # Set default figsize based on number of neurons if not provided
    if figsize is None:
        figsize = (70, 4 * n_neurons)
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle the case of a single neuron (1D array)
    if n_neurons == 1:
        axes = axes.reshape(1, -1)
    
    # Plot histogram for each neuron and position
    for i, neuron in enumerate(target_neurons):
        for pos_idx in range(8):
            ax = axes[i, pos_idx]
            
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
            if pos_idx == 0:  # Add neuron labels only on leftmost column
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
    print(f"Done! Plotted histograms for {len(target_neurons)} neurons with overlapping set types.")
    
    return fig

from itertools import combinations
from collections import Counter

tokenizer = load_tokenizer(f"{PATH_PREFIX}/all_tokenizer.pkl")

shapes = ["oval", "squiggle", "diamond"]
colors = ["green", "blue", "pink"]
numbers = ["one", "two", "three"]
shadings = ["solid", "striped", "open"]

shape_ids = tokenizer.encode(shapes)
color_ids = tokenizer.encode(colors)
number_ids = tokenizer.encode(numbers)
shading_ids = tokenizer.encode(shadings)

card_ids = tokenizer.encode(["A", "B", "C", "D", "E"])

def get_per_card_attributes(input):
    grid = [[0 for _ in range(4)] for _ in range(5)]
    card_indices = {card_ids[0]: 0, card_ids[1]: 1, card_ids[2]: 2, card_ids[3]: 3, card_ids[4]: 4}
    
    i = 0
    while i < 40:
        card = input[i]
        attr = input[i+1]

        if attr in shape_ids:
            grid[card_indices[card]][0] = attr
        elif attr in number_ids:
            grid[card_indices[card]][1] = attr
        elif attr in shading_ids:
            grid[card_indices[card]][2] = attr
        elif attr in color_ids:
            grid[card_indices[card]][3] = attr

        i += 2
    return grid

def overall_count(grid):
    # same_count = []
    # diff_count = []
    same_count = 0
    diff_count = 0

    for comb in combinations(range(5), 3):
        card_one = grid[comb[0]]
        card_two = grid[comb[1]]
        card_three = grid[comb[2]]

        cards = [card_one, card_two, card_three]

        for attribute_type in range(4):
            same = True
            diff = True
            for card in cards:
                if card[attribute_type] != card_one[attribute_type]:
                    same = False
                if card[attribute_type] == card_one[attribute_type]:
                    diff = False
            if same:
                same_count += 1
            if diff:
                diff_count += 1

    return same_count, diff_count

def test_overall_summary_stat(peaks_info, top=None, output_file="overall_summary_statistics.txt",):
    # Open the output file
    same_count_total = 0
    diff_count_total = 0
    total = 0
    with open(output_file, 'w') as f:
        for peak_idx in peaks_info['examples_by_peak']:
            for index, (_, example) in enumerate(peaks_info['examples_by_peak'][peak_idx][:top]):
                grid = get_per_card_attributes(example)
                same_count, diff_count = overall_count(grid)
                same_count_total += same_count
                diff_count_total += diff_count
        
        total = same_count_total + diff_count_total
        # Write the summary statistics to the file
        for peak_idx in sorted(peaks_info['examples_by_peak']):
            f.write(f"\nPeak {peak_idx+1}:\n")
            
            f.write(f"Same Count: {same_count_total}\n")
            f.write(f"Diff Count: {diff_count_total}\n")
            f.write(f"Total: {total}\n")

if __name__ == "__main__":
    input_examples_file = "results/val_input_examples.pkl"
    print(f"Loading input examples from {input_examples_file}")
    with open(input_examples_file, 'rb') as f:
        input_examples = pickle.load(f)

    # layer_0_target_neurons = [148, 164, 191]
    # layer_0_target_neurons = [10, 11, 12, 14, 16, 24]
    # layer_0_target_neurons = [84, 88, 97, 115, 128, 196, 209, 220]
    # target_neurons = [84, 88, 97, 115, 128, 196, 209, 220]
    target_neurons = [220]
    # # for set_type_filter in [0, 1]:

    # for curr_layer in range(4):
    for curr_layer in [0]:
        for set_type_filter in [0, 1, 2]:
            # for neuron in [2, 4, 12, 19, 34, 36, 37, 43, 44, 54, 60, 61]:
            # for neuron in layer_1_target_neurons:
            for neuron in target_neurons:
                for pos_idx in range(8):
                    print(
                        f"Layer {curr_layer}, Set type filter: {set_type_filter}, Neuron: {neuron}, Pos: {pos_idx}")

                    if set_type_filter is None:
                        set_type_filter_name = "all"
                    else:
                        set_type_filter_name = set_type_filter
                    peaks_dir = f"{PATH_PREFIX}/data/mlp_fixed/layer{curr_layer}/neuron{neuron}/pos{pos_idx}/set_type_{set_type_filter_name}"

                    peaks_info_path = os.path.join(peaks_dir, f"{set_type_filter_name}_info.pkl")

                    with open(peaks_info_path, 'rb') as f:
                        peaks_info = pickle.load(f)

                    output_dir = f"results/mlp_fixed/peaks/layer{curr_layer}/neuron{neuron}/pos{pos_idx}/set_type_{set_type_filter_name}"
                    os.makedirs(output_dir, exist_ok=True)

                    top = 100
                    test_overall_summary_stat(
                        peaks_info,
                        top=top,
                        output_file=os.path.join(output_dir, "overall_summary_statistics.txt")
                    )

    # curr_layer = 0
    # output_dir = f"{PATH_PREFIX}/data/mlp_fixed/layer{curr_layer}"
    # os.makedirs(output_dir, exist_ok=True)

    # pkl_filename = os.path.join(
    #     output_dir, f"neuron_activations_layer{curr_layer}.pkl")

    # with open(pkl_filename, 'rb') as f:
    #     neuron_activations = pickle.load(f)

    # for i in range(32):
    #     print(f"Plotting histograms for neurons {i*8} to {(i+1)*8}")
    #     output_dir = f"results/mlp_fixed/peaks/layer{curr_layer}"
    #     os.makedirs(output_dir, exist_ok=True)
    
    #     fig = plot_overlap_histograms(
    #         neuron_activations = neuron_activations,
    #         target_neurons = range(i * 8, (i+1) * 8),
    #         num_bins = 50,
    #     )

    #     plt.savefig(f"{output_dir}/consolidated_overlap_heatmap_{i}.png")


    # config = GPTConfig44_Complete()
    # # Load the checkpoint
    # checkpoint = torch.load(config.filename, weights_only=False)

    # # Create the model architecture
    # model = GPT(config).to(device)

    # # Load the weights
    # model.load_state_dict(checkpoint['model'])
    # model.eval()  # Set to evaluation mode

    # dataset_path = f"{PATH_PREFIX}/base_card_randomization_tuple_randomization_dataset.pth"
    # dataset = torch.load(dataset_path)
    # _, val_loader = initialize_loaders(config, dataset)

    # for curr_layer in [1, 2, 3]:
    #     print(f"Analyzing layer {curr_layer}")
    #     neuron_activations = analyze_mlp_neurons(
    #         model=model,
    #         data_loader=val_loader,
    #         layer_idx=0,
    #         neuron_indices=None,
    #         mode='hidden',
    #         position_slice=slice(-8,None))

    #     output_dir = f"{PATH_PREFIX}/data/mlp_fixed/layer{curr_layer}"
    #     # # Make sure the output directory exists
    #     os.makedirs(output_dir, exist_ok=True)

    #     pkl_filename = os.path.join(
    #         output_dir, f"neuron_activations_layer{curr_layer}.pkl")

    #     # Save the activations to a pickle file
    #     with open(pkl_filename, 'wb') as f:
    #         pickle.dump(neuron_activations, f)

    #     for i in range(32):
    #         print(f"Plotting histograms for neurons {i*8} to {(i+1)*8}")
    #         output_dir = f"results/mlp_fixed/peaks/layer{curr_layer}"
    #         os.makedirs(output_dir, exist_ok=True)
        
    #         fig = plot_overlap_histograms(
    #             neuron_activations = neuron_activations,
    #             target_neurons = range(i * 8, (i+1) * 8),
    #             num_bins = 50,
    #         )

    #         plt.savefig(f"{output_dir}/consolidated_overlap_heatmap_{i}.png")

    

    # input_examples_file = "results/val_input_examples.pkl"
    # print(f"Loading input examples from {input_examples_file}")
    # with open(input_examples_file, 'rb') as f:
    #     input_examples = pickle.load(f)

    # # layer_0_target_neurons = [148, 164, 191]
    # # layer_0_target_neurons = [10, 11, 12, 14, 16, 24]
    # # layer_0_target_neurons = [84, 88, 97, 115, 128, 196, 209, 220]
    # target_neurons = [84, 88, 97, 115, 128, 196, 209, 220]
    # # # for set_type_filter in [0, 1]:

    # for curr_layer in range(4):
    #     output_dir = f"{PATH_PREFIX}/data/mlp_fixed/layer{curr_layer}"
    #     # # Make sure the output directory exists
    #     os.makedirs(output_dir, exist_ok=True)

    #     pkl_filename = os.path.join(
    #         output_dir, f"neuron_activations_layer{curr_layer}.pkl")
    #     with open(pkl_filename, 'rb') as f:
    #         neuron_activations = pickle.load(f)

    #     for set_type_filter in [0, 1, 2]:
    #         # for neuron in [2, 4, 12, 19, 34, 36, 37, 43, 44, 54, 60, 61]:
    #         # for neuron in layer_1_target_neurons:
    #         for neuron in target_neurons:
    #             for pos_idx in range(8):
    #                 print(
    #                     f"Layer {curr_layer}, Set type filter: {set_type_filter}, Neuron: {neuron}, Pos: {pos_idx}")

    #                 if set_type_filter is None:
    #                     set_type_filter_name = "all"
    #                 else:
    #                     set_type_filter_name = set_type_filter
    #                 peaks_dir = f"{PATH_PREFIX}/data/mlp_fixed/layer{curr_layer}/neuron{neuron}/pos{pos_idx}/set_type_{set_type_filter_name}"

    #                 peaks_info_path = os.path.join(peaks_dir, f"{set_type_filter_name}_info.pkl")

    #                 if not os.path.exists(peaks_dir):

    #                     peaks_info = get_peak_info(
    #                         neuron_activations=neuron_activations,
    #                         layer=curr_layer,
    #                         position=pos_idx,
    #                         neuron=neuron,
    #                         input_examples=input_examples,
    #                         min_peak_height=min_peak_height,
    #                         min_peak_distance=min_peak_distance,
    #                         prominence=prominence,
    #                         num_bins=num_bins,
    #                         set_type_filter=set_type_filter,
    #                     )

    #                     os.makedirs(peaks_dir, exist_ok=True)
    #                     with open(peaks_info_path, 'wb') as f:
    #                         pickle.dump(peaks_info, f)
    #                     print(f"Peaks info saved to {peaks_info_path}")
    #                 else:
    #                     # peaks_info = load_peaks_info(layer, neuron=neuron, set_type_filter=set_type_filter)
    #                     with open(peaks_info_path, 'rb') as f:
    #                         peaks_info = pickle.load(f)

    #                 output_dir = f"results/mlp_fixed/peaks/layer{curr_layer}/neuron{neuron}/pos{pos_idx}/set_type_{set_type_filter_name}"
    #                 os.makedirs(output_dir, exist_ok=True)

    #                 top = 100
    #                 save_summary_statistics_from_peak_info(
    #                     peaks_info,
    #                     top=top,
    #                     output_file=os.path.join(
    #                         output_dir, f"top_{top}_summary_statistics.txt")
    #                 )

    #                 save_peak_figure(
    #                     peaks_info,
    #                     filename=os.path.join(output_dir, "histogram_peaks.png"),
    #                 )

    #                 save_top_peak_examples_as_txt(
    #                     config=GPTConfig44_Complete,
    #                     peaks_info=peaks_info,
    #                     filename=os.path.join(output_dir, "peak_examples.txt"),
    #                     top=top)

    # print(f"Saved neuron activations to {pkl_filename}")

    # curr_layer = 0
    # output_dir = f"{PATH_PREFIX}/data/mlp/layer{curr_layer}"
    # pkl_filename = os.path.join(
    #     output_dir, f"neuron_activations_layer{curr_layer}.pkl")
    # with open(pkl_filename, 'rb') as f:
    #     neuron_activations = pickle.load(f)

    # # for neuron_idx in range(256):
    # for neuron_idx in [21, 71, 88, 97, 111, 118, 130, 148, 161, 164, 191, 214]:
    #     for pos_idx in range(8):
    #         at_threshold, num_peaks = check_peak_threshold(
    #             neuron_activations, neuron_idx, pos_idx)
    #         if at_threshold:
    #             print(
    #                 f"Neuron {neuron_idx}, position {pos_idx}, num_peaks {num_peaks}")
    #         # if check_peak_threshold(neuron_activations, neuron_idx, pos_idx):
    #         #     print(f"Neuron {neuron_idx}, position {pos_idx} has enough peaks")
    #         fig = plot_neuron_activations(
    #             neuron_activations, neuron_idx, pos_idx)
    #         peaks_dir = f"results/mlp_fixed/peaks/layer{curr_layer}/neuron{neuron_idx}"
    #         os.makedirs(peaks_dir, exist_ok=True)
    #         fig.savefig(f"{peaks_dir}/pos{pos_idx}_hist.png")
    # # Plot the histogram of activations for a specific neuron
    # neuron_idx = 0
    # # pos_idx = 0
    # for pos_idx in range(8):
    #     print(f"Plotting histogram for neuron {neuron_idx}, position {pos_idx}")
    #     fig = plot_neuron_activations(neuron_activations, neuron_idx, pos_idx)
    #     peaks_dir = f"results/mlp_fixed/peaks/layer{curr_layer}/neuron{neuron_idx}"
    #     os.makedirs(peaks_dir, exist_ok=True)
    #     fig.savefig(f"{peaks_dir}/pos{pos_idx}_hist.png")
