import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks
import os
from collections import defaultdict
import torch
from tokenizer import load_tokenizer
import random
from model import GPTConfig44_Complete
from data_utils import initialize_loaders, pretty_print_input

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'
device = "cuda" if torch.cuda.is_available() else "cpu"


def find_activation_peaks(layer, neuron, input_examples_file=None, min_peak_height=0.05,
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
    from scipy.signal import find_peaks
    import matplotlib.pyplot as plt
    from collections import defaultdict

    pkl_filename = f"neuron_activations_layer{layer}.pkl"
    if not os.path.exists(pkl_filename):
        raise FileNotFoundError(
            f"Activation data file {pkl_filename} not found")

    print(f"Loading activation data from {pkl_filename}")
    with open(pkl_filename, 'rb') as f:
        neuron_activations = pickle.load(f)

    # Check if the neuron exists in the data
    if neuron not in neuron_activations:
        raise ValueError(f"Neuron {neuron} not found in layer {layer} data")

    # Combine activations from all set types or filter by specified set type
    all_activations = []
    activation_by_type = {}

    if set_type_filter is not None:
        # Verify that the requested set type exists
        if set_type_filter not in neuron_activations[neuron]:
            available_sets = list(neuron_activations[neuron].keys())
            raise ValueError(
                f"Set type '{set_type_filter}' not found. Available set types: {available_sets}")

        # Only use the specified set type
        activations = neuron_activations[neuron][set_type_filter]
        all_activations.extend(activations)
        activation_by_type[set_type_filter] = activations
        print(
            f"Analyzing {len(all_activations)} activations for neuron {neuron} in layer {layer} (set type: {set_type_filter})")
    else:
        # Use all available set types
        for set_type in neuron_activations[neuron]:
            activations = neuron_activations[neuron][set_type]
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
    prom = prominence * hist.max()  # Minimum prominence

    # Find peaks in the histogram
    peaks, peak_properties = find_peaks(
        hist, height=height, distance=distance, prominence=prom)

    print(f"Found {len(peaks)} peaks in the histogram")

    # Create a dictionary to store peak information (without figure yet)
    peaks_info = {
        'neuron': neuron,
        'layer': layer,
        'set_type': set_type_filter if set_type_filter is not None else 'all',
        'bin_centers': bin_centers[peaks],
        'heights': hist[peaks],
        'prominences': peak_properties['prominences'],
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
        f'Neuron {neuron} in Layer {layer}{title_suffix} - Activation Histogram with {len(peaks)} Peaks')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)

    # Load input examples if a file is provided
    if input_examples_file and os.path.exists(input_examples_file):
        print(f"Loading input examples from {input_examples_file}")
        with open(input_examples_file, 'rb') as f:
            input_examples = pickle.load(f)

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
            peaks_info['examples_by_peak'][peak_idx] = peaks_info['examples_by_peak'][peak_idx][:5]

    else:
        print("No input examples file provided or file not found")

        # For each peak, find the activation values closest to it
        peak_values = bin_centers[peaks]
        for i, peak_val in enumerate(peak_values):
            # Find indices of activations closest to this peak
            closest_indices = np.argsort(
                np.abs(np.array(all_activations) - peak_val))[:5]
            closest_activations = [all_activations[idx]
                                   for idx in closest_indices]
            peaks_info['examples_by_peak'][i] = [
                (act, f"Activation: {act:.4f}") for act in closest_activations]

    # Print information about each peak
    for i, peak in enumerate(peaks):
        peak_value = bin_centers[peak]
        peak_height = hist[peak]
        prominence = peak_properties['prominences'][i]

        print(f"\nPeak {i+1}:")
        print(f"  Value: {peak_value:.4f}")
        print(f"  Height: {peak_height} ({peak_height/hist.max():.2%} of max)")
        print(
            f"  Prominence: {prominence} ({prominence/hist.max():.2%} of max)")

        if i in peaks_info['examples_by_peak']:
            print("  Example activations near this peak:")
            for j, (act_value, example) in enumerate(peaks_info['examples_by_peak'][i][:3]):
                print(f"    Example {j+1}: {act_value:.4f} - {example[:50]}..." if len(str(example)) > 50
                      else f"    Example {j+1}: {act_value:.4f} - {example}")

    # Now add the completed figure to the dictionary
    peaks_info['figure'] = fig

    plt.show()
    return peaks_info


def get_peak_info(layer, neuron, input_examples_file=None, min_peak_height=0.05,
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
    from scipy.signal import find_peaks
    import matplotlib.pyplot as plt
    from collections import defaultdict

    pkl_filename = f"neuron_activations_layer{layer}.pkl"
    if not os.path.exists(pkl_filename):
        raise FileNotFoundError(
            f"Activation data file {pkl_filename} not found")

    print(f"Loading activation data from {pkl_filename}")
    with open(pkl_filename, 'rb') as f:
        neuron_activations = pickle.load(f)

    # Check if the neuron exists in the data
    if neuron not in neuron_activations:
        raise ValueError(f"Neuron {neuron} not found in layer {layer} data")

    # Combine activations from all set types or filter by specified set type
    all_activations = []
    activation_by_type = {}

    if set_type_filter is not None:
        # Verify that the requested set type exists
        if set_type_filter not in neuron_activations[neuron]:
            available_sets = list(neuron_activations[neuron].keys())
            raise ValueError(
                f"Set type '{set_type_filter}' not found. Available set types: {available_sets}")

        # Only use the specified set type
        activations = neuron_activations[neuron][set_type_filter]
        all_activations.extend(activations)
        activation_by_type[set_type_filter] = activations
        print(
            f"Analyzing {len(all_activations)} activations for neuron {neuron} in layer {layer} (set type: {set_type_filter})")
    else:
        # Use all available set types
        for set_type in neuron_activations[neuron]:
            activations = neuron_activations[neuron][set_type]
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
    prom = prominence * hist.max()  # Minimum prominence

    # Find peaks in the histogram
    peaks, peak_properties = find_peaks(
        hist, height=height, distance=distance, prominence=prom)

    print(f"Found {len(peaks)} peaks in the histogram")

    # Create a dictionary to store peak information (without figure yet)
    peaks_info = {
        'neuron': neuron,
        'layer': layer,
        'set_type': set_type_filter if set_type_filter is not None else 'all',
        'bin_centers': bin_centers[peaks],
        'heights': hist[peaks],
        'prominences': peak_properties['prominences'],
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
        f'Neuron {neuron} in Layer {layer}{title_suffix} - Activation Histogram with {len(peaks)} Peaks')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)

    # Load input examples if a file is provided
    if input_examples_file and os.path.exists(input_examples_file):
        print(f"Loading input examples from {input_examples_file}")
        with open(input_examples_file, 'rb') as f:
            input_examples = pickle.load(f)

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

    # # Print information about each peak
    # for i, peak in enumerate(peaks):
    #     peak_value = bin_centers[peak]
    #     peak_height = hist[peak]
    #     prominence = peak_properties['prominences'][i]

    #     print(f"\nPeak {i+1}:")
    #     print(f"  Value: {peak_value:.4f}")
    #     print(f"  Height: {peak_height} ({peak_height/hist.max():.2%} of max)")
    #     print(
    #         f"  Prominence: {prominence} ({prominence/hist.max():.2%} of max)")

    #     if i in peaks_info['examples_by_peak']:
    #         print("  Example activations near this peak:")
    #         for j, (act_value, example) in enumerate(peaks_info['examples_by_peak'][i][:3]):
    #             print(f"    Example {j+1}: {act_value:.4f} - {example[:50]}..." if len(str(example)) > 50
    #                   else f"    Example {j+1}: {act_value:.4f} - {example}")

    # Now add the completed figure to the dictionary
    peaks_info['figure'] = fig

    plt.show()
    return peaks_info

tokenizer = load_tokenizer(f"{PATH_PREFIX}/all_tokenizer.pkl")

shapes = ["oval", "squiggle", "diamond"]
colors = ["green", "blue", "pink"]
numbers = ["one", "two", "three"]
shadings = ["solid", "striped", "open"]

shape_ids = tokenizer.encode(shapes)
color_ids = tokenizer.encode(colors)
number_ids = tokenizer.encode(numbers)
shading_ids = tokenizer.encode(shadings)

all_ids = shape_ids + color_ids + number_ids + shading_ids

def initialize_attribute_count_dict():
    attribute_count_dict = {}

    for id in all_ids:
        attribute_count_dict[id] = 0

    return attribute_count_dict

from itertools import combinations

def count_combinations(numbers):
    """
    Counts how many 3-number combinations from a list of 5 numbers
    have all numbers either all the same or all different.
    """
    same_count = 0
    diff_count = 0

    # Iterate through all 5 choose 3 combinations
    for comb in combinations(numbers, 3):
        # Check if the combination is all the same
        if len(set(comb)) == 1:
            same_count += 1
        # Check if the combination is all different
        elif len(set(comb)) == 3:
            diff_count += 1

    return same_count, diff_count

def summary_statistics_from_peak_info(peaks_info, top=None):
    peaks_attribute_dict = {}
    peaks_same_diff_dict = {
        "shape": {"same": 0, "diff": 0, "total": 0},
        "color": {"same": 0, "diff": 0, "total": 0},
        "number": {"same": 0, "diff": 0, "total": 0},
        "shading": {"same": 0, "diff": 0, "total": 0}
    }
    total_num_attributes = 0
    for peak_idx in peaks_info['examples_by_peak']:
        peaks_attribute_dict[peak_idx] = initialize_attribute_count_dict()

        if top is None:
            top = len(peaks_info['examples_by_peak'][peak_idx])
        for example in peaks_info['examples_by_peak'][peak_idx][:top]:
            i = 0
            attrs_dict_by_category = {
                "shape": [],
                "color": [],
                "number": [],
                "shading": []
            }
            # shape_attrs = []
            # color_attrs = []
            # number_attrs = []
            # shading_attrs = []
            while i < 40:
                card = example[i]
                attr = example[i+1]

                peaks_attribute_dict[peak_idx][attr] += 1
                total_num_attributes += 1

                if attr in shape_ids:
                    attrs_dict_by_category["shape"].append(attr)
                elif attr in color_ids:
                    attrs_dict_by_category["color"].append(attr)
                elif attr in number_ids:
                    attrs_dict_by_category["number"].append(attr)
                elif attr in shading_ids:
                    attrs_dict_by_category["shading"].append(attr)
            
            for attribute_type in attrs_dict_by_category:
                same_count, diff_count = count_combinations(attrs_dict_by_category[attribute_type])
                peaks_same_diff_dict[attribute_type]["same"] += same_count
                peaks_same_diff_dict[attribute_type]["diff"] += diff_count
                peaks_same_diff_dict[attribute_type]["total"] += 10

    for peak_idx in peaks_attribute_dict:
        print(f"\nPeak {peak_idx+1}:")

        print("Attribute Breakdown")
        for id in all_ids:
            print(f"    Percentage of {tokenizer.decode([id])}: {peaks_attribute_dict[peak_idx][id]/total_num_attributes:.2%}")

        print(f"Total Same and Different")
        for attribute_type in peaks_same_diff_dict:
            same_pct = peaks_same_diff_dict[attribute_type]["same"]/peaks_same_diff_dict[attribute_type]["total"]
            diff_pct = peaks_same_diff_dict[attribute_type]["diff"]/peaks_same_diff_dict[attribute_type]["total"]
            print(f"    {attribute_type}: Same - {same_pct:.2%}, Different - {diff_pct:.2%}")


def save_peak_figure(peaks_info, filename=None, dpi=300, format='png'):
    """
    Save the figure from the peaks_info dictionary to a file.

    Parameters:
        peaks_info: Dictionary returned by find_activation_peaks
        filename: Output filename (if None, will generate based on neuron and layer)
        dpi: Resolution for the saved image
        format: File format (png, jpg, svg, pdf, etc.)

    Returns:
        The filename where the figure was saved
    """
    if 'figure' not in peaks_info:
        raise ValueError(
            "No figure found in peaks_info. Make sure you're using the updated find_activation_peaks function.")

    # Generate filename if not provided
    if filename is None:
        neuron = peaks_info['neuron']
        layer = peaks_info['layer']
        set_type_filter = peaks_info['set_type']
        # filename = f"COMPLETE_FIGS/mlp/layer_{layer}/neuron_{neuron}_set_type_{set_type_filter}_peaks.{format}"
        filename = f"COMPLETE_FIGS/mlp/layer_{layer}/neuron_{neuron}_set_type_{set_type_filter}_peaks.{format}"

    # Save the figure
    fig = peaks_info['figure']
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {filename}")

    return filename


def create_input_sequence_dataset(data_loader, config):
    """
    Create a dataset that maps input sequences to their activation values.
    This allows us to find examples for each activation peak.

    This function should be integrated with your data collection process.
    """

    # Define set types
    set_types = [0, 1, 2]  # No Set, One Set, Two Set

    # Initialize tracking structures for examples by set type
    input_examples = {set_type: [] for set_type in set_types}

    tokenizer = load_tokenizer(config.tokenizer_path)
    no_set_token = tokenizer.token_to_id["*"]
    two_set_token = tokenizer.token_to_id["/"]

    # Process each batch in the data loader
    print(f"Collecting input examples...")
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}/{len(data_loader)}")

        batch = batch.to(device)
        batch_size = batch.shape[0]

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

            # Record the sequence itself as the example (would need to decode to text)
            sequence_tokens = sequence.cpu().tolist()
            # decoded_sequence = tokenizer.decode(sequence_tokens)
            # input_examples[current_set_type].append(decoded_sequence)
            input_examples[current_set_type].append(sequence_tokens)

    # Save the input examples to a pickle file
    examples_filename = f"results/val_input_examples.pkl"
    with open(examples_filename, 'wb') as f:
        pickle.dump(input_examples, f)

    print(f"Saved input examples to {examples_filename}")

    return input_examples


def load_peaks_info(layer, neuron, config, set_type_filter=None):
    """
    Load peaks information from a pickle file.

    Parameters:
        layer: Layer number
        neuron: Neuron index
        set_type_filter: Set type to filter by (if None, all set types)

    Returns:
        peaks_info: Dictionary with peak information
    """

    tokenizer = load_tokenizer(config.tokenizer_path)
    filename = f"results/peaks_info_layer{layer}_neuron{neuron}_set_type_{set_type_filter}.pkl"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Peaks info file {filename} not found")

    with open(filename, 'rb') as f:
        peaks_info = pickle.load(f)

    # Print tokenized versions of all the saved examples
    for peak_idx in peaks_info['examples_by_peak']:
        print(f"\nPeak {peak_idx+1}:")        
        for example in peaks_info['examples_by_peak'][peak_idx][0:3]:
            # Decode the tokenized example
            decoded_example = tokenizer.decode(example[1])

            print(f"  Activation: {example[0]:.4f}")
            print(f"  {decoded_example}")
            print(f"  {pretty_print_input(decoded_example)}")
    return peaks_info

# def save_top_peak_examples_as_txt(config, peaks_info, filename, top=2):
#     tokenizer = load_tokenizer(config.tokenizer_path)

#     with open(filename, 'w') as f:
#         for peak_idx in peaks_info['examples_by_peak']:
#             f.write(f"\nPeak {peak_idx+1}:\n")
#             for example in peaks_info['examples_by_peak'][peak_idx][0:top]:
#                 decoded_example = tokenizer.decode(example[1])
#                 f.write(f"  Activation: {example[0]:.4f}\n")
#                 f.write(f"  {decoded_example}\n")
#                 f.write(f"  {pretty_print_input(decoded_example)}\n")

def save_top_peak_examples_as_txt(config, peaks_info, filename, top=2):
    tokenizer = load_tokenizer(config.tokenizer_path)

    with open(filename, 'w') as f:
        # Get the peak indices and sort them
        peak_indices = sorted(peaks_info['examples_by_peak'].keys())
        
        for peak_idx in peak_indices:
            f.write(f"\nPeak {peak_idx+1}:\n")
            for example in peaks_info['examples_by_peak'][peak_idx][0:top]:
                decoded_example = tokenizer.decode(example[1])
                f.write(f"  Activation: {example[0]:.4f}\n")
                f.write(f"  {decoded_example}\n")
                
                # Handle the pretty print table with proper indentation
                pretty_table = pretty_print_input(decoded_example)
                # Add two spaces of indentation to each line in the table
                indented_table = "\n".join("  " + line for line in pretty_table.split("\n"))
                f.write(f"{indented_table}\n")

# Example usage:
if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # layer = 0
    # config = GPTConfig44_Complete()

    # # dataset = torch.load(config.dataset_path)
    # # _, val_loader = initialize_loaders(config, dataset)

    # dataset_path = f"{PATH_PREFIX}/base_card_randomization_tuple_randomization_dataset.pth"
    # dataset = torch.load(dataset_path)
    # _, val_loader = initialize_loaders(config, dataset)

    # # Save the validation data loader to a file for later use
    # val_loader_filename = f"{PATH_PREFIX}/val_loader.pth"
    # torch.save(val_loader, val_loader_filename)
    # print(f"Validation data loader saved to {val_loader_filename}")

    # create_input_sequence_dataset(
    #     data_loader=val_loader,
    #     config=GPTConfig44_Complete(),)

    layer = 0
    # set_type_filter = 0

    # Get parameters for peak detection
    min_peak_height = 0.04
    min_peak_distance = 0.05
    prominence = 0.01
    num_bins = 50
    
    for set_type_filter in [0, 1]:
        for neuron in [36]:
            examples_file = "results/val_input_examples.pkl"

            peaks_info = get_peak_info(
                layer=layer,
                neuron=neuron,
                input_examples_file=examples_file,
                min_peak_height=min_peak_height,
                min_peak_distance=min_peak_distance,
                prominence=prominence,
                num_bins=num_bins,
                set_type_filter=set_type_filter,
            )

            peaks_dir = f"results/mlp/peaks/layer{layer}/neuron{neuron}/set_type_{set_type_filter}"
            os.makedirs(peaks_dir, exist_ok=True)

            peaks_info_path = os.path.join(peaks_dir, "all_info.pkl")
            with open(peaks_info_path, 'wb') as f:
                pickle.dump(peaks_info, f)
            print(f"Peaks info saved to {peaks_info_path}")

            summary_statistics_from_peak_info(peaks_info, top=100)


    # for set_type_filter in [0, 1]:
    #     # for neuron in [12, 14, 36, 43, 44, 60, 61]:
    #     for neuron in [36, 43, 44, 60, 61]:
    #         # neuron = 4
    #         # Check if there's an input examples file
    #         examples_file = "results/val_input_examples.pkl"
    #         use_examples = True

    #         # Find peaks for the specified neuron
    #         peaks_info = find_activation_peaks(
    #             layer=layer,
    #             neuron=neuron,
    #             input_examples_file=examples_file if use_examples else None,
    #             min_peak_height=min_peak_height,
    #             min_peak_distance=min_peak_distance,
    #             prominence=prominence,
    #             set_type_filter=set_type_filter,
    #         )

    #         # Save peaks_info to a pickle file
    #         # peaks_info_filename = f"results/mlp/peaks/peaks_info_layer{layer}_neuron{neuron}_set_type_{set_type_filter}.pkl"
    #         peaks_dir = f"results/mlp/peaks/layer{layer}/neuron{neuron}/set_type_{set_type_filter}"
    #         os.makedirs(peaks_dir, exist_ok=True)

    #         peaks_info_path = os.path.join(peaks_dir, "info.pkl")
    #         with open(peaks_info_path, 'wb') as f:
    #             pickle.dump(peaks_info, f)
    #         print(f"Peaks info saved to {peaks_info_path}")

    #         # Save the figure
    #         save_peak_figure(
    #             peaks_info,
    #             filename=os.path.join(peaks_dir, "histogram_peaks.png"),
    #         )

    #         save_top_peak_examples_as_txt(
    #             config=GPTConfig44_Complete, 
    #             peaks_info=peaks_info, 
    #             filename=os.path.join(peaks_dir, "peak_examples.txt"), 
    #             top=10)



    # load_peaks_info(layer, neuron=61, config=GPTConfig44_Complete(), set_type_filter=set_type_filter)
