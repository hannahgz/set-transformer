import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks
import os
from collections import defaultdict

def find_activation_peaks(layer, neuron, input_examples_file=None, min_peak_height=0.05, 
                         min_peak_distance=0.1, prominence=0.02, num_bins=50):
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
    
    Returns:
        peaks_info: Information about the peaks, including indices, heights, and examples
    """
    # Load activation data from the pickle file
    pkl_filename = f"neuron_activations_layer{layer}.pkl"
    if not os.path.exists(pkl_filename):
        raise FileNotFoundError(f"Activation data file {pkl_filename} not found")
    
    print(f"Loading activation data from {pkl_filename}")
    with open(pkl_filename, 'rb') as f:
        neuron_activations = pickle.load(f)
    
    # Check if the neuron exists in the data
    if neuron not in neuron_activations:
        raise ValueError(f"Neuron {neuron} not found in layer {layer} data")
    
    # Combine activations from all set types for the specified neuron
    all_activations = []
    activation_by_type = {}
    for set_type in neuron_activations[neuron]:
        activations = neuron_activations[neuron][set_type]
        all_activations.extend(activations)
        activation_by_type[set_type] = activations
    
    print(f"Analyzing {len(all_activations)} total activations for neuron {neuron} in layer {layer}")
    
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
    peaks, peak_properties = find_peaks(hist, height=height, distance=distance, prominence=prom)
    
    print(f"Found {len(peaks)} peaks in the histogram")
    
    # Create a dictionary to store peak information (without figure yet)
    peaks_info = {
        'neuron': neuron,
        'layer': layer,
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
    
    plt.title(f'Neuron {neuron} in Layer {layer} - Activation Histogram with {len(peaks)} Peaks')
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
        for set_type in activation_by_type:
            if set_type in input_examples:
                for i, act_value in enumerate(activation_by_type[set_type]):
                    if i < len(input_examples[set_type]):
                        all_activations_with_examples.append((act_value, input_examples[set_type][i]))
        
        # Assign each activation to its nearest peak
        peak_values = bin_centers[peaks]
        for act_value, example in all_activations_with_examples:
            # Find the closest peak
            closest_peak_idx = np.abs(peak_values - act_value).argmin()
            # Store the example with its exact activation value
            peaks_info['examples_by_peak'][closest_peak_idx].append((act_value, example))
        
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
            closest_indices = np.argsort(np.abs(np.array(all_activations) - peak_val))[:5]
            closest_activations = [all_activations[idx] for idx in closest_indices]
            peaks_info['examples_by_peak'][i] = [(act, f"Activation: {act:.4f}") for act in closest_activations]
    
    # Print information about each peak
    for i, peak in enumerate(peaks):
        peak_value = bin_centers[peak]
        peak_height = hist[peak]
        prominence = peak_properties['prominences'][i]
        
        print(f"\nPeak {i+1}:")
        print(f"  Value: {peak_value:.4f}")
        print(f"  Height: {peak_height} ({peak_height/hist.max():.2%} of max)")
        print(f"  Prominence: {prominence} ({prominence/hist.max():.2%} of max)")
        
        if i in peaks_info['examples_by_peak']:
            print("  Example activations near this peak:")
            for j, (act_value, example) in enumerate(peaks_info['examples_by_peak'][i][:3]):
                print(f"    Example {j+1}: {act_value:.4f} - {example[:50]}..." if len(str(example)) > 50 
                      else f"    Example {j+1}: {act_value:.4f} - {example}")
    
    # Now add the completed figure to the dictionary
    peaks_info['figure'] = fig
    
    plt.show()
    return peaks_info
    
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
        raise ValueError("No figure found in peaks_info. Make sure you're using the updated find_activation_peaks function.")
    
    # Generate filename if not provided
    if filename is None:
        neuron = peaks_info['neuron']
        layer = peaks_info['layer']
        filename = f"COMPLETE_FIGS/mlp/layer_{layer}/neuron_{neuron}_peaks.{format}"
    
    # Save the figure
    fig = peaks_info['figure']
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {filename}")
    
    return filename

# def create_input_sequence_dataset(model, data_loader, target_neurons, config, capture_layer):
#     """
#     Create a dataset that maps input sequences to their activation values.
#     This allows us to find examples for each activation peak.
    
#     This function should be integrated with your data collection process.
#     """
#     import torch
#     import pickle
#     import os
    
#     model.eval()
    
#     # Define set types
#     set_types = [0, 1, 2]  # No Set, One Set, Two Set
    
#     # Initialize tracking structures for examples by set type
#     input_examples = {set_type: [] for set_type in set_types}
    
#     tokenizer = load_tokenizer(config.tokenizer_path)
#     no_set_token = tokenizer.token_to_id["*"]
#     two_set_token = tokenizer.token_to_id["/"]

#     # Process each batch in the data loader
#     print(f"Collecting input examples...")
#     for batch_idx, batch in enumerate(data_loader):
#         if batch_idx % 10 == 0:
#             print(f"Processing batch {batch_idx}/{len(data_loader)}")
        
#         batch = batch.to(device)
#         batch_size = batch.shape[0]
        
#         # For each sequence in the batch
#         for seq_idx in range(batch_size):
#             sequence = batch[seq_idx]
            
#             # Determine which set type this sequence belongs to
#             current_set_type = None
#             if no_set_token in sequence:
#                 current_set_type = 0  # No Set
#             elif two_set_token in sequence:
#                 current_set_type = 2  # Two Set
#             else:
#                 current_set_type = 1  # One Set
            
#             # Record the sequence itself as the example (would need to decode to text)
#             sequence_tokens = sequence.cpu().tolist()
#             decoded_sequence = tokenizer.decode(sequence_tokens)
#             input_examples[current_set_type].append(decoded_sequence)
    
#     # Save the input examples to a pickle file
#     examples_filename = f"input_examples_layer{capture_layer}.pkl"
#     with open(examples_filename, 'wb') as f:
#         pickle.dump(input_examples, f)
    
#     print(f"Saved input examples to {examples_filename}")
    
#     return input_examples

# Example usage:
if __name__ == "__main__":
    # Get user input for layer and neuron
    # layer = int(input("Enter the layer number: "))
    # neuron = int(input("Enter the neuron index: "))

    layer = 0

    for neuron in [12, 14, 36, 43, 44, 60, 61]:
    # neuron = 4
        # Check if there's an input examples file
        examples_file = f"input_examples_layer{layer}.pkl"
        if os.path.exists(examples_file):
            use_examples = input(f"Found input examples file {examples_file}. Use it? (y/n): ").lower() == 'y'
        else:
            print(f"No input examples file found. Will only show activation values.")
            use_examples = False
        
        # Get parameters for peak detection
        min_peak_height = 0.04
        min_peak_distance = 0.05
        prominence = 0.01
        num_bins = 50
        
        # Find peaks for the specified neuron
        peaks_info = find_activation_peaks(
            layer=layer,
            neuron=neuron,
            input_examples_file=examples_file if use_examples else None,
            min_peak_height=min_peak_height,
            min_peak_distance=min_peak_distance,
            prominence=prominence
        )

        # Save the figure
        saved_filename = save_peak_figure(
            peaks_info,
        )
