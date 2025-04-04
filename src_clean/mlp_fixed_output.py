from mlp_fixed import analyze_mlp_neurons
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
# from mlp_fixed import plot_overlap_histograms
from mlp_fixed import get_peak_info, test_overall_summary_stat


device = "cuda" if torch.cuda.is_available() else "cpu"
PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

min_peak_height = 0.04
min_peak_distance = 0.03
prominence = 0.03
num_bins = 50

def plot_overlap_histograms(neuron_activations, target_neurons, num_pos=8, num_bins=50, figsize=None):
    # Define set filtering types and their colors
    set_types = {
        0: {"name": "No Set", "color": "blue", "alpha": 0.5},
        1: {"name": "One Set", "color": "green", "alpha": 0.5},
        2: {"name": "Two Set", "color": "red", "alpha": 0.5}
    }

    # Calculate rows and columns for subplot grid
    n_neurons = len(target_neurons)
    n_cols = num_pos  # Each column represents a position
    n_rows = n_neurons  # Each row represents a neuron

    # Set default figsize based on number of neurons if not provided
    if figsize is None:
        figsize = (120, 4 * n_neurons)

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle the case of a single neuron (1D array)
    if n_neurons == 1:
        axes = axes.reshape(1, -1)

    # Plot histogram for each neuron and position
    for i, neuron in enumerate(target_neurons):
        for pos_idx in range(num_pos):
            print(f"Plotting neuron {neuron} at position {pos_idx}...")
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
    print(
        f"Done! Plotted histograms for {len(target_neurons)} neurons with overlapping set types.")

    return fig


def analyze_mlp_weights(mlp_weights):
    for layer_name, layer_weights in mlp_weights.items():
        for weight_name, weights in layer_weights.items():
            col_norms = np.linalg.norm(weights, axis=0)
            row_norms = np.linalg.norm(weights, axis=1)
            if layer_name == "layer_3":
                breakpoint()


if __name__ == "__main__":

    # mlp_weights = torch.load(f"{PATH_PREFIX}/mlp_triples_card_randomization_tuple_randomization_layers_4_heads_4.pt")
    # analyze_mlp_weights(mlp_weights)

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

    # neuron_type = "output"
    # indices = [5, 13, 20, 36, 60]

    # # neuron_type = "hidden"
    # # indices = sorted([185, 25, 93, 36, 166, 89])

    # for curr_layer in [3]:
    #     print(f"Analyzing layer {curr_layer}")
    #     neuron_activations = analyze_mlp_neurons(
    #         model=model,
    #         data_loader=val_loader,
    #         layer_idx=curr_layer,
    #         neuron_indices=indices,
    #         mode='hidden',)
    #     # position_slice=slice(-8,None))

    #     output_dir = f"{PATH_PREFIX}/data/mlp_fixed/{neuron_type}/layer{curr_layer}"
    #     # # Make sure the output directory exists
    #     os.makedirs(output_dir, exist_ok=True)

    #     pkl_filename = os.path.join(
    #         output_dir, f"neuron_activations_layer{curr_layer}_all_positions.pkl")

    #     # Save the activations to a pickle file
    #     with open(pkl_filename, 'wb') as f:
    #         pickle.dump(neuron_activations, f)

    #     print("Plotting histograms for interesting hidden neurons")
    #     output_dir = f"results/mlp_fixed/peaks/{neuron_type}/layer{curr_layer}"
    #     os.makedirs(output_dir, exist_ok=True)

    #     fig = plot_overlap_histograms(
    #         neuron_activations=neuron_activations,
    #         target_neurons=indices,
    #         num_pos=49,
    #         num_bins=50,
    #     )

    #     plt.savefig(
    #         f"{output_dir}/{neuron_type}_consolidated_overlap_heatmap_interesting_from_heatmap.png")

    # PIPELINE FOR ANALYZING ALL SUMMARY STATISTICS
    input_examples_file = "results/val_input_examples.pkl"
    print(f"Loading input examples from {input_examples_file}")
    with open(input_examples_file, 'rb') as f:
        input_examples = pickle.load(f)

    neuron_type = "hidden"
    target_neurons = [36, 89]
    for curr_layer in [3]:
        output_dir = f"{PATH_PREFIX}/data/mlp_fixed/{neuron_type}/layer{curr_layer}"
        # # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        pkl_filename = os.path.join(
            output_dir, f"neuron_activations_layer{curr_layer}_all_positions.pkl")
        with open(pkl_filename, 'rb') as f:
            neuron_activations = pickle.load(f)

        for set_type_filter in [0, 1, 2]:
            # for neuron in [2, 4, 12, 19, 34, 36, 37, 43, 44, 54, 60, 61]:
            # for neuron in layer_1_target_neurons:
            for neuron in target_neurons:
                for pos_idx in range(49):
                    print(
                        f"Layer {curr_layer}, Set type filter: {set_type_filter}, Neuron: {neuron}, Pos: {pos_idx}")

                    if set_type_filter is None:
                        set_type_filter_name = "all"
                    else:
                        set_type_filter_name = set_type_filter
                    peaks_dir = f"{PATH_PREFIX}/data/mlp_fixed/{neuron_type}/layer{curr_layer}/neuron{neuron}/pos{pos_idx}/set_type_{set_type_filter_name}"

                    peaks_info_path = os.path.join(
                        peaks_dir, f"{set_type_filter_name}_info.pkl")

                    if not os.path.exists(peaks_dir):

                        peaks_info = get_peak_info(
                            neuron_activations=neuron_activations,
                            layer=curr_layer,
                            position=pos_idx,
                            neuron=neuron,
                            input_examples=input_examples,
                            min_peak_height=min_peak_height,
                            min_peak_distance=min_peak_distance,
                            prominence=prominence,
                            num_bins=num_bins,
                            set_type_filter=set_type_filter,
                        )

                        os.makedirs(peaks_dir, exist_ok=True)
                        with open(peaks_info_path, 'wb') as f:
                            pickle.dump(peaks_info, f)
                        print(f"Peaks info saved to {peaks_info_path}")
                    else:
                        # peaks_info = load_peaks_info(layer, neuron=neuron, set_type_filter=set_type_filter)
                        with open(peaks_info_path, 'rb') as f:
                            peaks_info = pickle.load(f)

                    output_dir = f"results/mlp_fixed/peaks/{neuron_type}/layer{curr_layer}/neuron{neuron}/pos{pos_idx}/set_type_{set_type_filter_name}"
                    os.makedirs(output_dir, exist_ok=True)

                    top = 100
                    save_summary_statistics_from_peak_info(
                        peaks_info,
                        top=top,
                        output_file=os.path.join(
                            output_dir, f"top_{top}_summary_statistics.txt")
                    )

                    save_peak_figure(
                        peaks_info,
                        filename=os.path.join(
                            output_dir, "histogram_peaks.png"),
                    )

                    save_top_peak_examples_as_txt(
                        config=GPTConfig44_Complete,
                        peaks_info=peaks_info,
                        filename=os.path.join(output_dir, "peak_examples.txt"),
                        top=top)
                    
                    test_overall_summary_stat(
                        peaks_info,
                        top=top,
                        output_file=os.path.join(output_dir, "overall_summary_statistics.txt")
                    )
