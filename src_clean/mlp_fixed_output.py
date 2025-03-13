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
from mlp_fixed import plot_overlap_histograms


device = "cuda" if torch.cuda.is_available() else "cpu"
PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

def analyze_mlp_weights(mlp_weights):
    for layer_name, layer_weights in mlp_weights.items():
        for weight_name, weights in layer_weights.items():
            col_norms = np.linalg.norm(weights, axis=0)
            breakpoint()

if __name__ == "__main__":

    mlp_weights = torch.load(f"{PATH_PREFIX}/mlp_triples_card_randomization_tuple_randomization_layers_4_heads_4.pt")
    analyze_mlp_weights(mlp_weights)

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

    # neuron_indices = [5, 13, 20, 36, 60]
    # for curr_layer in [3]:
    #     print(f"Analyzing layer {curr_layer}")
    #     neuron_activations = analyze_mlp_neurons(
    #         model=model,
    #         data_loader=val_loader,
    #         layer_idx=curr_layer,
    #         neuron_indices=neuron_indices,
    #         mode='output',
    #         position_slice=slice(-8,None))

    #     output_dir = f"{PATH_PREFIX}/data/mlp_fixed/output/layer{curr_layer}"
    #     # # Make sure the output directory exists
    #     os.makedirs(output_dir, exist_ok=True)

    #     pkl_filename = os.path.join(
    #         output_dir, f"neuron_activations_layer{curr_layer}.pkl")

    #     # Save the activations to a pickle file
    #     with open(pkl_filename, 'wb') as f:
    #         pickle.dump(neuron_activations, f)

    #     print("Plotting histograms for interesting output neurons")
    #     output_dir = f"results/mlp_fixed/peaks/output/layer{curr_layer}"
    #     os.makedirs(output_dir, exist_ok=True)
    
    #     fig = plot_overlap_histograms(
    #         neuron_activations = neuron_activations,
    #         target_neurons = neuron_indices,
    #         num_bins = 50,
    #     )

    #     plt.savefig(f"{output_dir}/consolidated_overlap_heatmap_interesting_from_heatmap.png")
