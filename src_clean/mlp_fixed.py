from model import GPTConfig44_Complete, GPT
from data_utils import initialize_loaders, pretty_print_input
from tokenizer import load_tokenizer
import torch

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
        hidden_hook_handle = model.transformer.h[layer_idx].mlp.gelu.register_forward_hook(hidden_hook)
        hooks.append(hidden_hook_handle)
    
    if position in ['output', 'both']:
        # Hook on the c_proj layer to capture its output (the 64-dimensional output)
        output_hook_handle = model.transformer.h[layer_idx].mlp.c_proj.register_forward_hook(output_hook)
        hooks.append(output_hook_handle)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_data, get_loss=False)
    
    # Remove all hooks
    for hook in hooks:
        hook.remove()
    
    return activations


def analyze_mlp_neurons(model, data_loader, layer_idx=0, neuron_indices=None, mode='hidden'):
    """
    Analyze specific neurons in the MLP's hidden or output layer.
    
    Parameters:
        model: The GPT model
        data_loader: DataLoader yielding batches of input data
        layer_idx: Which transformer layer to analyze
        neuron_indices: List of neuron indices to analyze (None for all)
        mode: 'hidden' for 256-dim hidden neurons, 'output' for 64-dim output neurons
    
    Returns:
        Dictionary mapping neuron indices to their activation distributions
    """
    model.eval()
    
    # Determine dimensions based on mode
    n_dims = 256 if mode == 'hidden' else 64
    
    # If no specific neurons provided, analyze all
    if neuron_indices is None:
        neuron_indices = list(range(n_dims))
    
    # Initialize dictionary to collect activation values for each neuron
    neuron_activations = {idx: [] for idx in neuron_indices}
    
    # Process each batch
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}/{len(data_loader)}")
        
        batch = batch.to(device)
        
        # Get activations for this batch
        activations_dict = analyze_mlp_activations(model, batch, layer_idx, position='both')
        
        # Extract the activations we're interested in
        act_tensor = activations_dict['hidden'] if mode == 'hidden' else activations_dict['output']
        
        breakpoint()

        # For each neuron of interest, collect its activations
        for neuron_idx in neuron_indices:
            # Get all activations for this neuron (across batch and sequence)
            # We're taking the activations at the final position in the sequence
            neuron_acts = act_tensor[:, -1, neuron_idx].cpu().numpy()
            neuron_activations[neuron_idx].extend(neuron_acts.tolist())
    
    return neuron_activations

def plot_neuron_activations(neuron_activations, neuron_idx, num_bins=50):
    """Plot histogram of activations for a specific neuron"""
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    activations = neuron_activations[neuron_idx]
    
    plt.figure(figsize=(10, 6))
    hist, bin_edges = plt.hist(activations, bins=num_bins, alpha=0.7)
    
    # Find peaks in the histogram
    peaks, _ = find_peaks(hist, height=0.1*hist.max(), distance=num_bins/10)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Mark the peaks
    plt.plot(bin_centers[peaks], hist[peaks], "x", color='red', markersize=10)
    
    plt.title(f'Neuron {neuron_idx} Activation Distribution')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.show()
    
    return bin_centers[peaks], hist[peaks]


if __name__ == "__main__":
    config = GPTConfig44_Complete()
    # Load the checkpoint
    checkpoint = torch.load(config.filename, weights_only = False)
    
    # Create the model architecture
    model = GPT(config).to(device)
    
    # Load the weights
    model.load_state_dict(checkpoint['model'])
    model.eval()  # Set to evaluation mode

    # dataset = torch.load(config.dataset_path)
    # _, val_loader = initialize_loaders(config, dataset)

    dataset_path = f"{PATH_PREFIX}/base_card_randomization_tuple_randomization_dataset.pth"
    dataset = torch.load(dataset_path)
    _, val_loader = initialize_loaders(config, dataset)

    analyze_mlp_neurons(
        model=model, 
        data_loader=val_loader, 
        layer_idx=0, 
        neuron_indices=None, 
        mode='hidden')
