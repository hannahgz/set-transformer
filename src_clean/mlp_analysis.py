from model import GPTConfig44_Complete, GPT
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

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
            'c_fc': block.mlp.c_fc.weight.detach().cpu().numpy(),  # Expansion weights
            'c_proj': block.mlp.c_proj.weight.detach().cpu().numpy()  # Projection weights
        }
        if block.mlp.c_fc.bias is not None:
            layer_weights['c_fc_bias'] = block.mlp.c_fc.bias.detach().cpu().numpy()
        if block.mlp.c_proj.bias is not None:
            layer_weights['c_proj_bias'] = block.mlp.c_proj.bias.detach().cpu().numpy()
        
        mlp_weights[f'layer_{i}'] = layer_weights
    
    return model, mlp_weights

if __name__ == "__main__":
    config = GPTConfig44_Complete()
    print("Extracting mlp weights")
    model, mlp_weights = load_model_and_extract_mlp_weights(config)
    breakpoint()
