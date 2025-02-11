import torch
import torch.nn.functional as F
import random
import numpy as np
from classify import load_model_from_config
from model import GPTConfig44_Complete
from data_utils import load_tokenizer

def logit_lens_analysis(model, inputs):
    """
    Perform logit lens analysis on the model by projecting intermediate representations
    to vocabulary space using the unembedding matrix (lm_head weights).
    """
    device = inputs.device
    b, t = inputs.size()
    
    # Get position embeddings
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(inputs)
    pos_emb = model.transformer.wpe(pos)
    
    # Initial embedding
    x = tok_emb + pos_emb
    
    # Store intermediate logits
    layer_logits = []
    
    # Project input embeddings to logit space
    initial_logits = model.lm_head(model.transformer.ln_f(x))
    layer_logits.append(("embedding", initial_logits))
    
    # Get intermediate representations and project each to logit space
    for i, block in enumerate(model.transformer.h):
        # Apply attention and MLP
        x_normalized = block.ln_1(x)
        attn_output, _, _ = block.attn(x_normalized)
        x = x + attn_output
        
        # Project post-attention state to logits
        post_attn_logits = model.lm_head(model.transformer.ln_f(x))
        layer_logits.append((f"layer_{i}_attn", post_attn_logits))
        
        # Apply MLP
        x = x + block.mlp(block.ln_2(x))
        
        # Project post-MLP state to logits
        post_mlp_logits = model.lm_head(model.transformer.ln_f(x))
        layer_logits.append((f"layer_{i}_mlp", post_mlp_logits))
    
    # # Get final logits
    # final_logits = model.lm_head(model.transformer.ln_f(x))
    # layer_logits.append(("final", final_logits))
    
    return layer_logits

def analyze_predictions(layer_logits, model_config, tokenizer, tokenized_input_sequence):
    """
    Analyze how predictions evolve through the layers.
    """
    results = []
    
    for layer_name, logits in layer_logits:
        # Get predictions at each layer
        probs = F.softmax(logits, dim=-1)
        # predictions = torch.argmax(logits, dim=-1)[:, -(model_config.target_size+1):-1]
        predictions = torch.argmax(logits, dim=-1)[:, -(model_config.target_size+1):-1]
        targets = tokenized_input_sequence[:, -model_config.target_size:]

        mask = targets != model_config.padding_token 
        total_non_mask_count = mask.sum().item()
        
        matches = ((predictions == targets) | ~mask)
        accuracy = (matches.sum().item() - (model_config.target_size - total_non_mask_count)) / total_non_mask_count
            
        # Get top k predictions if we have a vocabulary
        
        top_k_values, top_k_indices = torch.topk(probs, k=5, dim=-1)
        top_k_tokens = [[idx.item() for idx in batch] for batch in top_k_indices[0]]
            
        results.append({
            'layer': layer_name,
            'accuracy': accuracy,
            'top_k': top_k_tokens,
            'logits': logits,
            'probs': probs,
            'predictions': tokenizer.decode(predictions[0].cpu().numpy()),
        })
    
    return results

# Example usage:
def run_logit_lens(model_config, input_sequence):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_from_config(model_config)
    tokenizer = load_tokenizer(model_config.tokenizer_path)
    tokenized_input_sequence = torch.tensor(
        tokenizer.encode(input_sequence)).unsqueeze(0).to(device)
    with torch.no_grad():
        # Get logits at each layer
        layer_logits = logit_lens_analysis(model, tokenized_input_sequence)
        
        # Analyze how predictions evolve
        # target_indices = input_sequence[:, -model.config.target_size:]  # Assuming these are the targets
        results = analyze_predictions(layer_logits, model_config, tokenizer, tokenized_input_sequence)
        breakpoint()
        return results, layer_logits
    

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F

def visualize_layer_logits(results):
    """
    Create visualizations for layer-wise logit analysis.
    
    Args:
        results: List of dictionaries containing layer-wise analysis results
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Extract layer names and accuracies
    layers = [r['layer'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Accuracy progression plot
    plt.subplot(2, 1, 1)
    plt.plot(accuracies, marker='o')
    plt.title('Prediction Accuracy Through Layers')
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(layers)), layers, rotation=45)
    plt.grid(True)
    
    # 2. Top token probability heatmap
    plt.subplot(2, 1, 2)
    probs_matrix = []
    for r in results:
        # Get probabilities for top 5 tokens of the last position
        probs = r['probs'][0, -1, :].cpu().numpy()
        top_k_indices = torch.topk(r['probs'][0, -1, :], k=5)[1].cpu().numpy()
        top_k_probs = probs[top_k_indices]
        probs_matrix.append(top_k_probs)
    
    probs_matrix = np.array(probs_matrix)
    sns.heatmap(probs_matrix, 
                cmap='YlOrRd',
                xticklabels=['Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5'],
                yticklabels=layers)
    plt.title('Top-5 Token Probabilities Across Layers')
    plt.xlabel('Top Predicted Tokens')
    plt.ylabel('Layer')
    
    plt.tight_layout()
    return plt.gcf()

def compare_attention_mlp_logits(layer_logits, model_config, num_last_tokens=8):
    """
    Compare logit distributions after attention and MLP operations across layers
    for the last n tokens in the sequence.
    
    Args:
        layer_logits: List of tuples containing (layer_name, logits)
        num_last_tokens: Number of last tokens to analyze (default=9)
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Separate attention and MLP layer logits
    attn_layers = [(name, logits) for name, logits in layer_logits if 'attn' in name]
    mlp_layers = [(name, logits) for name, logits in layer_logits if 'mlp' in name]
    
    # Create a figure with subplots for each token position
    fig = plt.figure(figsize=(20, 4 * num_last_tokens))

    x_ticks = np.arange(model_config.vocab_size)
    tokenizer = load_tokenizer(model_config.tokenizer_path)
    x_labels = [tokenizer.token_to_id.get(i, str(i)) for i in range(model_config.vocab_size)]
    
    for token_idx in range(num_last_tokens):
        # Position from the end of sequence
        pos = -(num_last_tokens - token_idx + 1)
        
        # Create subplots for attention and MLP
        ax1 = plt.subplot(num_last_tokens, 2, 2*token_idx + 1)
        ax2 = plt.subplot(num_last_tokens, 2, 2*token_idx + 2)
        
        # Plot logits after attention layers
        for i, (name, logits) in enumerate(attn_layers):
            logit_values = logits[0, pos, :].cpu().numpy()
            ax1.plot(logit_values, alpha=0.5, label=f'Layer {i}')
        ax1.set_title(f'Attention Logits (Token Position {pos})')
        ax1.set_xlabel('Vocabulary Index')
        ax1.set_ylabel('Logit Value')
        ax1.legend()
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_labels, rotation=45, ha='right')
        ax1.grid(True)
        
        # Plot logits after MLP layers
        for i, (name, logits) in enumerate(mlp_layers):
            logit_values = logits[0, pos, :].cpu().numpy()
            ax2.plot(logit_values, alpha=0.5, label=f'Layer {i}')
        ax2.set_title(f'MLP Logits (Token Position {pos})')
        ax2.set_xlabel('Vocabulary Index')
        ax2.set_ylabel('Logit Value')
        ax2.legend()
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_labels, rotation=45, ha='right')
        ax2.grid(True)
    
    plt.suptitle('Comparison of Logit Distributions After Attention vs MLP Operations\nFor Last 9 Tokens', y=1.02)
    plt.tight_layout()
    return plt.gcf()

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    input1 = [
        "E", "striped", "B", "green", "D", "two", "B", "oval", "C", "green", "D", "green", "D", "solid", "E", "two", "B", "one", "D", "oval", "E", "green", "C", "one", "A", "green", "C", "open", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
        ">", "A", "B", "C", ".", "_", "_", "_", "_"
    ]

    model_config = GPTConfig44_Complete()
    results, layer_logits = run_logit_lens(model_config, input_sequence=input1)
    
    # logits_fig = visualize_layer_logits(results)
    compare_fig = compare_attention_mlp_logits(layer_logits, model_config, num_last_tokens=8)

    # logits_fig.savefig('COMPLETE_FIGS/logit_lens_results.png', bbox_inches="tight")
    compare_fig.savefig('COMPLETE_FIGS/logit_lens_comparison.png', bbox_inches="tight")


