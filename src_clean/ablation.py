import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from torch.nn import functional as F
from tokenizer import load_tokenizer
from model import GPTConfig44_Complete, GPT

# def token_ablation_study(model, base_input, token_replacement_positions, replacement, target_pos=41):
#     """
#     Perform ablation study by replacing target tokens at specific positions
#     and measuring the impact on the model's prediction.

#     Creates a bar chart showing KL divergence for each ablated position.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     results = {}
#     base_logits, _, _, _, _ = model(base_input, get_loss=False)
#     base_logits = base_logits[:, target_pos].squeeze(0)
#     base_probs = F.softmax(base_logits, dim=-1)
#     base_prediction = torch.argmax(base_probs).item()

#     for position in token_replacement_positions:
#         modified_input = base_input.clone()

#         modified_input[0, position] = replacement
#         modified_logits, _, _, _, _ = model(modified_input, get_loss=False)
#         modified_logits = modified_logits[:, target_pos].squeeze(0)
#         modified_probs = F.softmax(modified_logits, dim=-1)
#         modified_prediction = torch.argmax(modified_probs).item()

#         # Measure the KL divergence between the base and modified probability distributions
#         kl_div = F.kl_div(
#             base_probs.log(),
#             modified_probs,
#             reduction='batchmean'
#         ).item()

#         results[position] = {
#             'kl_div': kl_div,
#             'base_prediction': base_prediction,
#             'modified_prediction': modified_prediction
#         }

#     # Create visualization of KL divergence per position
#     plt.figure(figsize=(12, 6))
#     positions = sorted(results.keys())
#     kl_values = [results[pos]['kl_div'] for pos in positions]  # Extract KL div values

#     bars = plt.bar(positions, kl_values)

#     # Find the position with maximum KL divergence
#     max_kl_pos = max(positions, key=lambda pos: results[pos]['kl_div'])

#     # Highlight the maximum bar
#     max_idx = positions.index(max_kl_pos)
#     bars[max_idx].set_color('red')

#     plt.xlabel('Input Sequence Position')
#     plt.ylabel('KL Divergence')
#     plt.title('Impact of Token Ablation by Position')

#     # Add value labels on top of each bar
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
#                 f'{height:.4f}', ha='center', va='bottom', rotation=90)

#     plt.tight_layout()

#     return results, plt.gcf()  # Return results and the figure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embedding_ablation_study(model, base_input, target_layer, position_to_ablate, tokenizer, target_pos=41, noise_scale=1.0, replace_with_zeros=False):
    """
    Performs an ablation study by replacing embeddings at a specific layer and position
    with either noise or zeros, then measuring the impact on model predictions.

    Args:
        model: The GPT model
        base_input: Input sequence tensor
        target_layer: Which layer's embeddings to modify (0-indexed)
        position_to_ablate: Which position in the sequence to modify
        target_pos: Which position to examine in the output logits
        noise_scale: Scale of Gaussian noise to add (if not replacing with zeros)
        replace_with_zeros: If True, replace with zeros instead of noise

    Returns:
        Dictionary with results including KL divergence and visualization
    """
    results = {}
    target_pos -= 1  # 0-indexed, predicts result at first position

    # Step 1: Get the base model prediction
    base_logits, _, _, _, _ = model(base_input, get_loss=True)
    base_logits = base_logits[:, target_pos].squeeze(0)
    base_probs = F.softmax(base_logits, dim=-1)
    base_prediction = torch.argmax(base_probs).item()

    # Step 2: Run the model up to the target layer to get embeddings
    # We need to implement a way to get intermediate embeddings
    with torch.no_grad():
        # Get token and position embeddings
        b, t = base_input.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = model.transformer.wte(base_input)
        pos_emb = model.transformer.wpe(pos)
        x = tok_emb + pos_emb

        # Run through layers up to the target layer
        for layer_idx, block in enumerate(model.transformer.h):
            if layer_idx < target_layer:
                x, _, _ = block(x)
            else:
                break

        # Capture the embedding at this layer
        original_embedding = x.clone()

        # Create a modified embedding
        modified_embedding = original_embedding.clone()

        # Replace the target position with either noise or zeros
        if replace_with_zeros:
            # Replace with zeros
            modified_embedding[0, position_to_ablate, :] = 0
        else:
            # Replace with Gaussian noise scaled to match embedding norm
            emb_norm = torch.norm(original_embedding[0, position_to_ablate, :])
            noise = torch.randn_like(
                original_embedding[0, position_to_ablate, :]) * noise_scale
            # Scale noise to match the original embedding norm
            noise = noise * (emb_norm / torch.norm(noise))
            modified_embedding[0, position_to_ablate, :] = noise

        # Continue the forward pass with modified embeddings
        modified_x = modified_embedding

        # Continue through remaining layers
        for layer_idx, block in enumerate(model.transformer.h):
            if layer_idx >= target_layer:
                modified_x, _, _ = block(modified_x)

        # Final layer norm
        modified_x = model.transformer.ln_f(modified_x)

        # Get logits
        modified_logits = model.lm_head(modified_x)
        modified_logits = modified_logits[:, target_pos].squeeze(0)
        modified_probs = F.softmax(modified_logits, dim=-1)
        modified_prediction = torch.argmax(modified_probs).item()

        # Calculate KL divergence
        kl_div = F.kl_div(
            base_probs.log(),
            modified_probs,
            reduction='batchmean'
        ).item()

        # Store results
        results = {
            'kl_div': kl_div,
            'base_prediction': base_prediction,
            'modified_prediction': modified_prediction,
            'base_probs': base_probs.cpu().numpy(),
            'modified_probs': modified_probs.cpu().numpy()
        }

        # Create visualizations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot probability distributions
        top_k = 10  # Show top-k tokens
        token_indices = torch.argsort(base_probs, descending=True)[
            :top_k].cpu().numpy()

        # Convert token IDs to string labels if possible (placeholder)
        # token_labels = [str(idx) for idx in token_indices]
        token_labels = tokenizer.decode(token_indices)

        # Plot base probabilities
        base_values = base_probs[token_indices].cpu().numpy()
        ax1.bar(token_labels, base_values)
        ax1.set_title('Base Prediction Probabilities')
        ax1.set_xlabel('Token ID')
        ax1.set_ylabel('Probability')

        # Plot modified probabilities for the same tokens
        modified_values = modified_probs[token_indices].cpu().numpy()
        ax2.bar(token_labels, modified_values)
        ax2.set_title(
            f'Modified Prediction Probabilities (Layer {target_layer}, Pos {position_to_ablate})')
        ax2.set_xlabel('Token ID')
        ax2.set_ylabel('Probability')

        plt.tight_layout()
        results['figure'] = fig

    return results


def comprehensive_embedding_ablation(model, base_input, layers_to_ablate, positions_to_ablate, target_pos=41, noise_scale=1.0, replace_with_zeros=False):
    """
    Performs embedding ablation across multiple layers and positions,
    creating a heatmap visualization of KL divergence.

    Args:
        model: The GPT model
        base_input: Input sequence tensor
        layers_to_ablate: List of layers to ablate
        positions_to_ablate: List of positions to ablate
        target_pos: Which position to examine in the output logits
        noise_scale: Scale of Gaussian noise to add
        replace_with_zeros: If True, replace with zeros instead of noise

    Returns:
        Dictionary with results and visualization
    """
    results = {}
    kl_matrix = np.zeros((len(layers_to_ablate), len(positions_to_ablate)))

    # Perform ablation for each layer and position combination
    for i, layer in enumerate(layers_to_ablate):
        layer_results = {}
        for j, position in enumerate(positions_to_ablate):
            print(f"Layer {layer}, Position {position}")
            result = embedding_ablation_study(
                model, base_input, layer, position,
                target_pos, noise_scale, replace_with_zeros
            )
            layer_results[position] = result
            kl_matrix[i, j] = result['kl_div']
        results[layer] = layer_results

    # Create heatmap visualization
    plt.figure(figsize=(12, 8))

    sns.heatmap(kl_matrix, annot=True, fmt=".4f", cmap="viridis",
                xticklabels=positions_to_ablate, yticklabels=layers_to_ablate)

    plt.xlabel('Sequence Position')
    plt.ylabel('Layer')
    plt.title('Impact of Embedding Ablation (KL Divergence)')

    plt.tight_layout()

    results['heatmap_figure'] = plt.gcf()
    results['kl_matrix'] = kl_matrix

    return results


if __name__ == "__main__":
    config = GPTConfig44_Complete()
    checkpoint = torch.load(config.filename, weights_only=False)

    # Create the model architecture
    model = GPT(config).to(device)

    # Load the weights
    model.load_state_dict(checkpoint['model'])
    model.eval()  # Set to evaluation mode

    tokenizer = load_tokenizer(config.tokenizer_path)

    input_seq = [
        "B", "diamond", "C", "green", "B", "three", "B", "green", "D", "two",
        "E", "pink", "E", "two", "A", "blue", "A", "one", "D", "solid",
        "C", "two", "A", "open", "D", "squiggle", "B", "solid", "D", "green",
        "A", "diamond", "E", "diamond", "C", "solid", "C", "diamond", "E", "solid",
        ">", "*", ".", ".", ".", ".", ".", "."
    ]
    # First encode the sequence - this returns a list
    encoded_seq = tokenizer.encode(input_seq)

    # Convert the list to a PyTorch tensor and add batch dimension
    base_input = torch.tensor(
        encoded_seq, dtype=torch.long).unsqueeze(0).to(device)

    target_layer = 0
    position_to_ablate = 2
    replace_with_zeros = False

    ablate_type = "noise"
    if replace_with_zeros:
        ablate_type = "zeros"

    # results = embedding_ablation_study(
    #     model=model,
    #     base_input=base_input,
    #     target_layer=target_layer,
    #     position_to_ablate=position_to_ablate,
    #     tokenizer=tokenizer,
    #     target_pos=41,
    #     noise_scale=1.0,
    #     replace_with_zeros=replace_with_zeros)
    # breakpoint()

    comprehensive_results = comprehensive_embedding_ablation(
        model=model,
        base_input=base_input,
        layers_to_ablate=[0, 1, 2, 3],
        positions_to_ablate=range(40),
        target_pos=41,
        noise_scale=1.0,
        replace_with_zeros=replace_with_zeros)
    
    breakpoint()

    # Save the heatmap figure
    fig_save_path = f"COMPLETE_FIGS/ablation_study"
    os.makedirs(fig_save_path, exist_ok=True)
    comprehensive_results['heatmap_figure'].savefig(
        os.path.join(fig_save_path, f"embedding_ablation_heatmap_ablate_type_{ablate_type}.png"), bbox_inches="tight")
    
    # Save the KL divergence matrix
    matrix_path = f"results/ablation_study"
    os.makedirs(matrix_path, exist_ok=True)
    np.save(os.path.join(matrix_path, f"kl_divergence_matrix_ablate_type_{ablate_type}.npy"), comprehensive_results['kl_matrix'])

    # fig_save_path = f"COMPLETE_FIGS/ablation_study/layer_{target_layer}/ablate_type_{ablate_type}"
    # os.makedirs(fig_save_path, exist_ok=True)
    # results["figure"].savefig(
    #     f"embedding_ablation_position_{position_to_ablate}.png", bbox_inches="tight")
