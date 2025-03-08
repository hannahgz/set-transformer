import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import random
from torch.nn import functional as F
from tokenizer import load_tokenizer
from model import GPTConfig44_Complete, GPT
from data_utils import initialize_loaders, pretty_print_input

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'
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


@torch.no_grad()
def embedding_ablation_study(model, base_input, target_layer, position_to_ablate, tokenizer, target_pos=41, noise_scale=1.0, replace_with_zeros=False, generate_fig=False):
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

        if not generate_fig:
            return results

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

        # # Plot modified probabilities for the same tokens
        # modified_values = modified_probs[token_indices].cpu().numpy()
        # ax2.bar(token_labels, modified_values)
        # ax2.set_title(
        #     f'Modified Prediction Probabilities (Layer {target_layer}, Pos {position_to_ablate})')
        # ax2.set_xlabel('Token ID')
        # ax2.set_ylabel('Probability')

        # Get top tokens for this modified distribution
        modified_token_indices = torch.argsort(modified_probs, descending=True)[
            :top_k].cpu().numpy()
        modified_token_labels = tokenizer.decode(modified_token_indices)
        modified_values = modified_probs[modified_token_indices].cpu().numpy()

        ax2.bar(modified_token_labels, modified_values)
        ax2.set_title(
            f'Modified Prediction Probabilities (Layer {target_layer}, Pos {position_to_ablate}, (KL: {kl_div:.3f}))')

        ax2.set_xlabel('Token ID')
        ax2.set_ylabel('Probability')

        plt.tight_layout()
        results['figure'] = fig

        return results


# @torch.no_grad()
# def embedding_ablation_study_layer(model, base_input, target_layer, tokenizer,
#                                    target_pos=41, noise_scale=1.0, replace_with_zeros=False):
#     """
#     Performs an ablation study for an entire layer, creating subplots for each position.

#     Args:
#         model: The GPT model
#         base_input: Input sequence tensor
#         target_layer: Which layer's embeddings to modify (0-indexed)
#         tokenizer: Tokenizer to decode token IDs
#         target_pos: Which position to examine in the output logits
#         noise_scale: Scale of Gaussian noise to add (if not replacing with zeros)
#         replace_with_zeros: If True, replace with zeros instead of noise

#     Returns:
#         Figure with subplots for each position in the sequence
#     """
#     sequence_length = 40
#     target_pos -= 1  # 0-indexed, predicts result at first position

#     # Step 1: Get the base model prediction
#     base_logits, _, _, _, _ = model(base_input, get_loss=True)
#     base_logits = base_logits[:, target_pos].squeeze(0)
#     base_probs = F.softmax(base_logits, dim=-1)

#     # Determine top_k tokens to display in the plots
#     top_k = 10
#     base_token_indices = torch.argsort(base_probs, descending=True)[
#         :top_k].cpu().numpy()

#     # Create a grid of subplots based on sequence length
#     # Calculate grid dimensions (aim for roughly square layout)
#     import math
#     grid_size = math.ceil(math.sqrt(sequence_length + 1)
#                           )  # +1 for the base prediction

#     fig = plt.figure(figsize=(5*grid_size, 5*grid_size))

#     # First subplot is for base prediction
#     ax_base = fig.add_subplot(grid_size, grid_size, 1)

#     # Create individual token labels instead of a single string
#     base_token_labels = [tokenizer.decode([idx]) for idx in base_token_indices]

#     # Plot base probabilities
#     base_values = base_probs[base_token_indices].cpu().numpy()
#     ax_base.bar(base_token_labels, base_values)
#     ax_base.set_title('Base Prediction')
#     ax_base.set_xlabel('Token')
#     ax_base.set_ylabel('Probability')
#     ax_base.tick_params(axis='x', rotation=45)

#     # Store KL divergences for a possible summary plot
#     kl_divergences = []

#     # Now run ablation for each position in the sequence
#     for position_to_ablate in range(sequence_length):
#         # Get results for this position
#         with torch.no_grad():
#             # Get token and position embeddings
#             b, t = base_input.size()
#             pos = torch.arange(0, t, dtype=torch.long,
#                                device=base_input.device)
#             tok_emb = model.transformer.wte(base_input)
#             pos_emb = model.transformer.wpe(pos)
#             x = tok_emb + pos_emb

#             # Run through layers up to the target layer
#             for layer_idx, block in enumerate(model.transformer.h):
#                 if layer_idx < target_layer:
#                     x, _, _ = block(x)
#                 else:
#                     break

#             # Capture the embedding at this layer
#             original_embedding = x.clone()

#             # Create a modified embedding
#             modified_embedding = original_embedding.clone()

#             # Replace the target position with either noise or zeros
#             if replace_with_zeros:
#                 # Replace with zeros
#                 modified_embedding[0, position_to_ablate, :] = 0
#             else:
#                 # Replace with Gaussian noise scaled to match embedding norm
#                 emb_norm = torch.norm(
#                     original_embedding[0, position_to_ablate, :])
#                 noise = torch.randn_like(
#                     original_embedding[0, position_to_ablate, :]) * noise_scale
#                 # Scale noise to match the original embedding norm
#                 noise = noise * (emb_norm / torch.norm(noise))
#                 modified_embedding[0, position_to_ablate, :] = noise

#             # Continue the forward pass with modified embeddings
#             modified_x = modified_embedding

#             # Continue through remaining layers
#             for layer_idx, block in enumerate(model.transformer.h):
#                 if layer_idx >= target_layer:
#                     modified_x, _, _ = block(modified_x)

#             # Final layer norm
#             modified_x = model.transformer.ln_f(modified_x)

#             # Get logits
#             modified_logits = model.lm_head(modified_x)
#             modified_logits = modified_logits[:, target_pos].squeeze(0)
#             modified_probs = F.softmax(modified_logits, dim=-1)
#             modified_prediction = torch.argmax(modified_probs).item()

#             # Calculate KL divergence
#             kl_div = F.kl_div(
#                 base_probs.log(),
#                 modified_probs,
#                 reduction='batchmean'
#             ).item()

#             kl_divergences.append(kl_div)

#             # Add subplot for this position
#             # +2 because base is at position 1
#             ax = fig.add_subplot(grid_size, grid_size, position_to_ablate + 2)

#             # Get top tokens for this modified distribution
#             modified_token_indices = torch.argsort(modified_probs, descending=True)[
#                 :top_k].cpu().numpy()

#             # Create individual token labels for each token
#             modified_token_labels = [tokenizer.decode(
#                 [idx]) for idx in modified_token_indices]
#             modified_values = modified_probs[modified_token_indices].cpu(
#             ).numpy()

#             ax.bar(modified_token_labels, modified_values)
#             ax.set_title(f'Position {position_to_ablate} (KL: {kl_div:.4f})')
#             ax.set_xlabel('Token')
#             ax.tick_params(axis='x', rotation=45)

#             # Only add y-label for leftmost plots
#             if (position_to_ablate + 2) % grid_size == 1:
#                 ax.set_ylabel('Probability')

#     plt.tight_layout()

#     # Create an additional figure showing KL divergence by position
#     kl_fig = plt.figure(figsize=(10, 6))
#     ax_kl = kl_fig.add_subplot(111)
#     ax_kl.bar(range(sequence_length), kl_divergences)
#     ax_kl.set_title(f'KL Divergence by Position (Layer {target_layer})')
#     ax_kl.set_xlabel('Position')
#     ax_kl.set_ylabel('KL Divergence')

#     return fig, kl_fig


def comprehensive_embedding_ablation(model, base_input, layers_to_ablate, positions_to_ablate, tokenizer, target_pos=41, noise_scale=1.0, replace_with_zeros=False):
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
                model, base_input, layer, position, tokenizer,
                target_pos, noise_scale, replace_with_zeros
            )
            layer_results[position] = result
            kl_matrix[i, j] = result['kl_div']
        results[layer] = layer_results

    # # Create heatmap visualization
    # plt.figure(figsize=(20, 10))

    # sns.heatmap(kl_matrix, annot=True, fmt=".2f", cmap="viridis",
    #             xticklabels=positions_to_ablate, yticklabels=layers_to_ablate)

    # plt.xlabel('Sequence Position')
    # plt.ylabel('Layer')
    # plt.title('Impact of Embedding Ablation (KL Divergence)')

    # plt.tight_layout()

    heatmap_fig = generate_heatmap_from_kl_matrix(
        kl_matrix, positions_to_ablate, layers_to_ablate)

    results['heatmap_figure'] = heatmap_fig
    results['kl_matrix'] = kl_matrix

    return results


# def generate_heatmap_from_kl_matrix(kl_matrix, positions_to_ablate, layers_to_ablate):
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     # Create heatmap visualization
#     plt.figure(figsize=(24, 6))

#     # Generate heatmap
#     heatmap = sns.heatmap(
#         kl_matrix, annot=True, fmt=".2f", cmap="viridis",
#         xticklabels=positions_to_ablate, yticklabels=layers_to_ablate,
#         cbar_kws={'label': 'KL Divergence'}  # No fontsize here
#     )

#     # Customize colorbar font size
#     # Adjust colorbar tick labels
#     heatmap.figure.axes[-1].tick_params(labelsize=14)

#     # Set font sizes for labels, ticks, and title
#     plt.xlabel('Sequence Position', fontsize=18)
#     plt.ylabel('Layer', fontsize=18, rotation=0)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14, rotation=0)
#     plt.title('Impact of Embedding Ablation (KL Divergence)', fontsize=20)

#     plt.tight_layout()

#     return plt.gcf()

def generate_heatmap_from_kl_matrix(kl_matrix, positions_to_ablate, layers_to_ablate):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create figure with more padding on the left side
    plt.figure(figsize=(24, 6))
    
    # Add padding to the left margin to make room for the y-label
    plt.subplots_adjust(left=0.07)  # Increase left margin
    
    # Generate heatmap with tighter colorbar spacing
    heatmap = sns.heatmap(
        kl_matrix, annot=True, fmt=".2f", cmap="viridis",
        xticklabels=positions_to_ablate, yticklabels=layers_to_ablate,
        cbar_kws={'label': 'KL Divergence', 'pad': 0.02}  # Reduced padding between heatmap and colorbar
    )
    
    # Customize colorbar font size
    heatmap.figure.axes[-1].tick_params(labelsize=14)
    
    # Set font sizes for labels, ticks, and title
    plt.xlabel('Sequence Position', fontsize=18)
    
    # Position the y-label to the left of the plot
    plt.ylabel('Layer', fontsize=18, rotation=0, ha='right', va='center')
    
    # Add padding between y-label and y-axis
    plt.gca().yaxis.set_label_coords(-0.06, 0.5)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, rotation=0)
    plt.title('Impact of Embedding Ablation (KL Divergence)', fontsize=20)
    
    # Use tight_layout with specific padding settings
    plt.tight_layout(pad=1.1, w_pad=0.5, h_pad=0.5)
    
    return plt.gcf()


@torch.no_grad()
def comprehensive_ablation_batch_optimized(model, data_loader, layers_to_ablate, positions_to_ablate, tokenizer,
                                           target_pos=41, noise_scale=1.0, replace_with_zeros=False,
                                           max_batches=None):
    """
    Optimized version that processes entire batches at once for better efficiency.
    When possible, this version computes base predictions for the whole batch in one pass.

    Args:
        model: The GPT model
        data_loader: PyTorch DataLoader containing input sequences
        layers_to_ablate: List of layers to ablate
        positions_to_ablate: List of positions to ablate
        tokenizer: Tokenizer for token decoding
        target_pos: Which position to examine in the output logits
        noise_scale: Scale of Gaussian noise to add
        replace_with_zeros: If True, replace with zeros instead of noise
        max_batches: Maximum number of batches to process (None = process all)
        device: Device to run computations on ('cuda' or 'cpu')

    Returns:
        Dictionary with results and visualization
    """
    target_pos -= 1
    results = {}

    # Initialize KL matrix to accumulate results
    kl_matrix = np.zeros((len(layers_to_ablate), len(positions_to_ablate)))

    # Count total sequences and batches processed
    total_sequences = 0
    batch_count = 0

    no_set_token = tokenizer.token_to_id["*"]

    # Process each batch
    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = batch.to(device)
        batch_size = batch.shape[0]

        # Get base predictions for the entire batch at once
        with torch.no_grad():
            base_logits, _, _, _, _ = model(batch, get_loss=True)
            base_logits = base_logits[:, target_pos].squeeze(-1)
            base_probs = F.softmax(base_logits, dim=-1)

        # Process each sequence in the batch for each layer and position
        for i, layer in enumerate(layers_to_ablate):
            for j, position in enumerate(positions_to_ablate):
                batch_kl_sum = 0.0
                valid_count = 0

                for seq_idx in range(batch_size):

                    # Get single sequence
                    input_seq = batch[seq_idx].unsqueeze(0)

                    if no_set_token not in input_seq:
                        continue

                    # Get token and position embeddings for this sequence
                    with torch.no_grad():
                        b, t = input_seq.size()
                        pos = torch.arange(
                            0, t, dtype=torch.long, device=device)
                        tok_emb = model.transformer.wte(input_seq)
                        pos_emb = model.transformer.wpe(pos)
                        x = tok_emb + pos_emb

                        # Run through layers up to target layer
                        for layer_idx, block in enumerate(model.transformer.h):
                            if layer_idx < layer:
                                x, _, _ = block(x)
                            else:
                                break

                        # Modify embedding at target position
                        modified_embedding = x.clone()

                        if replace_with_zeros:
                            modified_embedding[0, position, :] = 0
                        else:
                            emb_norm = torch.norm(x[0, position, :])
                            noise = torch.randn_like(
                                x[0, position, :]) * noise_scale
                            noise = noise * (emb_norm / torch.norm(noise))
                            modified_embedding[0, position, :] = noise

                        # Continue forward pass with modified embeddings
                        for layer_idx, block in enumerate(model.transformer.h):
                            if layer_idx >= layer:
                                modified_embedding, _, _ = block(
                                    modified_embedding)

                        # Final layer norm and get logits
                        modified_embedding = model.transformer.ln_f(
                            modified_embedding)
                        modified_logits = model.lm_head(modified_embedding)
                        # Adjust for 0-indexing
                        modified_logits = modified_logits[:,
                                                          target_pos].squeeze(0)
                        modified_probs = F.softmax(modified_logits, dim=-1)

                        # Calculate KL divergence for this sequence
                        kl_div = F.kl_div(
                            base_probs[seq_idx].log().unsqueeze(0),
                            modified_probs,
                            reduction='batchmean'
                        ).item()

                        batch_kl_sum += kl_div
                        valid_count += 1

                # Add average KL for this layer/position to the matrix
                if valid_count > 0:
                    kl_matrix[i, j] += batch_kl_sum

            # Print progress
            print(
                f"Processed batch {batch_idx+1}/{len(data_loader)}, layer {layer} ({i+1}/{len(layers_to_ablate)})")

        total_sequences += batch_size
        batch_count += 1

    # Calculate the final average
    kl_matrix /= total_sequences

    # Create heatmap visualization
    plt.figure(figsize=(20, 10))

    sns.heatmap(kl_matrix, annot=True, fmt=".3f", cmap="viridis",
                xticklabels=positions_to_ablate, yticklabels=layers_to_ablate)

    plt.xlabel('Sequence Position')
    plt.ylabel('Layer')
    plt.title(
        f'Average Impact of Embedding Ablation (KL Divergence) across {total_sequences} samples')

    plt.tight_layout()

    results['heatmap_figure'] = plt.gcf()
    results['kl_matrix'] = kl_matrix
    results['total_sequences'] = total_sequences
    results['batch_count'] = batch_count

    return results


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    config = GPTConfig44_Complete()
    checkpoint = torch.load(config.filename, weights_only=False)

    # Create the model architecture
    model = GPT(config).to(device)

    # Load the weights
    model.load_state_dict(checkpoint['model'])
    model.eval()  # Set to evaluation mode

    tokenizer = load_tokenizer(config.tokenizer_path)

    # input_seq = [
    #     "B", "diamond", "C", "green", "B", "three", "B", "green", "D", "two",
    #     "E", "pink", "E", "two", "A", "blue", "A", "one", "D", "solid",
    #     "C", "two", "A", "open", "D", "squiggle", "B", "solid", "D", "green",
    #     "A", "diamond", "E", "diamond", "C", "solid", "C", "diamond", "E", "solid",
    #     ">", "*", ".", ".", ".", ".", ".", "."
    # ]
    # First encode the sequence - this returns a list
    # encoded_seq = tokenizer.encode(input_seq)

    # # Convert the list to a PyTorch tensor and add batch dimension
    # base_input = torch.tensor(
    #     encoded_seq, dtype=torch.long).unsqueeze(0).to(device)

    # target_layer = 1
    # position_to_ablate = 0
    # replace_with_zeros = False

    # ablate_type = "noise"
    # if replace_with_zeros:
    #     ablate_type = "zeros"

    # for target_layer in range(4):
    #     layer_fig, kl_fig = embedding_ablation_study_layer(
    #         model=model,
    #         base_input=base_input,
    #         target_layer=target_layer,
    #         tokenizer=tokenizer,
    #         target_pos=41,
    #         noise_scale=1.0,
    #         replace_with_zeros=replace_with_zeros)
    #     fig_save_path = f"COMPLETE_FIGS/ablation_study/layer_{target_layer}"
    #     os.makedirs(fig_save_path, exist_ok=True)

    #     layer_fig.savefig(
    #         os.path.join(fig_save_path, f"embedding_ablation_layer_{target_layer}_ablate_type_{ablate_type}.png"), bbox_inches="tight")

    # print(f"Layer {target_layer}, Position {position_to_ablate}")
    # results = embedding_ablation_study(
    #     model=model,
    #     base_input=base_input,
    #     target_layer=target_layer,
    #     position_to_ablate=position_to_ablate,
    #     tokenizer=tokenizer,
    #     target_pos=41,
    #     noise_scale=1.0,
    #     replace_with_zeros=replace_with_zeros,
    #     generate_fig=True)

    # fig_save_path = f"COMPLETE_FIGS/ablation_study/layer_{target_layer}/ablate_type_{ablate_type}"
    # os.makedirs(fig_save_path, exist_ok=True)
    # results["figure"].savefig(
    #     os.path.join(fig_save_path, f"embedding_ablation_position_{position_to_ablate}.png"), bbox_inches="tight")

    # for target_layer in range(4):
    #     for position_to_ablate in range(40):
    #         print(f"Layer {target_layer}, Position {position_to_ablate}")
    #         results = embedding_ablation_study(
    #             model=model,
    #             base_input=base_input,
    #             target_layer=target_layer,
    #             position_to_ablate=position_to_ablate,
    #             tokenizer=tokenizer,
    #             target_pos=41,
    #             noise_scale=1.0,
    #             replace_with_zeros=replace_with_zeros,
    #             generate_fig=True)

    #         fig_save_path = f"COMPLETE_FIGS/ablation_study/layer_{target_layer}/ablate_type_{ablate_type}"
    #         os.makedirs(fig_save_path, exist_ok=True)
    #         results["figure"].savefig(
    #             os.path.join(fig_save_path, f"embedding_ablation_position_{position_to_ablate}.png"), bbox_inches="tight")

    # comprehensive_results = comprehensive_embedding_ablation(
    #     model=model,
    #     base_input=base_input,
    #     layers_to_ablate=[0, 1, 2, 3],
    #     positions_to_ablate=range(40),
    #     tokenizer=tokenizer,
    #     target_pos=41,
    #     noise_scale=1.0,
    #     replace_with_zeros=replace_with_zeros)

    # # PIPELINE FOR GENERATING ABLATION HEATMAP

    # replace_with_zeros = False

    # ablate_type = "noise"
    # if replace_with_zeros:
    #     ablate_type = "zeros"

    # dataset_path = f"{PATH_PREFIX}/base_card_randomization_tuple_randomization_dataset.pth"
    # dataset = torch.load(dataset_path)
    # train_loader, val_loader = initialize_loaders(config, dataset)

    # batch_results = comprehensive_ablation_batch_optimized(
    #     model=model,
    #     data_loader=train_loader,
    #     layers_to_ablate=range(4),
    #     positions_to_ablate=range(40),
    #     tokenizer=tokenizer,
    #     target_pos=41,
    #     noise_scale=1.0,
    #     replace_with_zeros=replace_with_zeros,
    #     max_batches=10)

    # breakpoint()
    # # Save the heatmap figure
    # fig_save_path = f"COMPLETE_FIGS/ablation_study"
    # os.makedirs(fig_save_path, exist_ok=True)
    # batch_results['heatmap_figure'].savefig(
    #     os.path.join(fig_save_path, f"avg_embedding_ablation_heatmap_ablate_type_{ablate_type}.png"), bbox_inches="tight")

    # matrix_path = f"results/ablation_study"
    # os.makedirs(matrix_path, exist_ok=True)
    # np.save(os.path.join(matrix_path, f"avg_kl_divergence_matrix_ablate_type_{ablate_type}.npy"), batch_results['kl_matrix'])

    ablate_type = "noise"
    matrix_path = f"results/ablation_study"
    os.makedirs(matrix_path, exist_ok=True)

    # Load the KL divergence matrix
    loaded_kl_matrix = np.load(os.path.join(
        matrix_path, f"avg_kl_divergence_matrix_ablate_type_{ablate_type}.npy"))

    fig = generate_heatmap_from_kl_matrix(
        loaded_kl_matrix, range(40), range(1, 5))
    fig_save_path = f"COMPLETE_FIGS/paper/ablation_study"
    os.makedirs(fig_save_path, exist_ok=True)
    fig.savefig(os.path.join(
        fig_save_path, f"avg_embedding_ablation_heatmap_ablate_type_{ablate_type}.png"), bbox_inches="tight")

    # # Save the KL divergence matrix
    # matrix_path = f"results/ablation_study"
    # os.makedirs(matrix_path, exist_ok=True)
    # np.save(os.path.join(matrix_path, f"kl_divergence_matrix_ablate_type_{ablate_type}.npy"), comprehensive_results['kl_matrix'])

    # fig_save_path = f"COMPLETE_FIGS/ablation_study/layer_{target_layer}/ablate_type_{ablate_type}"
    # os.makedirs(fig_save_path, exist_ok=True)
    # results["figure"].savefig(
    #     f"embedding_ablation_position_{position_to_ablate}.png", bbox_inches="tight")
