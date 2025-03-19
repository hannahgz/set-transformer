import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pickle
import os
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from data_utils import split_data
from tokenizer import load_tokenizer
from model import GPT, GPTConfig44_Complete
from data_utils import initialize_loaders
from dataclasses import dataclass
from torch.nn import functional as F

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

def load_model_from_config(config, device=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT(config).to(device)
    checkpoint = torch.load(config.filename, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def compute_card_attribute_embedding_similarities(model_config, capture_layer, batch_limit=None, use_global_centroid=False):
    """
    Computes average cosine similarities between attribute embeddings grouped by their corresponding cards.
    
    Args:
        model_config: Configuration for the model
        capture_layer: Which layer to extract embeddings from
        batch_limit: Optional limit on number of batches to processf
    
    Returns:
        similarity_matrix: NxN matrix where N is the number of unique cards
        card_embeddings_dict: Dictionary mapping cards to their collected attribute embeddings
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    model = load_model_from_config(model_config).to(device)
    tokenizer = load_tokenizer(model_config.tokenizer_path)
    
    # Dictionary to store attribute embeddings for each card
    card_embeddings_dict = {}
    all_embeddings = []
    
    # Load dataset
    dataset = torch.load(model_config.dataset_path)
    _, val_loader = initialize_loaders(model_config, dataset)
    
    # Process batches
    for batch_idx, batch in enumerate(val_loader):
        print(f"Processing batch {batch_idx+1}/{len(val_loader)}")
        # Optional batch limit for testing or memory constraints
        if batch_limit and batch_idx >= batch_limit:
            break
            
        print(f"Processing batch {batch_idx+1}/{len(val_loader)}")
        batch = batch.to(device)
        
        # Get embeddings from the specified layer
        with torch.no_grad():
            _, _, _, layer_embeddings, _ = model(batch, capture_layer=capture_layer)
        
        # Process each sequence in the batch
        for seq_idx, (sequence, embeddings) in enumerate(zip(batch, layer_embeddings)):
            # Process pairs of (card, attribute) in the sequence
            for pos in range(0, model_config.input_size-1, 2):
                # Get card token and its corresponding attribute embedding
                card_token = sequence[pos].item()
                card_text = tokenizer.id_to_token[card_token]
                
                # Get the embedding of the attribute that follows the card
                attr_embedding = embeddings[pos+1].cpu().numpy()
                
                # Store embedding grouped by card
                if card_text not in card_embeddings_dict:
                    card_embeddings_dict[card_text] = []
                card_embeddings_dict[card_text].append(attr_embedding)
                if use_global_centroid:
                    all_embeddings.append(attr_embedding)

                # breakpoint()
    
    if use_global_centroid:
        all_embeddings = np.vstack(all_embeddings)
        global_centroid = np.mean(all_embeddings, axis=0)
        print(f"Global centroid shape: {global_centroid.shape}")

    # Convert lists of embeddings to arrays
    for card in card_embeddings_dict:
        if card_embeddings_dict[card]:
            card_embeddings_dict[card] = np.vstack(card_embeddings_dict[card])
            if use_global_centroid:
                card_embeddings_dict[card] -= global_centroid
    
    # Sort cards for consistent ordering (usually A, B, C, D, E)
    sorted_cards = sorted(card_embeddings_dict.keys())
    
    # Initialize similarity matrix
    n_cards = len(sorted_cards)
    similarity_matrix = np.zeros((n_cards, n_cards))
    
    # Compute centroid (mean) for each card's attribute embeddings
    card_centroids = {}
    for card in sorted_cards:
        if len(card_embeddings_dict[card]) > 0:
            card_centroids[card] = np.mean(card_embeddings_dict[card], axis=0)
    
    # Compute pairwise cosine similarities between centroids
    from sklearn.metrics.pairwise import cosine_similarity
    for i, card1 in enumerate(sorted_cards):
        for j, card2 in enumerate(sorted_cards):
            if card1 in card_centroids and card2 in card_centroids:
                similarity = cosine_similarity(
                    card_centroids[card1].reshape(1, -1),
                    card_centroids[card2].reshape(1, -1)
                )[0][0]
                similarity_matrix[i, j] = similarity
    
    return similarity_matrix, card_embeddings_dict, sorted_cards

def plot_card_attribute_similarity_matrix(similarity_matrix, card_labels, capture_layer):
    """
    Creates a heatmap visualization of the cosine similarities.
    
    Args:
        similarity_matrix: NxN matrix of similarity values
        card_labels: List of card labels (typically A, B, C, D, E)
        capture_layer: Layer number for the title
        
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        vmin=-1, 
        vmax=1,
        center=0,
        square=True,
        xticklabels=card_labels,
        yticklabels=card_labels
    )
    
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.title(f'Layer {capture_layer}: Average Cosine Similarity Between Attribute Embeddings by Card')
    plt.xlabel('Card')
    plt.ylabel('Card')
    
    plt.tight_layout()
    return plt.gcf()

def analyze_card_attribute_embeddings(model_config, capture_layer, use_global_centroid=False):
    """
    Complete analysis pipeline that computes similarities and generates visualization.
    
    Args:
        model_config: Model configuration
        capture_layer: Layer to analyze
        
    Returns:
        matplotlib figure, similarity matrix, and embedding dictionary
    """
    similarity_matrix, card_embeddings_dict, card_labels = compute_card_attribute_embedding_similarities(
        model_config, 
        capture_layer,
        use_global_centroid=use_global_centroid
    )
    
    fig = plot_card_attribute_similarity_matrix(
        similarity_matrix,
        card_labels,
        capture_layer
    )
    
    # Save results
    if not use_global_centroid:
        output_dir = f"{PATH_PREFIX}/complete/attribute_similarity"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save matrix for future reference
        np.save(f"{output_dir}/attr_similarity_matrix_layer{capture_layer}.npy", similarity_matrix)
        
        fig_dir = f"COMPLETE_FIGS/attribute_similarity"
        os.makedirs(fig_dir, exist_ok=True)
        # Save figure
        fig.savefig(f"{fig_dir}/attr_similarity_heatmap_layer{capture_layer}.png", bbox_inches="tight")
    else:
        output_dir = f"{PATH_PREFIX}/complete/attribute_similarity_global_centroid"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save matrix for future reference
        np.save(f"{output_dir}/attr_similarity_matrix_layer{capture_layer}.npy", similarity_matrix)
        
        fig_dir = f"COMPLETE_FIGS/attribute_similarity_global_centroid"
        os.makedirs(fig_dir, exist_ok=True)
        # Save figure
        fig.savefig(f"{fig_dir}/attr_similarity_heatmap_layer{capture_layer}.png", bbox_inches="tight")
    
    # Additional analysis: Print stats on number of embeddings collected per card
    print("\nAttribute embeddings collected per card:")
    for card in card_labels:
        count = len(card_embeddings_dict[card]) if card in card_embeddings_dict else 0
        print(f"Card {card}: {count} embeddings")
    
    return fig, similarity_matrix, card_embeddings_dict


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    config = GPTConfig44_Complete()

    analyze_card_attribute_embeddings(
        model_config=config,
        capture_layer=1,
        use_global_centroid=True
    )

    analyze_card_attribute_embeddings(
        model_config=config,
        capture_layer=0,
        use_global_centroid=True
    )
