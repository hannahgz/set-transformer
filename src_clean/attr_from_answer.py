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
from all_attribute_from_last_attribute import train_binary_probe, init_binary_dataset, plot_all_layers_metrics
from sklearn.preprocessing import LabelEncoder
from data_utils import split_data
from tokenizer import load_tokenizer
from model import GPT, GPTConfig44_Complete
from data_utils import initialize_loaders
from dataclasses import dataclass
from torch.nn import functional as F
from tokenizer import load_tokenizer

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'


# def init_attr_from_predict_token(config, capture_layer):
#     dataset = torch.load(config.dataset_path)
#     train_loader, val_loader = initialize_loaders(config, dataset)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model = GPT(config).to(device)
#     checkpoint = torch.load(config.filename, weights_only=False)
#     model.load_state_dict(checkpoint["model"])
#     model.eval()

#     tokenizer = load_tokenizer(config.tokenizer_path)

#     predict_id = tokenizer.token_to_id[">"]


def init_attr_from_answer(config, capture_layer, val_loader, model):
    # dataset = torch.load(config.dataset_path)
    # train_loader, val_loader = initialize_loaders(config, dataset)
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = GPT(config).to(device)
    # checkpoint = torch.load(config.filename, weights_only=False)
    # model.load_state_dict(checkpoint["model"])
    # model.eval()

    tokenizer = load_tokenizer(config.tokenizer_path)

    A_id = tokenizer.token_to_id["A"]
    B_id = tokenizer.token_to_id["B"]
    C_id = tokenizer.token_to_id["C"]
    D_id = tokenizer.token_to_id["D"]
    E_id = tokenizer.token_to_id["E"]
    no_set_id = tokenizer.token_to_id["*"]

    all_input_embeddings = []
    all_target_attributes = []
    set_counts = []

    # batch.shape, torch.Size([512, 49])
    for batch_index, batch in enumerate(val_loader):
        print(f"Batch {batch_index + 1}/{len(val_loader)}")

        batch = batch.to(device)
        # captured_embedding.shape, torch.Size([512, 49, 64])
        _, _, _, captured_embedding, _ = model(batch, True, capture_layer)

        for seq_index, sequence in enumerate(batch):
            seen_card_dict = {
                A_id: [],
                B_id: [],
                C_id: [],
                D_id: [],
                E_id: []
            }
            sequence = sequence.tolist()
            for card_index, card_id in enumerate(sequence[0:(config.input_size-1):2]):
                # print(f"Card {card_id}, index {card_index}")

                attr_index = card_index * 2 + 1
                attr_id = sequence[attr_index]
                seen_card_dict[card_id].append(attr_id)

            if no_set_id in sequence:
                continue

            for card_index in range(config.input_size, config.block_size - 1):
                card_id = sequence[card_index]

                if card_id in [A_id, B_id, C_id, D_id, E_id]:
                    current_embedding = captured_embedding[seq_index,
                                                           card_index, :]

                    if card_index <= 43:
                        # second element in tuple represents number of sets
                        set_counts.append(1)
                    elif card_index > 43:
                        set_counts.append(2)
                    all_input_embeddings.append(current_embedding)
                    all_target_attributes.append(seen_card_dict[card_id])

    # After the loop completes, convert lists to tensors
    input_embeddings_tensor = torch.stack(all_input_embeddings)
    # This will create a tensor of shape [num_samples, 4]
    target_attributes_tensor = torch.tensor(all_target_attributes)

    save_path_dir = f"{PATH_PREFIX}/attr_from_answer/layer{capture_layer}"
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)

    # Save set_counts as a pkl file
    set_counts_path = f"{save_path_dir}/set_counts.pkl"
    with open(set_counts_path, 'wb') as f:
        pickle.dump(set_counts, f)

    # Save the tensors

    torch.save({
        'input_embeddings': input_embeddings_tensor,
        'target_attributes': target_attributes_tensor
    }, f"{save_path_dir}/embeddings_and_attributes.pt")

    return input_embeddings_tensor, target_attributes_tensor


def init_binary_probe_data(attribute_id, capture_layer):
    load_existing_dataset_path = f"{PATH_PREFIX}/attr_from_answer/layer{capture_layer}/embeddings_and_attributes.pt"
    saved_data = torch.load(load_existing_dataset_path)

    input_embeddings = saved_data['input_embeddings']
    target_attributes = saved_data['target_attributes']

    binary_targets = []
    for sample in range(len(target_attributes)):
        if sample % 1000 == 0:
            print(f"Processing sample {sample}/{len(target_attributes)}")
        if attribute_id in target_attributes[sample]:
            binary_targets.append(1)
        else:
            binary_targets.append(0)

    torch.save({
        'input_embeddings': input_embeddings,
        'binary_targets': torch.tensor(binary_targets).float()
    }, f"{PATH_PREFIX}/attr_from_answer/layer{capture_layer}/binary_dataset_{attribute_id}.pt")

def load_embeddings_and_probe(layer, attribute_id, project):
    """Load embeddings and binary probe for a specific layer and attribute"""
    # Load embeddings
    embeddings_path = f"{PATH_PREFIX}/{project}/layer{layer}/embeddings_and_attributes.pt"
    embeddings_data = torch.load(embeddings_path)
    embeddings = embeddings_data['input_embeddings']
    
    # Load binary probe
    probe_path = f"{PATH_PREFIX}/{project}/layer{layer}/attr_{attribute_id}/binary_probe_model.pt"
    if not os.path.exists(probe_path):
        probe_weights = None
    else:
        probe_weights = torch.load(probe_path)
    
    return embeddings, probe_weights

def compute_average_cosine_similarity(embeddings, probe_weights):
    """Compute average cosine similarity between embeddings and probe weights"""
    # Extract linear layer weights from probe state dict
    linear_weights = probe_weights['linear.weight'].squeeze()
    
    # Compute cosine similarity for each embedding
    similarities = []
    for index, emb in enumerate(embeddings):
        if index % 100000 == 0:
            print(f"Processing embedding {index + 1}/{len(embeddings)}")
        sim = F.cosine_similarity(emb.unsqueeze(0), linear_weights.unsqueeze(0))
        similarities.append(sim.item())
    
    # Return average similarity
    return np.mean(similarities)


def compute_similarity_matrix(layers, attributes, project, save_matrix_path=None):
    """Create heatmap of cosine similarities across layers and attributes"""
    # Initialize similarity matrix
    similarity_matrix = np.zeros((len(attributes), len(layers)))
    
    # Compute similarities for each layer and attribute
    for i, attr_id in enumerate(attributes):
        print(f"Processing attribute {attr_id}")
        for j, layer in enumerate(layers):
            print(f"Processing layer {layer}")
            try:
                embeddings, probe_weights = load_embeddings_and_probe(layer, attr_id, project)
                print(f"Embeddings shape: {embeddings.shape}")
                if probe_weights is None:
                    similarity = 0
                else:
                    similarity = compute_average_cosine_similarity(embeddings, probe_weights)
                similarity_matrix[i, j] = similarity
            except Exception as e:
                print(f"Error processing layer {layer}, attribute {attr_id}: {e}")
                similarity_matrix[i, j] = np.nan

    if save_matrix_path:
        if not os.path.exists(os.path.dirname(save_matrix_path)):
            os.makedirs(os.path.dirname(save_matrix_path))
        np.save(save_matrix_path, similarity_matrix)
        
    return similarity_matrix
    
def create_cosine_similarity_heatmap(layers, attributes, tokenizer_path, similarity_matrix, project, save_path=None):
    # Load tokenizer for attribute names
    tokenizer = load_tokenizer(tokenizer_path)

    # Create attribute labels using tokenizer
    attr_labels = [tokenizer.id_to_token[attr_id] for attr_id in attributes]
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        xticklabels=[f'Layer {l}' for l in layers],
        yticklabels=attr_labels,
        cmap='viridis',
        annot=True,
        fmt='.3f'
    )
    
    plt.title(f'{project}: Average Cosine Sim')
    plt.xlabel('Layers')
    plt.ylabel('Attributes')
    
    # Save figure if path provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return plt.gcf()

def load_probe_weights(layer, attribute_id):
    """Load binary probe weights for a specific layer and attribute"""
    probe_path = f"{PATH_PREFIX}/attr_from_answer/layer{layer}/attr_{attribute_id}/binary_probe_model.pt"
    if not os.path.exists(probe_path):
        return None
    else:
        return torch.load(probe_path)['linear.weight'].squeeze()

def compute_probe_similarity_per_layer(layer, attributes, save_matrix_path=None):
    """Create matrix of cosine similarities between probe weights for a specific layer"""
    # Initialize dictionary to store probe weights
    probe_weights_dict = {}
    
    # Load all probe weights for this layer
    for attr_id in attributes:
        try:
            weights = load_probe_weights(layer, attr_id)
            if weights is not None:
                probe_weights_dict[attr_id] = weights
                print(f"Loaded probe weights for layer {layer}, attribute {attr_id}")
            else:
                print(f"No probe weights found for layer {layer}, attribute {attr_id}")
        except Exception as e:
            print(f"Error loading probe weights for layer {layer}, attribute {attr_id}: {e}")
    
    # Initialize similarity matrix
    attr_ids = list(probe_weights_dict.keys())
    n = len(attr_ids)
    similarity_matrix = np.zeros((n, n))
    
    # Compute cosine similarities between all pairs of probe weights
    for i, attr_i in enumerate(attr_ids):
        for j, attr_j in enumerate(attr_ids):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Same weights, perfect similarity
            else:
                # Compute cosine similarity between probe weights
                weights_i = probe_weights_dict[attr_i]
                weights_j = probe_weights_dict[attr_j]
                sim = F.cosine_similarity(weights_i.unsqueeze(0), weights_j.unsqueeze(0))
                similarity_matrix[i, j] = sim.item()
    
    # Create a DataFrame for easier interpretation
    import pandas as pd
    sim_df = pd.DataFrame(similarity_matrix, index=attr_ids, columns=attr_ids)
    
    if save_matrix_path:
        if not os.path.exists(os.path.dirname(save_matrix_path)):
            os.makedirs(os.path.dirname(save_matrix_path))
        np.save(save_matrix_path, similarity_matrix)
        sim_df.to_csv(f"{save_matrix_path.replace('.npy', '.csv')}")

    return similarity_matrix, sim_df, attr_ids

def create_probe_similarity_heatmap_per_layer(layer, similarity_matrix, attr_ids, tokenizer_path, project, save_path=None):
    """Create heatmap of cosine similarities between probe weights for a specific layer"""
    # Load tokenizer for attribute names
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Create labels for the heatmap using attribute tokens
    labels = [tokenizer.id_to_token[attr_id] for attr_id in attr_ids]
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap='viridis',
        annot=True,  # Setting to True as the matrix should be smaller now
        fmt='.2f',
        vmin=-1,
        vmax=1
    )
    
    plt.title(f'{project}: Layer {layer} Probe Weights Cosine Similarity')
    plt.xlabel('Attribute')
    plt.ylabel('Attribute')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    # Save figure if path provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return plt.gcf()

def run_layer_probe_similarity_analysis(project, layers, attributes, tokenizer_path):
    """Run probe similarity analysis for each layer separately"""
    results = {}
    tokenizer = load_tokenizer(tokenizer_path)
    for layer in layers:
        print(f"\nProcessing Layer {layer}")
        matrix_save_path = f"{PATH_PREFIX}/attr_from_answer/layer{layer}_probe_similarity_matrix.npy"
        heatmap_save_path = f"COMPLETE_FIGS/attr_from_answer/layer{layer}_probe_similarity_heatmap.png"
        
        # Compute similarity matrix for this layer
        similarity_matrix, sim_df, attr_ids = compute_probe_similarity_per_layer(
            layer, attributes, save_matrix_path=matrix_save_path
        )
        
        # Create and save heatmap for this layer
        heatmap = create_probe_similarity_heatmap_per_layer(
            layer, similarity_matrix, attr_ids, tokenizer_path, project, save_path=heatmap_save_path
        )
        
        # Store results
        results[layer] = {
            'similarity_matrix': similarity_matrix,
            'dataframe': sim_df,
            'attribute_ids': attr_ids,
            'heatmap': heatmap
        }
        
        # Print some statistics for this layer
        if len(attr_ids) > 1:
            # Extract upper triangle (excluding diagonal)
            upper_tri = np.triu(similarity_matrix, k=1)
            valid_values = upper_tri[upper_tri != 0]
            
            if len(valid_values) > 0:
                print(f"Layer {layer} Statistics:")
                print(f"  Mean similarity: {np.mean(valid_values):.4f}")
                print(f"  Max similarity: {np.max(valid_values):.4f}")
                print(f"  Min similarity: {np.min(valid_values):.4f}")
                
                # Find max similarity pair
                max_idx = np.unravel_index(np.argmax(upper_tri), upper_tri.shape)
                attr1 = tokenizer.id_to_token[attr_ids[max_idx[0]]]
                attr2 = tokenizer.id_to_token[attr_ids[max_idx[1]]]
                max_sim = upper_tri[max_idx]
                print(f"  Most similar pair: {attr1} ↔ {attr2} ({max_sim:.4f})")
    
    print("\nCompleted analysis for all layers")
    return results

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    config = GPTConfig44_Complete()

    config = GPTConfig44_Complete()
    project = "Attribute From Answer"
    run_layer_probe_similarity_analysis(
        project=project,
        layers=range(4),
        attributes=[6, 19, 20, 3, 17, 18, 9, 5, 15, 8, 1, 11],
        tokenizer_path=config.tokenizer_path
    )

    # layers = range(4)
    # attributes = [6, 19, 20, 3, 17, 18, 9, 5, 15, 8, 1, 11]
    # project = "attr_from_answer"
    # save_matrix_path = f"{PATH_PREFIX}/{project}/similarity_matrix.npy"
    # save_fig_path = f"COMPLETE_FIGS/{project}/similarity_heatmap.png"

    # sim_matrix = compute_similarity_matrix(layers, attributes, project, save_matrix_path=save_matrix_path)
    # create_cosine_similarity_heatmap(layers, attributes, config.tokenizer_path, sim_matrix, project, save_fig_path)

    # tokenizer = load_tokenizer(config.tokenizer_path)
    # for attribute_id in [1, 3, 5, 6, 8, 9, 11, 15, 17, 18, 19, 20]:
    #     for capture_layer in range(4):
    #         binary_dataset_path = f"{PATH_PREFIX}/attr_from_answer/layer{capture_layer}/binary_dataset_{attribute_id}.pt"
    #         if os.path.exists(binary_dataset_path):
    #             data = torch.load(binary_dataset_path)
    #             binary_targets = data['binary_targets']
    #             positive_samples = torch.sum(binary_targets).item()
    #             total_samples = len(binary_targets)
    #             negative_samples = total_samples - positive_samples
    #             positive_percentage = (positive_samples / total_samples) * 100
    #             negative_percentage = (negative_samples / total_samples) * 100

    #             print(f"Layer {capture_layer}, Attribute {tokenizer.id_to_token[attribute_id]}, {attribute_id}, :")
    #             print(f"Positive samples: {positive_samples} ({positive_percentage:.2f}%)")
    #             print(f"Negative samples: {negative_samples} ({negative_percentage:.2f}%)")
    #         else:
    #             print(f"Dataset for layer {capture_layer}, attribute {attribute_id} not found.")
                
    # dataset = torch.load(config.dataset_path)
    # train_loader, val_loader = initialize_loaders(config, dataset)
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # GPT_model = GPT(config).to(device)
    # checkpoint = torch.load(config.filename, weights_only=False)
    # GPT_model.load_state_dict(checkpoint["model"])
    # GPT_model.eval()

    # project = "attr_from_answer"
    # plot_all_layers_metrics(
    #     layers=[0, 1, 2, 3],
    #     tokenizer_path=config.tokenizer_path,
    #     project_name=project,
    #     loss_range=[0, 0.63],
    #     acc_range=[0.65, 1]
    #     )
    
    # for attribute_id in [1, 3, 5, 6, 8, 9, 11, 15, 17, 18, 19, 20]:
    #     for capture_layer in range(4):
    # attribute_id = 1
    # capture_layer = 0
    # print(
    #     f"binary probe for attribute {attribute_id}, layer {capture_layer}")

    # init_attr_from_answer(config, capture_layer, val_loader, GPT_model)
    # init_binary_probe_data(attribute_id, capture_layer)

    # train_binary_probe(
    #     capture_layer=capture_layer,
    #     attribute_id=attribute_id,
    #     project="attr_from_answer",
    #     model_save_path=f"{PATH_PREFIX}/attr_from_answer/layer{capture_layer}/binary_probe_model.pt",
    #     patience=5,
    #     num_epochs=5
    # )

    # for attribute_id in [1, 3, 5, 6, 8, 9, 11, 15, 17, 18, 19, 20]:
    # for attribute_id in [18, 19, 20]:
    #     capture_layer = 0
    #     print(
    #         f"binary probe for attribute {attribute_id}, layer {capture_layer}")

    #     init_attr_from_answer(config, capture_layer, val_loader, GPT_model)
    #     init_binary_probe_data(attribute_id, capture_layer)
    #     init_binary_dataset(attribute_id, capture_layer, project)

    #     train_binary_probe(
    #         capture_layer=capture_layer,
    #         attribute_id=attribute_id,
    #         project=project,
    #         patience=5
    #     )

    # for attribute_id in [1, 3, 5, 6, 8, 9, 11, 15, 17, 18, 19, 20]:
    #     capture_layer = 1
    #     print(
    #         f"binary probe for attribute {attribute_id}, layer {capture_layer}")

    #     init_attr_from_answer(config, capture_layer, val_loader, GPT_model)
    #     init_binary_probe_data(attribute_id, capture_layer)
    #     init_binary_dataset(attribute_id, capture_layer, project)

    #     train_binary_probe(
    #         capture_layer=capture_layer,
    #         attribute_id=attribute_id,
    #         project=project,
    #         patience=5
    #     )

    # for attribute_id in [1, 3, 5, 6, 8, 9, 11, 15, 17, 18, 19, 20]:
    #     capture_layer = 2
    #     print(
    #         f"binary probe for attribute {attribute_id}, layer {capture_layer}")

    #     init_attr_from_answer(config, capture_layer, val_loader, GPT_model)
    #     init_binary_probe_data(attribute_id, capture_layer)
    #     init_binary_dataset(attribute_id, capture_layer, project)

    #     train_binary_probe(
    #         capture_layer=capture_layer,
    #         attribute_id=attribute_id,
    #         project=project,
    #         patience=5
    #     )

    # for attribute_id in [1, 3, 5, 6, 8, 9, 11, 15, 17, 18, 19, 20]:
    #     capture_layer = 3
    #     print(
    #         f"binary probe for attribute {attribute_id}, layer {capture_layer}")

    #     init_attr_from_answer(config, capture_layer, val_loader, GPT_model)
    #     init_binary_probe_data(attribute_id, capture_layer)
    #     init_binary_dataset(attribute_id, capture_layer, project)

    #     train_binary_probe(
    #         capture_layer=capture_layer,
    #         attribute_id=attribute_id,
    #         project=project,
    #         patience=5
    #     )
