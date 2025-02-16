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
from tokenizer import load_tokenizer

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

def init_all_attr_from_last_atrr_binding_dataset(config, capture_layer):
    dataset = torch.load(config.dataset_path)
    train_loader, val_loader = initialize_loaders(config, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT(config).to(device)
    checkpoint = torch.load(config.filename, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = load_tokenizer(config.tokenizer_path)

    all_flattened_input_embeddings = []
    all_flattened_target_tokens = []

    A_id = tokenizer.token_to_id["A"]
    B_id = tokenizer.token_to_id["B"]
    C_id = tokenizer.token_to_id["C"]
    D_id = tokenizer.token_to_id["D"]
    E_id = tokenizer.token_to_id["E"]

    seen_card_dict = {
        A_id: [],
        B_id: [],
        C_id: [],
        D_id: [],
        E_id: []
    }

    for batch_index, batch in enumerate(val_loader):
        print(f"Batch {batch_index + 1}/{len(val_loader)}")
        batch = batch.to(device)
        _, _, _, captured_embedding, _ = model(batch, True, capture_layer)
        breakpoint()

        # for seq_index, sequence in enumerate(batch):
        #     for token_index, token in sequence:


    #     input_start_index = 1
    #     target_start_index = 0

    #     # Get every other embedding in the input sequence starting from input_start_index, representing either all attribute or card embeddings
    #     # input_embeddings.shape, torch.Size([512, 20, 64]), [batch size, sequence length, embedding size]
    #     input_embeddings = captured_embedding[:, input_start_index:(
    #         config.input_size-1):2, :]
    #     # flattened_input_embeddings.shape, torch.Size([10240, 64]), [batch size * sequence length, embedding size]
    #     flattened_input_embeddings = input_embeddings.reshape(-1, 64)

    #     # Get every other token in the input starting from index target_start_index, representing either all the card tokens or attribute tokens
    #     # target_tokens.shape, torch.Size([512, 20])
    #     target_tokens = batch[:, target_start_index:(config.input_size - 1):2]
    #     # flattened_target_tokens.shape, torch.Size([10240])
    #     flattened_target_tokens = target_tokens.reshape(-1)

    #     # Append the flattened tensors to the respective lists
    #     all_flattened_input_embeddings.append(flattened_input_embeddings)
    #     all_flattened_target_tokens.append(flattened_target_tokens)

    # combined_input_embeddings = torch.cat(
    #     all_flattened_input_embeddings, dim=0)

    # combined_target_tokens = torch.cat(all_flattened_target_tokens, dim=0)
    # mapped_target_tokens, continuous_to_original = map_non_continuous_vals_to_continuous(
    #     combined_target_tokens)

    # # Create the directory structure if it doesn't exist
    # base_dir = f"{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}"
    # os.makedirs(base_dir, exist_ok=True)

    # input_embeddings_path = f"{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}/input_embeddings.pt"
    # mapped_target_tokens_path = f"{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}/continuous_target_tokens.pt"
    # continuous_to_original_path = f"{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}/continuous_to_original.pkl"

    # # Save the combined_input_embeddings tensor
    # torch.save(combined_input_embeddings, input_embeddings_path)

    # # Save the mapped_target_attributes tensor
    # torch.save(mapped_target_tokens, mapped_target_tokens_path)

    # with open(continuous_to_original_path, "wb") as f:
    #     pickle.dump(continuous_to_original, f)

    # return combined_input_embeddings, mapped_target_tokens, continuous_to_original


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    config = GPTConfig44_Complete()

    init_all_attr_from_last_atrr_binding_dataset(
        config=config, 
        capture_layer=0)
