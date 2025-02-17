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

    A_id = tokenizer.token_to_id["A"]
    B_id = tokenizer.token_to_id["B"]
    C_id = tokenizer.token_to_id["C"]
    D_id = tokenizer.token_to_id["D"]
    E_id = tokenizer.token_to_id["E"]


    all_input_embeddings = []
    all_target_attributes = []

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

            last_attr_embeddings = {
                A_id: None,
                B_id: None,
                C_id: None,
                D_id: None,
                E_id: None
            }
            
            sequence = sequence.tolist()
            for card_index, card_id in enumerate(sequence[0:(config.input_size-1):2]):
                # print(f"Card {card_id}, index {card_index}")
                attr_index = card_index * 2 + 1
                attr_id = sequence[attr_index]
                seen_card_dict[card_id].append(attr_id)

                if len(seen_card_dict[card_id]) == 4:
                    # print(f"Card {card_id}, {tokenizer.id_to_token[card_id]}, seen attributes {seen_card_dict[card_id]}")
                    last_attr_embeddings[card_id] = captured_embedding[seq_index, attr_index, :]
                    all_input_embeddings.append(last_attr_embeddings[card_id])
                    all_target_attributes.append(seen_card_dict[card_id])
            

    # After the loop completes, convert lists to tensors
    input_embeddings_tensor = torch.stack(all_input_embeddings)  # This will create a tensor of shape [num_samples, embedding_dim]
    target_attributes_tensor = torch.tensor(all_target_attributes)  # This will create a tensor of shape [num_samples, 4]

    # Save the tensors
    save_path_dir = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}"
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)

    torch.save({
        'input_embeddings': input_embeddings_tensor,
        'target_attributes': target_attributes_tensor
    }, f"{save_path_dir}/embeddings_and_attributes.pt")

    return input_embeddings_tensor, target_attributes_tensor


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import wandb
import numpy as np

class SimpleProbe(nn.Module):
    def __init__(self, embedding_dim, num_classes=12, sequence_length=4):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes * sequence_length)
        self.sequence_length = sequence_length
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.linear(x).reshape(-1, self.sequence_length, self.num_classes)

def compute_accuracy(outputs, targets):
    """Compute sequence-level accuracy (all positions must match)"""
    predictions = outputs.argmax(dim=-1)
    correct = (predictions == targets).all(dim=1).float().mean()
    return correct.item()

def train_probe(model, embeddings, target_sequences, num_epochs=100, batch_size=32, val_split=0.2):
    """
    Train probe with validation and W&B logging
    
    Args:
        model: SimpleProbe model
        embeddings: tensor of shape (num_samples, embedding_dim)
        target_sequences: tensor of shape (num_samples, sequence_length)
        num_epochs: number of training epochs
        batch_size: batch size for training
        val_split: fraction of data to use for validation
    """
    # Initialize W&B
    wandb.init(
        project="sequence-probe",
        config={
            "embedding_dim": embeddings.shape[1],
            "sequence_length": model.sequence_length,
            "num_classes": model.num_classes,
            "batch_size": batch_size,
            "val_split": val_split,
            "num_epochs": num_epochs
        }
    )
    
    # Split data into train and validation
    num_val = int(len(embeddings) * val_split)
    num_train = len(embeddings) - num_val
    
    train_dataset, val_dataset = random_split(
        TensorDataset(embeddings, target_sequences),
        [num_train, num_val]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_acc = 0
        
        for batch_embeddings, batch_targets in train_loader:
            outputs = model(batch_embeddings)
            
            loss = criterion(
                outputs.reshape(-1, model.num_classes),
                batch_targets.reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += compute_accuracy(outputs, batch_targets)
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        
        with torch.no_grad():
            for batch_embeddings, batch_targets in val_loader:
                outputs = model(batch_embeddings)
                
                loss = criterion(
                    outputs.reshape(-1, model.num_classes),
                    batch_targets.reshape(-1)
                )
                
                val_loss += loss.item()
                val_acc += compute_accuracy(outputs, batch_targets)
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n')
    
    wandb.finish()
    

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    config = GPTConfig44_Complete()
    capture_layer = 0
    # init_all_attr_from_last_atrr_binding_dataset(
    #     config=config, 
    #     capture_layer=0)

    save_path_dir = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved_data = torch.load(f'{save_path_dir}/embeddings_and_attributes.pt')
    loaded_embeddings = saved_data['input_embeddings'].to(device)
    loaded_targets = saved_data['target_attributes'].to(device)
    
    model = SimpleProbe(config.n_embd).to(device)
    
    # Train with validation and logging
    train_probe(model, loaded_embeddings, loaded_targets)
    
    # # Test example
    # with torch.no_grad():
    #     test_embedding = torch.randn(1, config)
    #     output = model(test_embedding)
    #     predicted_sequence = output.argmax(dim=-1)
    #     print(f"Predicted sequence: {predicted_sequence[0]}")
