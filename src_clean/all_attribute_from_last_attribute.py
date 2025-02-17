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
import itertools

class SimpleProbe(nn.Module):
    def __init__(self, embedding_dim, num_classes=12, sequence_length=4):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes * sequence_length)
        self.sequence_length = sequence_length
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.linear(x).reshape(-1, self.sequence_length, self.num_classes)
    
class PermutationInvariantProbe(nn.Module):
    def __init__(self, embedding_dim, num_classes=12, sequence_length=4):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes * sequence_length)
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # Pre-compute all possible permutations
        self.permutations = list(itertools.permutations(range(sequence_length)))
    
    def forward(self, x):
        # x shape: (batch_size, embedding_dim)
        # output shape: (batch_size, sequence_length, num_classes)
        return self.linear(x).reshape(-1, self.sequence_length, self.num_classes)
    
    def compute_loss(self, outputs, targets):
        # outputs: (batch_size, sequence_length, num_classes)
        # targets: (batch_size, sequence_length)
        
        losses = []
        batch_size = outputs.size(0)
        
        # For each possible permutation of the target sequence
        for perm in self.permutations:
            # Permute the target sequence
            perm_targets = targets[:, perm]
            # Compute cross entropy loss for this permutation
            loss = nn.functional.cross_entropy(
                outputs.reshape(-1, self.num_classes),
                perm_targets.reshape(-1),
                reduction='none'
            ).view(batch_size, -1).sum(dim=1)
            losses.append(loss)
        
        # Stack all permutation losses and take minimum for each batch item
        # This means we consider the prediction correct if it matches ANY permutation
        stacked_losses = torch.stack(losses, dim=0)
        min_losses = stacked_losses.min(dim=0).values
        
        return min_losses.mean()

# def compute_accuracy(outputs, targets):
#     """Compute sequence-level accuracy (all positions must match)"""
#     predictions = outputs.argmax(dim=-1)
#     correct = (predictions == targets).all(dim=1).float().mean()
#     return correct.item()

def compute_accuracies(outputs, targets):
    """
    Compute sequence-level accuracy with different metrics:
    1. Strict sequence accuracy (exact position match)
    2. Position-wise accuracy (average across positions)
    3. Position-agnostic accuracy (correct elements in any order)
    """
    predictions = outputs.argmax(dim=-1)
    batch_size = predictions.shape[0]
    
    # 1. Strict sequence accuracy (all positions must match exactly)
    sequence_acc = (predictions == targets).all(dim=1).float().mean().item()
    
    # 2. Position-wise accuracy (average across positions)
    position_acc = (predictions == targets).float().mean().item()
    
    # 3. Position-agnostic accuracy
    agnostic_acc = 0
    for i in range(batch_size):
        pred_set = set(predictions[i].cpu().tolist())
        target_set = set(targets[i].cpu().tolist())
        num_correct = len(pred_set.intersection(target_set))
        agnostic_acc += num_correct / len(target_set)
    agnostic_acc = agnostic_acc / batch_size
    
    return sequence_acc, position_acc, agnostic_acc

def train_probe(model, embeddings, target_sequences, model_type, capture_layer, num_epochs=100, batch_size=32, val_split=0.2, early_stopping_patience=10):
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
            "num_epochs": num_epochs,
            "early_stopping_patience": early_stopping_patience,  # Number of epochs to wait before early stopping
        },
        name=f"{model_type}_probe_layer{capture_layer}"
    )
    
    # Initialize best validation loss and early stopping variables
    best_val_loss = float('inf')
    model_save_path = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}/{model_type}_model.pt"
    patience = wandb.config.early_stopping_patience
    patience_counter = 0
    
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
        train_seq_acc = 0
        train_pos_acc = 0
        train_agn_acc = 0
        
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
            seq_acc, pos_acc, agn_acc = compute_accuracies(outputs, batch_targets)
            train_seq_acc += seq_acc
            train_pos_acc += pos_acc
            train_agn_acc += agn_acc
        
        train_loss /= len(train_loader)
        train_seq_acc /= len(train_loader)
        train_pos_acc /= len(train_loader)
        train_agn_acc /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_seq_acc = 0
        val_pos_acc = 0
        val_agn_acc = 0
        
        with torch.no_grad():
            for batch_embeddings, batch_targets in val_loader:
                outputs = model(batch_embeddings)
                
                loss = criterion(
                    outputs.reshape(-1, model.num_classes),
                    batch_targets.reshape(-1)
                )
                
                val_loss += loss.item()
                seq_acc, pos_acc, agn_acc = compute_accuracies(outputs, batch_targets)
                val_seq_acc += seq_acc
                val_pos_acc += pos_acc
                val_agn_acc += agn_acc
        
        val_loss /= len(val_loader)
        val_seq_acc /= len(val_loader)
        val_pos_acc /= len(val_loader)
        val_agn_acc /= len(val_loader)
        
        # Save model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_sequence_acc': val_seq_acc,
                'val_position_acc': val_pos_acc,
                'val_agnostic_acc': val_agn_acc
            }, model_save_path)
            wandb.save(model_save_path)  # Save to W&B as well
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs!')
                print(f'Best validation loss: {best_val_loss:.4f}')
                break
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_sequence_acc": train_seq_acc,
            "train_position_acc": train_pos_acc,
            "train_agnostic_acc": train_agn_acc,
            "val_loss": val_loss,
            "val_sequence_acc": val_seq_acc,
            "val_position_acc": val_pos_acc,
            "val_agnostic_acc": val_agn_acc
        })
        
        if (epoch) % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Train Metrics:')
            print(f'  Sequence Acc: {train_seq_acc:.4f}')
            print(f'  Position Acc: {train_pos_acc:.4f}')
            print(f'  Agnostic Acc: {train_agn_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Val Metrics:')
            print(f'  Sequence Acc: {val_seq_acc:.4f}')
            print(f'  Position Acc: {val_pos_acc:.4f}')
            print(f'  Agnostic Acc: {val_agn_acc:.4f}\n')
    
if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    config = GPTConfig44_Complete()
    # capture_layer = 0
    # init_all_attr_from_last_atrr_binding_dataset(
    #     config=config, 
    #     capture_layer=0)

    # for capture_layer in [1, 2, 3]:
    #     print(f"Capture layer {capture_layer}")
    #     init_all_attr_from_last_atrr_binding_dataset(
    #         config=config, 
    #         capture_layer=capture_layer)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for capture_layer in range(4):
        save_path_dir = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}"

        saved_data = torch.load(f'{save_path_dir}/embeddings_and_attributes.pt')
        loaded_embeddings = saved_data['input_embeddings'].to(device)
        loaded_targets = saved_data['target_attributes'].to(device)
        unique_values, _ = torch.unique(loaded_targets, return_inverse=True)
        continuous_targets = torch.searchsorted(unique_values, loaded_targets)

        permutation_invariant_model = PermutationInvariantProbe(config.n_embd).to(device)
        train_probe(
            model=permutation_invariant_model, 
            embeddings=loaded_embeddings, 
            target_sequences=continuous_targets,
            model_type="permutation_invariant",
            capture_layer=capture_layer,
            num_epochs=1)

    # save_path_dir = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}"

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # saved_data = torch.load(f'{save_path_dir}/embeddings_and_attributes.pt')
    # loaded_embeddings = saved_data['input_embeddings'].to(device)
    # loaded_targets = saved_data['target_attributes'].to(device)
    # unique_values, _ = torch.unique(loaded_targets, return_inverse=True)
    # continuous_targets = torch.searchsorted(unique_values, loaded_targets)
    
    # # model = SimpleProbe(config.n_embd).to(device)
    # # train_probe(model, loaded_embeddings, continuous_targets)

    # permutation_invariant_model = PermutationInvariantProbe(config.n_embd).to(device)
    # train_probe(permutation_invariant_model, loaded_embeddings, continuous_targets)
    
    # # Test example
    # with torch.no_grad():
    #     test_embedding = torch.randn(1, config)
    #     output = model(test_embedding)
    #     predicted_sequence = output.argmax(dim=-1)
    #     print(f"Predicted sequence: {predicted_sequence[0]}")
