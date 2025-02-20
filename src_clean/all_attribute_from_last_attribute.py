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


def construct_binary_dataset(attribute_id, capture_layer):
    load_existing_dataset_path = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}/embeddings_and_attributes.pt"
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
    }, f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}/binary_dataset_{attribute_id}.pt")

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
    
class SortedProbe(nn.Module):
    def __init__(self, embedding_dim, num_classes=12, sequence_length=4):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes * sequence_length)
        self.sequence_length = sequence_length
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.linear(x).reshape(-1, self.sequence_length, self.num_classes)
    
    def compute_loss(self, outputs, targets):
        # Sort the target sequences - this gives us a consistent ordering
        sorted_targets, _ = torch.sort(targets, dim=-1)
        
        # Regular cross entropy on the sorted targets
        return nn.functional.cross_entropy(
            outputs.reshape(-1, self.num_classes),
            sorted_targets.reshape(-1)
        )

class BinaryProbe(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, embedding_dim)
        logits = self.linear(x)
        probs = self.sigmoid(logits)
        return probs
    
    def predict(self, x, threshold=0.5):
        # Returns binary predictions (0 or 1)
        probs = self.forward(x)
        return (probs >= threshold).float()
    
    def compute_loss(self, outputs, targets):
        # Binary cross entropy loss
        return nn.functional.binary_cross_entropy(outputs, targets)

def init_binary_dataset(attribute_id, capture_layer, val_split=0.2, batch_size=32):
    # Load the binary dataset
    dataset_path = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}/binary_dataset_{attribute_id}.pt"
    data = torch.load(dataset_path)
    
    input_embeddings = data['input_embeddings']
    binary_targets = data['binary_targets']
    
    # Create the full dataset
    dataset = TensorDataset(input_embeddings, binary_targets)
    
    # Calculate split sizes
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    dataset_save_path = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}/attr_{attribute_id}/binary_dataloader.pt"
    if not os.path.exists(os.path.dirname(dataset_save_path)):
        os.makedirs(os.path.dirname(dataset_save_path))
    # Save val and train laoder
    torch.save({
        'train_loader': train_loader,
        'val_loader': val_loader
    }, dataset_save_path)

def load_binary_dataloader(attribute_id, capture_layer):
    dataset_save_path = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}/attr_{attribute_id}/binary_dataloader.pt"
    saved_data = torch.load(dataset_save_path)
    return saved_data['train_loader'], saved_data['val_loader']

def train_binary_probe(
    capture_layer,
    attribute_id,
    embedding_dim=64,
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=100,
    patience=10,
    val_split=0.2,
    device='cuda' if torch.cuda.is_available() else 'cpu'
): 
    # Load the binary dataset
    train_loader, val_loader = load_binary_dataloader(attribute_id, capture_layer)

    # Initialize model, optimizer, and move to device
    model = BinaryProbe(embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize wandb
    wandb.init(
        project="binary-probe-training-all-attr",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "embedding_dim": embedding_dim,
            "capture_layer": capture_layer,
            "attribute_id": attribute_id,
            "val_split": val_split
        },
        name=f"attr_{attribute_id}_layer{capture_layer}"
    )
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_embeddings, batch_targets in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            outputs = model(batch_embeddings).squeeze()
            loss = model.compute_loss(outputs, batch_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            predictions = model.predict(batch_embeddings).squeeze()
            train_correct += (predictions == batch_targets).sum().item()
            train_total += batch_targets.size(0)
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_embeddings, batch_targets in val_loader:
                batch_embeddings = batch_embeddings.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_embeddings).squeeze()
                loss = model.compute_loss(outputs, batch_targets)
                
                predictions = model.predict(batch_embeddings).squeeze()
                val_correct += (predictions == batch_targets).sum().item()
                val_total += batch_targets.size(0)
                val_loss += loss.item()
        
        # Calculate average losses and accuracies
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        })
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save the best model
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), 
               f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}/attr_{attribute_id}/binary_probe_model.pt")
    wandb.finish()
    return model

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
            
            if model_type == "simple":
                loss = criterion(
                    outputs.reshape(-1, model.num_classes),
                    batch_targets.reshape(-1)
                )
            elif model_type == "sorted":
                loss = model.compute_loss(outputs, batch_targets)
            else:
                raise ValueError(f"Invalid model type: {model_type}")
            
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
                
                if model_type == "simple":
                    loss = criterion(
                        outputs.reshape(-1, model.num_classes),
                        batch_targets.reshape(-1)
                    )
                elif model_type == "sorted":
                    loss = model.compute_loss(outputs, batch_targets)
                else:
                    raise ValueError(f"Invalid model type: {model_type}")
                
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
    wandb.finish()


@dataclass
class SortedProbeConfig:
    model_type: str = "sorted"
    input_dim: int = 64
    num_classes: int = 12
    sequence_length: int = 4

def load_linear_probe_(config, capture_layer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe = SortedProbe(
        embedding_dim=config.input_dim, 
        num_classes=config.num_classes, 
        sequence_length=config.sequence_length).to(device)
    
    probe_path = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}/{config.model_type}_model.pt"

    probe.load_state_dict(torch.load(probe_path)["model_state_dict"])
    probe.eval()
    return probe

def predict_from_probe(config, capture_layer, batch_size=32):
    """
    Analyze predictions from a trained probe model using PyTorch tensors.
    
    Args:
        config: Configuration object with model parameters
        capture_layer: Layer number to analyze
        batch_size: Batch size for processing predictions
    
    Returns:
        dict: Dictionary containing prediction results and statistics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the probe model
    probe = load_linear_probe_(config, capture_layer)
    probe.eval()
    
    # Load the dataset
    dataset_path = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}"
    saved_data = torch.load(f'{dataset_path}/embeddings_and_attributes.pt')
    
    embeddings = saved_data['input_embeddings'].to(device)

    loaded_targets = saved_data['target_attributes'].to(device)
    unique_values, _ = torch.unique(loaded_targets, return_inverse=True)
    continuous_targets = torch.searchsorted(unique_values, loaded_targets)
    
    # Create DataLoader for batch processing
    dataset = TensorDataset(embeddings, continuous_targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_targets = []
    
    # Process batches
    with torch.no_grad():
        for index, (batch_embeddings, batch_targets) in enumerate(dataloader):
            print(f"Batch {index + 1}/{len(dataloader)}")
            outputs = probe(batch_embeddings)  # Shape: (batch_size, sequence_length, num_classes)
            outputs = outputs.reshape(-1, config.sequence_length, config.num_classes)  # Ensure shape is (batch_size, 4, 12)
            predictions = outputs.argmax(dim=-1)  # Shape: (batch_size, 4)
            
            # Store predictions and targets
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Compute specialized accuracies
    accuracy_stats = compute_position_and_token_accuracies(all_predictions, all_targets)

    return {
        'predictions': all_predictions,
        'target_attributes': all_targets,
        'stats': accuracy_stats
    }


def compute_position_and_token_accuracies(predictions, targets):
    print("Computing accuracies...")
    total_sequences, seq_length = predictions.shape
    print(f"Processing {total_sequences} total sequences with length {seq_length}")
    
    # Initialize position accuracies tensor
    position_accuracies = [0, 0, 0, 0]
    
    # Get unique tokens
    unique_tokens = [int(x) for x in np.unique(targets)]

    token_stats = {
        token: {
            'correct': 0,
            'total': 0
        }
        for token in unique_tokens
    }

    # Process each sequence using tensor operations
    for i in range(total_sequences):
        print(f"Sequence {i + 1}/{total_sequences}")
        pred_seq = predictions[i]
        target_seq = targets[i]

        # For each target position
        for target_pos, target_token in enumerate(target_seq):
            # If this target token appears anywhere in prediction,
            # increment the accuracy for this position
            token_stats[target_token]['total'] += 1
            if target_token in pred_seq:
                position_accuracies[target_pos] += 1
                token_stats[target_token]['correct'] += 1

        if (i + 1) % 10000 == 0:
            print(f"Current position_accuracies:")
            print(f"Total sequences: {i + 1}")
            for pos in range(4):
                print(f"Accuracy of pos {pos}: {position_accuracies[pos] / (i + 1)}")
            
            print(f"Current token_stats:")
            for token, stats in token_stats.items():
                print(f"Token {token}: {stats['correct']} / {stats['total']}")
    
    # Convert counts to accuracies
    for pos in range(4):
        position_accuracies[pos] = position_accuracies[pos] / total_sequences
    
    for token in unique_tokens:
        token_stats[token] = token_stats[token]['correct'] / token_stats[token]['total']
    
    print(f"Position accuracies: {position_accuracies}")
    print(f"Token accuracies: {token_stats}")

    return {
        'position_accuracies': position_accuracies,
        'token_accuracies': token_stats
    }

def convert_results_stats_to_readable_form(results_path, tokenizer_path):
    tokenizer = load_tokenizer(tokenizer_path)

    # Load results stats from pkl
    with open(results_path, 'rb') as f:
        results_stats = pickle.load(f)
    mapping_values = [ 1,  3,  5,  6,  8,  9, 11, 15, 17, 18, 19, 20]

    token_accuracies = results_stats['token_accuracies']

    mapped_token_accuracies = {}

    for token_id, token_acc in token_accuracies.items():
        token = mapping_values[token_id]
        mapped_token_accuracies[tokenizer.id_to_token[token]] = token_acc

    print(mapped_token_accuracies)
    return mapped_token_accuracies


def analyze_probe_weights(probe_config, capture_layer):
    """
    Analyze the weights of a SortedProbe model
    Args:
        probe_model: trained SortedProbe model
    """
    # Get the weights from the linear layer

    probe_model = load_linear_probe_(probe_config, capture_layer)

    weights = probe_model.linear.weight.data  # Shape: (48, embedding_dim)
    bias = probe_model.linear.bias.data      # Shape: (48,)


    # Reshape weights to match the model's output structure
    # Original output shape: (-1, 4, 12)
    # So weights should be viewed as (4, 12, embedding_dim)
    reshaped_weights = weights.view(4, 12, -1)
    reshaped_bias = bias.view(4, 12)
    

    # Compute statistics for each position and class
    weight_stats = {
        'mean_per_position': torch.mean(reshaped_weights, dim=(1, 2)),  # Average across classes and embedding
        'std_per_position': torch.std(reshaped_weights, dim=(1, 2)),   # Std across classes and embedding
        'mean_per_class': torch.mean(reshaped_weights, dim=(0, 2)),    # Average across positions and embedding
        'l2_norm_per_position': torch.norm(reshaped_weights, dim=2),    # L2 norm for each position-class pair
        'weight_magnitudes': torch.abs(reshaped_weights),               # Absolute values of weights
    }
    
    # Analyze position-wise patterns
    position_correlations = torch.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            # Flatten weights for each position and compute correlation
            pos_i = reshaped_weights[i].flatten()
            pos_j = reshaped_weights[j].flatten()
            correlation = torch.corrcoef(torch.stack([pos_i, pos_j]))[0, 1]
            position_correlations[i, j] = correlation
            
    # Analyze class-wise patterns
    class_correlations = torch.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            # Flatten weights for each class and compute correlation
            class_i = reshaped_weights[:, i, :].flatten()
            class_j = reshaped_weights[:, j, :].flatten()
            correlation = torch.corrcoef(torch.stack([class_i, class_j]))[0, 1]
            class_correlations[i, j] = correlation
    
    return {
        'weight_stats': weight_stats,
        'position_correlations': position_correlations,
        'class_correlations': class_correlations,
        'reshaped_weights': reshaped_weights,
        'reshaped_bias': reshaped_bias
    }


def new_analyze_probe_weights(probe_config, capture_layer):
    probe_model = load_linear_probe_(probe_config, capture_layer)

    weights = probe_model.linear.weight.data  # Shape: (48, embedding_dim)
    bias = probe_model.linear.bias.data      # Shape: (48,)


    # Reshape weights to match the model's output structure
    # Original output shape: (-1, 4, 12)
    # So weights should be viewed as (4, 12, embedding_dim)
    reshaped_weights = weights.view(4, 12, -1)
    reshaped_bias = bias.view(4, 12)

    # Analyze class-wise patterns
    class_cosine_sim = torch.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            # Flatten weights for each class and compute correlation
            class_i = reshaped_weights[:, i, :].flatten()
            class_j = reshaped_weights[:, j, :].flatten()
            cosine_sim = F.cosine_similarity(class_i, class_j, dim=0)
            class_cosine_sim[i, j] = cosine_sim
    
    # Compute pairwise cosine similarity for all weights (48, 48)
    all_weight_cosine_sim = torch.zeros((48, 48))
    for i in range(48):
        for j in range(48):
            cosine_sim = F.cosine_similarity(weights[i:i+1], weights[j:j+1], dim=1)
            all_weight_cosine_sim[i, j] = cosine_sim
    breakpoint()

    return {
        'class_cosine_sim': class_cosine_sim,
        'all_weight_cosine_sim': all_weight_cosine_sim
    }

def plot_new_weight_analysis(analysis_results, tokenizer_path, save_path):
    """
    Create visualizations for the new weight analysis with cosine similarities
    Args:
        analysis_results: dictionary containing the new analysis results
        tokenizer_path: path to the tokenizer
        save_path: path to save the figure
    """
    plt.figure(figsize=(15, 6))
    
    # 1. Class-wise cosine similarity heatmap (reordered)
    plt.subplot(1, 2, 1)
    reordered_labels, reordered_correlations = reorder_results(
        analysis_results['class_cosine_sim'], 
        tokenizer_path
    )
    sns.heatmap(
        reordered_correlations.numpy(),
        cmap='RdBu', 
        center=0,
        xticklabels=reordered_labels,
        yticklabels=reordered_labels
    )
    plt.title('Class-wise Cosine Similarities')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 2. All weights cosine similarity heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(
        analysis_results['all_weight_cosine_sim'].numpy(),
        cmap='RdBu', 
        center=0,
        xticklabels=range(48),
        yticklabels=range(48)
    )
    plt.title('All Weights Cosine Similarities')
    
    plt.tight_layout()
    
    # Save the figure
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, bbox_inches="tight")
    return plt.gcf()


def plot_weight_analysis(analysis_results, tokenizer_path):
    """
    Create visualizations for the weight analysis
    Args:
        analysis_results: dictionary containing the analysis results
    """
    tokenizer = load_tokenizer(tokenizer_path)
    plt.figure(figsize=(15, 12))
    
    # 1. Position correlations heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(analysis_results['position_correlations'].numpy(),
                cmap='RdBu', center=0,
                xticklabels=range(4),
                yticklabels=range(4))
    plt.title('Position-wise Weight Correlations')
    
    # 2. Class correlations heatmap
    reordered_labels, reordered_correlations = reorder_results(analysis_results['class_correlations'], tokenizer_path)
    plt.subplot(2, 2, 2)
    sns.heatmap(reordered_correlations.numpy(),
                cmap='RdBu', center=0,
                xticklabels=reordered_labels,
                yticklabels=reordered_labels)
    
    # sns.heatmap(analysis_results['class_correlations'].numpy(),
    #             cmap='RdBu', center=0,
    #             xticklabels=tokenizer.decode([ 1,  3,  5,  6,  8,  9, 11, 15, 17, 18, 19, 20]),
    #             yticklabels=tokenizer.decode([ 1,  3,  5,  6,  8,  9, 11, 15, 17, 18, 19, 20]))
    plt.title('Class-wise Weight Correlations')
    
    # 3. Weight magnitude distribution per position
    plt.subplot(2, 2, 3)
    weight_mags = analysis_results['weight_stats']['weight_magnitudes']
    for pos in range(4):
        plt.hist(weight_mags[pos].flatten().cpu().numpy(), 
                alpha=0.5, 
                label=f'Position {pos}',
                bins=30)
    plt.legend()
    plt.title('Weight Magnitude Distribution by Position')
    plt.xlabel('Magnitude')
    plt.ylabel('Count')
    
    # 4. L2 norms heatmap
    plt.subplot(2, 2, 4)
    sns.heatmap(analysis_results['weight_stats']['l2_norm_per_position'].cpu().numpy(),
                cmap='viridis',
                xticklabels=tokenizer.decode([ 1,  3,  5,  6,  8,  9, 11, 15, 17, 18, 19, 20]),
                yticklabels=range(4))
    plt.title('L2 Norm of Weights (Position vs Class)')
    
    plt.tight_layout()
    return plt.gcf()


def reorder_results(to_reorder, tokenizer_path):
    shapes = ["oval", "squiggle", "diamond"]
    colors = ["green", "blue", "pink"]
    numbers = ["one", "two", "three"]
    shadings = ["solid", "striped", "open"]
    tokenizer = load_tokenizer(tokenizer_path)
    # Token IDs to decode
    token_ids = [1, 3, 5, 6, 8, 9, 11, 15, 17, 18, 19, 20]
    
    # First, get the decoded labels
    decoded_labels = tokenizer.decode(token_ids)
    
    # Create the desired order
    desired_order = shapes + colors + numbers + shadings
    
    # Create mapping from current positions to desired positions
    current_to_desired = {label: i for i, label in enumerate(desired_order)}
    current_positions = {label: i for i, label in enumerate(decoded_labels)}
    
    # Create reordering indices
    reorder_indices = []
    for label in desired_order:
        if label in current_positions:
            reorder_indices.append(current_positions[label])
            
    # Convert to tensor for indexing
    reorder_indices = torch.tensor(reorder_indices)
    
    # Reorder the correlation matrix
    reordered = to_reorder[reorder_indices][:, reorder_indices]
    ordered_labels = desired_order

    return ordered_labels, reordered
    
def map_continuous_id_to_attr_name(continuous_id, tokenizer_path):
    tokenizer = load_tokenizer(tokenizer_path)
    original = [1, 3, 5, 6, 8, 9, 11, 15, 17, 18, 19, 20]
    target_value = original[continuous_id]
    return tokenizer.id_to_token[target_value]


import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

def plot_metrics_by_layer(target_layer, tokenizer_path, project_name="binary-probe-training-all-attr", entity="your-wandb-username"):
    """
    Create two separate plots: one for losses and one for accuracies, for runs from a specific layer.
    
    Args:
        target_layer (int): The layer number to filter runs for
        project_name (str): Name of the W&B project
        entity (str): Your W&B username
    """
    # Initialize wandb
    api = wandb.Api()
    
    # Get all runs from your project
    runs = api.runs(f"{entity}/{project_name}")
    
    # Filter runs for the specified layer
    # The pattern now looks for layer followed by digits at the end of the name
    layer_runs = []
    for run in runs:
        match = re.search(r'layer(\d+)$', run.name)
        if match and int(match.group(1)) == target_layer:
            layer_runs.append(run)
    
    if not layer_runs:
        print(f"No runs found for layer {target_layer}")
        return
    
    # Rest of the plotting code remains the same...
    plt.style.use('seaborn')
    
    # Create first figure for losses
    fig1, ax1 = plt.subplots(1, 2, figsize=(15, 5))
    fig1.suptitle(f'Training and Validation Losses for Layer {target_layer}', fontsize=14)
    
    # Create second figure for accuracies
    fig2, ax2 = plt.subplots(1, 2, figsize=(15, 5))
    fig2.suptitle(f'Training and Validation Accuracies for Layer {target_layer}', fontsize=14)
    
    # Color map for different attributes
    num_runs = len(layer_runs)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_runs))
    
    tokenizer = load_tokenizer(tokenizer_path)
    for run, color in zip(layer_runs, colors):
        # Extract attribute ID from run name
        attr_match = re.search(r'attr_(\d+)_', run.name)
        attr_id = attr_match.group(1) if attr_match else 'unknown'
        
        label = map_continuous_id_to_attr_name(int(attr_id), tokenizer_path)
        # Convert run history to pandas DataFrame
        history = pd.DataFrame(run.history())
        
        # Plot training loss
        ax1[0].plot(history['step'], history['train_loss'], 
                   label=f'{label}', color=color)
        ax1[0].set_title('Training Loss')
        ax1[0].set_xlabel('Epoch')
        ax1[0].set_ylabel('Loss')
        
        # Plot validation loss
        ax1[1].plot(history['step'], history['val_loss'], 
                   label=f'{label}', color=color)
        ax1[1].set_title('Validation Loss')
        ax1[1].set_xlabel('Epoch')
        ax1[1].set_ylabel('Loss')
        
        # Plot training accuracy
        ax2[0].plot(history['step'], history['train_accuracy'], 
                   label=f'{label}', color=color)
        ax2[0].set_title('Training Accuracy')
        ax2[0].set_xlabel('Epoch')
        ax2[0].set_ylabel('Accuracy')
        
        # Plot validation accuracy
        ax2[1].plot(history['step'], history['val_accuracy'], 
                   label=f'{label}', color=color)
        ax2[1].set_title('Validation Accuracy')
        ax2[1].set_xlabel('Epoch')
        ax2[1].set_ylabel('Accuracy')
    
    # Add legends
    for ax in ax1:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    for ax in ax2:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layouts
    fig1.tight_layout()
    fig2.tight_layout()
    
    save_path = f"COMPLETE_FIGS/attr_from_last_attr_binding/layer{target_layer}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Save figures
    fig1.savefig(f'{save_path}/losses.png', dpi=300, bbox_inches='tight')
    fig2.savefig(f'{save_path}/accuracies.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    config = GPTConfig44_Complete()
    # plot_metrics_by_layer(
    #     target_layer=2,
    #     tokenizer_path=config.tokenizer_path,
    #     entity="hazhou",
    # )
    # capture_layer = 0
    # init_all_attr_from_last_atrr_binding_dataset(
    #     config=config, 
    #     capture_layer=0)

    # for capture_layer in [1, 2, 3]:
    #     print(f"Capture layer {capture_layer}")
    #     init_all_attr_from_last_atrr_binding_dataset(
    #         config=config, 
    #         capture_layer=capture_layer)

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # capture_layer = 2
    # for attribute_id in [1, 3, 5, 6, 8, 9, 11, 15, 17, 18, 19, 20]:
    #     print(f"Training binary probe for attribute {attribute_id}")
    #     train_binary_probe(
    #         capture_layer=capture_layer,
    #         attribute_id=attribute_id,
    #     )

    # for attribute_id in [6, 8, 9, 11, 15, 17, 18, 19, 20]:
    #     capture_layer = 2
    #     print(f"Training binary probe for attribute {attribute_id}, layer {capture_layer}")
    #     train_binary_probe(
    #         capture_layer=capture_layer,
    #         attribute_id=attribute_id,
    #         patience=5,
    #     )

    for attribute_id in [17, 18, 19, 20]:
        capture_layer = 0
        print(f"Training binary probe for attribute {attribute_id}, layer {capture_layer}")
        construct_binary_dataset(attribute_id, capture_layer)
        init_binary_dataset(attribute_id, capture_layer)
        train_binary_probe(
            capture_layer=capture_layer,
            attribute_id=attribute_id,
            patience=5,
        )

    # for attribute_id in [3, 5, 6, 8, 9, 11, 15, 17, 18, 19, 20]:
    #     capture_layer = 1
    #     print(f"Training binary probe for attribute {attribute_id}, layer {capture_layer}")
    #     construct_binary_dataset(attribute_id, capture_layer)
    #     init_binary_dataset(attribute_id, capture_layer)
    #     train_binary_probe(
    #         capture_layer=capture_layer,
    #         attribute_id=attribute_id,
    #         patience=5,
    #     )

    # for attribute_id in [3, 5, 6, 8, 9, 11, 15, 17, 18, 19, 20]:
    #     capture_layer = 3
    #     print(f"Training binary probe for attribute {attribute_id}, layer {capture_layer}")
    #     construct_binary_dataset(attribute_id, capture_layer)
    #     init_binary_dataset(attribute_id, capture_layer)
    #     train_binary_probe(
    #         capture_layer=capture_layer,
    #         attribute_id=attribute_id,
    #         patience=5,
    #     )

    # for attribute_id in [1, 3, 5, 6, 8, 9, 11, 15, 17, 18, 19, 20]:
    #     for capture_layer in [0, 1, 3]:
    #         print(f"Training binary probe for attribute {attribute_id}, layer {capture_layer}")
    #         construct_binary_dataset(attribute_id, capture_layer)
    #         init_binary_dataset(attribute_id, capture_layer)
    #         train_binary_probe(
    #             capture_layer=capture_layer,
    #             attribute_id=attribute_id,
    #         )

    # for attribute_id in [1, 3, 5, 6, 8, 9, 11, 15, 17, 18, 19, 20]:
    #     print("Constructing binary dataset for attribute", attribute_id)
    #     # construct_binary_dataset(attribute_id, capture_layer)
    #     init_binary_dataset(attribute_id, capture_layer)

    # for attribute_id in [1, 3, 5, 6, 8, 9, 11, 15, 17, 18, 19, 20]:
    #     print("Constructing binary dataset for attribute", attribute_id)
    #     construct_binary_dataset(attribute_id, capture_layer)

    # analysis_results = analyze_probe_weights(probe_config=SortedProbeConfig(), capture_layer=capture_layer)
    # save_analysis_results = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}/weight_analysis.pkl"

    # new_save_analysis_results = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}/new_weight_analysis.pkl"
    # new_analysis_results = new_analyze_probe_weights(probe_config=SortedProbeConfig(), capture_layer=capture_layer)
    # plot_new_weight_analysis(new_analysis_results, config.tokenizer_path, f"COMPLETE_FIGS/all_attr_from_last_attr/layer{capture_layer}_weight_analysis_cosine_sim.png")

    # if not os.path.exists(os.path.dirname(save_analysis_results)):
    #     os.makedirs(os.path.dirname(save_analysis_results))

    # with open(save_analysis_results, 'wb') as f:
    #     pickle.dump(analysis_results, f)

    # with open(save_analysis_results, 'rb') as f:
    #     analysis_results = pickle.load(f)

    # with open(save_analysis_results, 'wb') as f:
    #     pickle.dump(analysis_results, f)

    # with open(new_save_analysis_results, 'rb') as f:
    #     new_analysis_results = pickle.load(f)

    # fig = plot_weight_analysis(analysis_results, tokenizer_path=config.tokenizer_path)
    # save_fig_path = f"COMPLETE_FIGS/all_attr_from_last_attr/layer{capture_layer}_weight_analysis_reordered.png"
    # if not os.path.exists(os.path.dirname(save_fig_path)):
    #     os.makedirs(os.path.dirname(save_fig_path))
    # fig.savefig(save_fig_path, bbox_inches="tight")
    
    # results = predict_from_probe(SortedProbeConfig(), capture_layer=capture_layer, batch_size=32)
    # Save results stats as pkl
    # results_path = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}/sorted_accuracy_stats.pkl"
    # convert_results_stats_to_readable_form(results_path, config.tokenizer_path)
    # if not os.path.exists(os.path.dirname(results_path)):
    #     os.makedirs(os.path.dirname(results_path))
    # with open(results_path, 'wb') as f:
    #     pickle.dump(results["stats"], f)

    

    # for capture_layer in [2,3]:
    #     save_path_dir = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}"

    #     saved_data = torch.load(f'{save_path_dir}/embeddings_and_attributes.pt')
    #     loaded_embeddings = saved_data['input_embeddings'].to(device)
    #     loaded_targets = saved_data['target_attributes'].to(device)
    #     unique_values, _ = torch.unique(loaded_targets, return_inverse=True)
    #     continuous_targets = torch.searchsorted(unique_values, loaded_targets)

    #     sorted_model = SortedProbe(config.n_embd).to(device)
    #     train_probe(
    #         model=sorted_model, 
    #         embeddings=loaded_embeddings, 
    #         target_sequences=continuous_targets,
    #         model_type="sorted",
    #         capture_layer=capture_layer,
    #         )
        

    # for capture_layer in [3]:
    #     save_path_dir = f"{PATH_PREFIX}/all_attr_from_last_attr_binding/layer{capture_layer}"

    #     saved_data = torch.load(f'{save_path_dir}/embeddings_and_attributes.pt')
    #     loaded_embeddings = saved_data['input_embeddings'].to(device)
    #     loaded_targets = saved_data['target_attributes'].to(device)
    #     unique_values, _ = torch.unique(loaded_targets, return_inverse=True)
    #     continuous_targets = torch.searchsorted(unique_values, loaded_targets)

    #     simple_model = SimpleProbe(config.n_embd).to(device)
    #     train_probe(
    #         model=simple_model, 
    #         embeddings=loaded_embeddings, 
    #         target_sequences=continuous_targets,
    #         model_type="simple",
    #         capture_layer=capture_layer,
    #         )

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
