import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_utils import initialize_loaders
from model import GPT
import os
from itertools import permutations
from data_utils import split_data
import subprocess
import random
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict
import torch
from tokenizer import load_tokenizer
import random
from model import GPTConfig44_Complete
from data_utils import initialize_loaders, pretty_print_input

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'


def get_gpu_memory():
    """Get the current gpu usage."""
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', 
         '--format=csv,nounits,noheader'
    ], encoding='utf-8')
    used, total = map(int, result.strip().split(','))
    return used, total


def construct_binding_id_dataset(config, capture_layer):

    perms = list(permutations(range(20), 2))

    dataset = torch.load(config.dataset_path)
    train_loader, val_loader = initialize_loaders(config, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = GPT(config).to(device)  # adjust path as needed
    checkpoint = torch.load(config.filename, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    X = []
    y = []
    save_threshold = 25000000  # Save after accumulating 50M samples

    base_dir = f"{PATH_PREFIX}/src_clean/binding_id/44_complete/layer{capture_layer}"
    os.makedirs(base_dir, exist_ok=True)

    for index, batch in enumerate(train_loader):
        used, total = get_gpu_memory()
        print(f"Batch {index}/{len(train_loader)}")
        print(f"GPU Memory: {used/1024:.2f}GB used out of {total/1024:.2f}GB total")
        print("Current samples: ", len(y))

        batch = batch.to(device)
        with torch.no_grad():  # Reduce memory usage during inference
            # _, _, _, captured_embedding = model(batch, True, capture_layer)
            _, _, _, captured_embedding, _ = model(batch, True, capture_layer)
        # torch.Size([64, 49, 64])
        # [batch_size, seq_len, embedding_dim]

        captured_embedding = captured_embedding.cpu()
        batch = batch.cpu()

        for index, indiv_embedding in enumerate(captured_embedding):
            curr_embedding = indiv_embedding[:40]
            curr_tokens = batch[index][:40]
            for (element1_index, element2_index) in perms:            
                element1 = curr_embedding[element1_index * 2 + 1]
                element2 = curr_embedding[element2_index * 2 + 1]

                token1 = curr_tokens[element1_index * 2]
                token2 = curr_tokens[element2_index * 2]

                X.append(torch.cat((element1, element2)))
                y.append(1 if token1 == token2 else 0)

                # breakpoint()
                # Save intermediate results when threshold is reached
                if len(y) >= save_threshold:
                    X_tensor = torch.stack(X)
                    y_tensor = torch.tensor(y)
                    
                    # Save current chunk
                    torch.save(X_tensor, os.path.join(base_dir, f"X.pt"))
                    torch.save(y_tensor, os.path.join(base_dir, f"y.pt"))
                    
                    # # Clear lists and increment counter
                    # X = []
                    # y = []

def train_binding_classifier(dataset_name, capture_layer, model_name, input_dim=128, num_epochs=100, batch_size=32, lr=0.001, patience=10):
    """Train a binary classifier for binding identification using randomly selected chunks."""
    wandb.init(
        project="binding-id-classifier", 
        config={"epochs": num_epochs, "batch_size": batch_size, "lr": lr, "patience": patience},
        name=model_name
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = f"{PATH_PREFIX}/binding_id/{dataset_name}/layer{capture_layer}"
    
    # Setup data splits
    num_chunks = torch.load(os.path.join(base_dir, "metadata.pt"))['num_chunks']
    split_info = select_chunks(num_chunks)
    torch.save(split_info, os.path.join(base_dir, f"{model_name}_split_info.pt"))
    
    # Load validation and test sets
    X_val, y_val = load_and_combine_chunks(base_dir, split_info['val_chunks'], device)
    X_test, y_test = load_and_combine_chunks(base_dir, split_info['test_chunks'], device)
    
    # Initialize model and training components
    model = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid()).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_chunks = random.sample(split_info['train_chunks'], len(split_info['train_chunks']))
        total_train_loss = 0
        total_samples = 0
        
        for index, chunk_idx in enumerate(train_chunks):
            print(f"Training chunk {index}/{len(train_chunks)} with id {chunk_idx}")
            chunk = {
                'X_path': os.path.join(base_dir, f"X_chunk_{chunk_idx}.pt"),
                'y_path': os.path.join(base_dir, f"y_chunk_{chunk_idx}.pt")
            }
            loss, samples = process_chunk(model, chunk, device, criterion, optimizer, batch_size)
            total_train_loss += loss
            total_samples += samples
        
        avg_train_loss = total_train_loss / total_samples
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val.float())
            val_acc = ((val_outputs > 0.5).float() == y_val.float()).float().mean()
        
        # Logging
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss.item(),
            'val_acc': val_acc.item()
        })
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'split_info': split_info,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, f'{PATH_PREFIX}/binding_id/{model_name}_best.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Final evaluation
    test_acc = get_binding_classifier_accuracy(X_test, y_test, f'{PATH_PREFIX}/binding_id/{model_name}_best.pt', input_dim)
    wandb.log({'test_accuracy': test_acc})
    wandb.finish()
    
    return model



from torch.utils.data import Dataset, DataLoader

class BalancedBindingDataset(Dataset):
    def __init__(self, class_0_data, class_1_data):
        """
        Args:
            class_0_data (Tensor): Data where y = 0.
            class_1_data (Tensor): Data where y = 1.
        """
        self.class_0_data = class_0_data
        self.class_1_data = class_1_data
        self.length = min(len(class_0_data), len(class_1_data)) * 2  # Ensure 50/50 split

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Alternate between class 0 and class 1 data.
        Args:
            idx (int): Index of the sample.
        Returns:
            A tuple (data, label) where label is 0 or 1.
        """
        if idx % 2 == 0:
            # Even index: Pick from class 0
            data = self.class_0_data[idx // 2]
            label = 0
        else:
            # Odd index: Pick from class 1
            data = self.class_1_data[idx // 2]
            label = 1
        return data, torch.tensor(label, dtype=torch.float32)


def initialize_binding_dataset(X, y, val_size, test_size, batch_size):
    # Combine X and y into a single dataset
    full_dataset = TensorDataset(X, y)

    # Split the combined dataset into training, validation, and test sets
    train_size = int((1-val_size-test_size) * len(full_dataset)) 
    val_size = int(val_size * len(full_dataset))  # 15% for validation
    test_size = len(full_dataset) - train_size - val_size  # Remainder for test

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Ensures reproducibility
    )

    # Separate training data into class 0 and class 1
    X_train, y_train = zip(*train_dataset)  # Unpack into X_train and y_train
    X_train = torch.stack(X_train)
    y_train = torch.stack(y_train)

    class_0_data_train = X_train[(y_train == 0).nonzero(as_tuple=True)[0]]
    class_1_data_train = X_train[(y_train == 1).nonzero(as_tuple=True)[0]]

    # Separate validation data into class 0 and class 1
    X_val, y_val = zip(*val_dataset)  # Unpack into X_val and y_val
    X_val = torch.stack(X_val)
    y_val = torch.stack(y_val)

    class_0_data_val = X_val[(y_val == 0).nonzero(as_tuple=True)[0]]
    class_1_data_val = X_val[(y_val == 1).nonzero(as_tuple=True)[0]]

    # Separate test data into class 0 and class 1
    X_test, y_test = zip(*test_dataset)  # Unpack into X_test and y_test
    X_test = torch.stack(X_test)
    y_test = torch.stack(y_test)

    class_0_data_test = X_test[(y_test == 0).nonzero(as_tuple=True)[0]]
    class_1_data_test = X_test[(y_test == 1).nonzero(as_tuple=True)[0]]


    # Create the balanced training dataset
    balanced_train_dataset = BalancedBindingDataset(class_0_data_train, class_1_data_train)
    balanced_val_dataset = BalancedBindingDataset(class_0_data_val, class_1_data_val)
    balanced_test_dataset = BalancedBindingDataset(class_0_data_test, class_1_data_test)

    # Create DataLoaders
    train_loader = DataLoader(balanced_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(balanced_val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(balanced_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_binding_classifier_single_chunk(
    dataset_name, 
    capture_layer, 
    chunk_id,
    model_name, 
    val_size=0.05,
    test_size=0.05,
    input_dim=128, 
    num_epochs=100, 
    batch_size=32, 
    lr=0.001, 
    patience=10
):
    """Train a binary classifier using a single chunk split into train/val/test sets."""
    wandb.init(
        project="binding-id-classifier", 
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience,
            "chunk_id": chunk_id,
            "val_size": val_size,
            "test_size": test_size
        },
        name=f"{model_name}_chunk{chunk_id}"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = f"{PATH_PREFIX}/binding_id/{dataset_name}/layer{capture_layer}"
    
    # Load the single chunk
    X = torch.load(os.path.join(base_dir, f"X_chunk_{chunk_id}.pt")).to(device)
    y = torch.load(os.path.join(base_dir, f"y_chunk_{chunk_id}.pt")).to(device)
    
    # X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, val_size, test_size)  
    
    # print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # print("Class distribution:")
    # print(f"Train - Class 0: {(y_train == 0).sum()/len(y_train):.3f}, Class 1: {(y_train == 1).sum()/len(y_train):.3f}")
    # print(f"Val   - Class 0: {(y_val == 0).sum()/len(y_val):.3f}, Class 1: {(y_val == 1).sum()/len(y_val):.3f}")
    # print(f"Test  - Class 0: {(y_test == 0).sum()/len(y_test):.3f}, Class 1: {(y_test == 1).sum()/len(y_test):.3f}")

#     # pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
#     pos_weight = torch.sqrt(torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()])).to(device)
# # This would give ~2.3 instead of 5.3
#     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Try below next
    # weights = torch.FloatTensor([1 if label == 0 else (y_train == 0).sum()/(y_train == 1).sum() for label in y_train])
    # sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, 
    #     batch_size=batch_size,
    #     sampler=sampler  # Use sampler instead of shuffle=True
    # )

    # Initialize model and training components
    # model = nn.Sequential(nn.Linear(input_dim, 1)).to(device)
    model = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid()).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    counter = 0
    
    # # Create DataLoader for training data
    # train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, 
    #     batch_size=batch_size,
    #     shuffle=True
    # )

    # # Separate data for class 0 and class 1
    # class_0_data = X_train[(y_train == 0).nonzero(as_tuple=True)[0]]
    # class_1_data = X_train[(y_train == 1).nonzero(as_tuple=True)[0]]

    # # Create the balanced dataset
    # balanced_dataset = BalancedBindingDataset(class_0_data, class_1_data)

    # # Create DataLoader for the balanced dataset
    # train_loader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)

    train_loader, val_loader, test_loader = initialize_binding_dataset(X, y, val_size, test_size, batch_size)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
       
        total_train_loss = 0
        total_samples = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * len(batch_X)
            total_samples += len(batch_X)
        avg_train_loss = total_train_loss / total_samples
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        avg_val_loss = 0

        with torch.no_grad():
            total_val_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val).squeeze()  # Get model predictions (probabilities)
                
                # Compute the loss
                val_loss = criterion(outputs, y_val.float())
                total_val_loss += val_loss.item() * len(X_val)
                
                # Convert outputs to binary predictions (0 or 1)
                predicted = (outputs >= 0.5).float()  # Apply 0.5 threshold
                
                # Count the number of correct predictions
                correct_predictions += (predicted == y_val).sum().item()
                total_predictions += len(y_val)
            
            # Calculate average validation loss
            avg_val_loss = total_val_loss / len(val_loader.dataset)
            
            # Calculate validation accuracy
            val_accuracy = correct_predictions / total_predictions * 100  # As percentage
            
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.2f}%")
        
        # Logging
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss.item(),
            'val_acc': val_accuracy
        })
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            model_path = f'{PATH_PREFIX}/binding_id/{model_name}_chunk{chunk_id}_best.pt'
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_accuracy,
                'chunk_id': chunk_id,
                'split_sizes': {
                    'val': val_size,
                    'test': test_size
                }
            }, model_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Final evaluation
    # test_acc = get_binding_classifier_accuracy(X_test, y_test, model_path, input_dim)
    # wandb.log({'test_accuracy': test_acc})
    wandb.finish()
    
    return model



def get_binding_classifier_accuracy(X, y, model_path, input_dim=128):
    """Test a trained binding classifier model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the same model architecture
    model = nn.Sequential(
        nn.Linear(input_dim, 1)
    ).to(device)
    
    # Load the best model weights
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    
    if not X.is_cuda and device.type == "cuda":
        X = X.to(device)

    if not y.is_cuda and device.type == "cuda":
        y = y.to(device)

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X).squeeze()
        # preds = (torch.sigmoid(outputs) > 0.5).float()  # Added sigmoid here
        preds = (outputs > 0.5).float()  # Added sigmoid here
        acc = (preds == y.float()).float().mean()
        
    print(f"Accuracy: {acc.item()*100:.2f}%")
    
    return acc.item()

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    config = GPTConfig44_Complete()

    for capture_layer in range(4):
        construct_binding_id_dataset(
            config, 
            capture_layer)
    # construct_binding_id_dataset(
    #     config=config, 
    #     capture_layer=0)
    
    # config = GPTConfig44_Com
    # plete()
    # project = "Attribute From Last Attribute"
    # run_layer_probe_similarity_analysis(
    #     project=project,
    #     layers=range(4),
    #     attributes=[6, 19, 20, 3, 17, 18, 9, 5, 15, 8, 1, 11],
    #     tokenizer_path=config.tokenizer_path
    # )
