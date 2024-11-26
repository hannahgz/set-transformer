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
PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'


def get_gpu_memory():
    """Get the current gpu usage."""
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', 
         '--format=csv,nounits,noheader'
    ], encoding='utf-8')
    used, total = map(int, result.strip().split(','))
    return used, total


def construct_binding_id_dataset(config, dataset_name, model_path, capture_layer):

    perms = list(permutations(range(20), 2))

    dataset_path = f"{PATH_PREFIX}/{dataset_name}.pth"
    dataset = torch.load(dataset_path)
    train_loader, val_loader = initialize_loaders(config, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = GPT(config).to(device)  # adjust path as needed
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    X = []
    y = []
    save_threshold = 10000000  # Save after accumulating 10M samples
    chunk_counter = 0

    base_dir = f"{PATH_PREFIX}/binding_id/{dataset_name}/layer{capture_layer}"
    os.makedirs(base_dir, exist_ok=True)

    for index, batch in enumerate(val_loader):
        used, total = get_gpu_memory()
        print(f"Batch {index}/{len(val_loader)}")
        print(f"GPU Memory: {used/1024:.2f}GB used out of {total/1024:.2f}GB total")
        print("Current chunk samples: ", len(y))

        batch = batch.to(device)
        with torch.no_grad():  # Reduce memory usage during inference
            _, _, _, captured_embedding = model(batch, True, capture_layer)
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

                # Save intermediate results when threshold is reached
                if len(y) >= save_threshold:
                    X_tensor = torch.stack(X)
                    y_tensor = torch.tensor(y)
                    
                    # Save current chunk
                    torch.save(X_tensor, os.path.join(base_dir, f"X_chunk_{chunk_counter}.pt"))
                    torch.save(y_tensor, os.path.join(base_dir, f"y_chunk_{chunk_counter}.pt"))
                    
                    # Clear lists and increment counter
                    X = []
                    y = []
                    chunk_counter += 1
                    print(f"Saved chunk {chunk_counter}")


    # Save any remaining data
    if len(y) > 0:
        X_tensor = torch.stack(X)
        y_tensor = torch.tensor(y)
        torch.save(X_tensor, os.path.join(base_dir, f"X_chunk_{chunk_counter}.pt"))
        torch.save(y_tensor, os.path.join(base_dir, f"y_chunk_{chunk_counter}.pt"))

    # Save metadata about chunks
    torch.save({'num_chunks': chunk_counter + 1}, os.path.join(base_dir, "metadata.pt"))

def load_and_combine_chunks(base_dir, chunk_indices, device):
    """Load and combine multiple data chunks."""
    X_combined = []
    y_combined = []
    for chunk in chunk_indices:
        X_combined.append(torch.load(os.path.join(base_dir, f"X_chunk_{chunk}.pt")))
        y_combined.append(torch.load(os.path.join(base_dir, f"y_chunk_{chunk}.pt")))
    return torch.cat(X_combined).to(device), torch.cat(y_combined).to(device)

def select_chunks(num_chunks, val_size=1, test_size=1):
    """Randomly select chunks for validation, test, and training sets."""
    all_chunk_indices = list(range(num_chunks))
    val_chunks = sorted(random.sample(all_chunk_indices, val_size))
    remaining_chunks = [i for i in all_chunk_indices if i not in val_chunks]
    test_chunks = sorted(random.sample(remaining_chunks, test_size))
    train_chunks = [i for i in remaining_chunks if i not in test_chunks]
    
    split_info = {
        'val_chunks': val_chunks,
        'test_chunks': test_chunks,
        'train_chunks': train_chunks
    }
    
    print(f"Using chunks {val_chunks} for validation")
    print(f"Using chunks {test_chunks} for testing")
    print(f"Using chunks {train_chunks} for training")
    
    return split_info

def process_chunk(model, chunk, device, criterion, optimizer=None, batch_size=32):
    """Process a single chunk of data for training or evaluation."""
    X_chunk = torch.load(chunk['X_path'])
    y_chunk = torch.load(chunk['y_path'])
    
    dataset = torch.utils.data.TensorDataset(X_chunk.to(device), y_chunk.float().to(device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    total_loss = 0
    samples_processed = 0
    
    for batch_X, batch_y in loader:
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        
        if optimizer is not None:  # Training mode
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * len(batch_y)
        samples_processed += len(batch_y)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return total_loss, samples_processed

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
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, val_size, test_size)  
    
    print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Initialize model and training components
    model = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid()).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    counter = 0
    
    # Create DataLoader for training data
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True
    )
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
       
        total_train_loss = 0
        total_samples = 0
        
        for batch_X, batch_y in train_loader:
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
            model_path = f'{PATH_PREFIX}/binding_id/{model_name}_chunk{chunk_id}_best.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
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
    test_acc = get_binding_classifier_accuracy(X_test, y_test, model_path, input_dim)
    wandb.log({'test_accuracy': test_acc})
    wandb.finish()
    
    return model



def get_binding_classifier_accuracy(X, y, model_path, input_dim=128):
    """Test a trained binding classifier model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the same model architecture
    model = nn.Sequential(
        nn.Linear(input_dim, 1),
        nn.Sigmoid()
    ).to(device)
    
    # Load the best model weights
    model.load_state_dict(torch.load(model_path))
    
    if not X.is_cuda and device.type == "cuda":
        X = X.to(device)

    if not y.is_cuda and device.type == "cuda":
        y = y.to(device)

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X).squeeze()
        preds = (outputs > 0.5).float()
        acc = (preds == y.float()).float().mean()
        
    print(f"Accuracy: {acc.item()*100:.2f}%")
    
    return acc.item()
