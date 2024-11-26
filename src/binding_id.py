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
    save_threshold = 1000000  # Save after accumulating 1M samples
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


def train_binding_classifier(X, y, model_name, input_dim=128, num_epochs=100, batch_size=32, lr=0.001, patience=10):
    """Train a binary classifier for binding identification."""
    # Initialize wandb
    wandb.init(
        project="binding-id-classifier", 
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience
        },
        name=model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)
    # Prepare data with validation split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)


    # # Initialize model, loss function, and optimizer
    # model = nn.Sequential(
    #     nn.Linear(input_dim, 64),
    #     nn.ReLU(),
    #     nn.Linear(64, 1),
    #     nn.Sigmoid()
    # ).to(device)

    # Initialize a simple linear model
    model = nn.Sequential(
        nn.Linear(input_dim, 1),
        nn.Sigmoid()  # Keep sigmoid at output for binary classification
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):

        model.train()
        total_train_loss = 0
        
        # Training loop
        
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train.float())
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True
        )

        # Process batches from the DataLoader
        for batch_X, batch_y in train_loader: 
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / (len(X_train) / batch_size)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
        
        # Validation loop
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val.float())
            
            val_preds = (val_outputs > 0.5).float()
            val_acc = (val_preds == y_val.float()).float().mean()
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss.item(),
            'val_acc': val_acc.item()
        })
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save best model
            torch.save(model.state_dict(), f'{PATH_PREFIX}/binding_id/{model_name}_best.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
                
    # Final evaluation
    test_acc = get_binding_classifier_accuracy(X_test, y_test, f'{PATH_PREFIX}/binding_id/{model_name}_best.pt', input_dim)
    val_acc = get_binding_classifier_accuracy(X_val, y_val, f'{PATH_PREFIX}/binding_id/{model_name}_best.pt', input_dim)
    
    wandb.log({'test_accuracy': test_acc})
    wandb.log({'val_accuracy': val_acc})
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
