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
    save_threshold = 50000000

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
                    print("Saving intermediate results")
                    X_tensor = torch.stack(X)
                    y_tensor = torch.tensor(y)
                    
                    # Save current chunk
                    print("Saving X tensor")
                    torch.save(X_tensor, os.path.join(base_dir, f"X.pt"))

                    print("Saving y tensor")
                    torch.save(y_tensor, os.path.join(base_dir, f"y.pt"))

                    print("Returning")
                    return

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


def initialize_binding_dataset(capture_layer, val_size=0.05, test_size=0.05, batch_size=32):
    # Combine X and y into a single dataset
    base_dir = f"{PATH_PREFIX}/src_clean/binding_id/44_complete/layer{capture_layer}"
    X = torch.load(os.path.join(base_dir, "X.pt"))
    y = torch.load(os.path.join(base_dir, "y.pt"))

    full_dataset = TensorDataset(X, y)

    # Split the combined dataset into training, validation, and test sets
    train_size = int((1-val_size-test_size) * len(full_dataset)) 
    val_size = int(val_size * len(full_dataset))  # 15% for validation
    test_size = len(full_dataset) - train_size - val_size  # Remainder for test
    # print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

    print("Random splitting")
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) 
    )

    print("Separating training data into class 0 and 1")
    # Separate training data into class 0 and class 1
    X_train, y_train = zip(*train_dataset)  # Unpack into X_train and y_train
    X_train = torch.stack(X_train)
    y_train = torch.stack(y_train)

    class_0_data_train = X_train[(y_train == 0).nonzero(as_tuple=True)[0]]
    class_1_data_train = X_train[(y_train == 1).nonzero(as_tuple=True)[0]]

    print("Separating validation data into class 0 and 1")
    # Separate validation data into class 0 and class 1
    X_val, y_val = zip(*val_dataset)  # Unpack into X_val and y_val
    X_val = torch.stack(X_val)
    y_val = torch.stack(y_val)

    class_0_data_val = X_val[(y_val == 0).nonzero(as_tuple=True)[0]]
    class_1_data_val = X_val[(y_val == 1).nonzero(as_tuple=True)[0]]

    print("Separating test data into class 0 and 1")
    # Separate test data into class 0 and class 1
    X_test, y_test = zip(*test_dataset)  # Unpack into X_test and y_test
    X_test = torch.stack(X_test)
    y_test = torch.stack(y_test)

    class_0_data_test = X_test[(y_test == 0).nonzero(as_tuple=True)[0]]
    class_1_data_test = X_test[(y_test == 1).nonzero(as_tuple=True)[0]]

    print("Saving indiv classes")
    torch.save({
        "class_0_data_train": class_0_data_train,
        "class_1_data_train": class_1_data_train,
        "class_0_data_val": class_0_data_val,
        "class_1_data_val": class_1_data_val,
        "class_0_data_test": class_0_data_test,
        "class_1_data_test": class_1_data_test
    }, os.path.join(base_dir, "balanced_data.pt"))
    
    # print("Creating balanced datasets")
    # # Create the balanced training dataset
    # balanced_train_dataset = BalancedBindingDataset(class_0_data_train, class_1_data_train)
    # balanced_val_dataset = BalancedBindingDataset(class_0_data_val, class_1_data_val)
    # balanced_test_dataset = BalancedBindingDataset(class_0_data_test, class_1_data_test)

    # print("Creating dataloaders")
    # # Create DataLoaders
    # train_loader = DataLoader(balanced_train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(balanced_val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(balanced_test_dataset, batch_size=batch_size, shuffle=False)

    # breakpoint()

    # print("Saving dataloaders")
    # torch.save({
    #     "train_loader": train_loader,
    #     "val_loader": val_loader,
    #     "test_loader": test_loader
    # }, os.path.join(base_dir, "balanced_dataloader.pt"))
    
    # return train_loader, val_loader, test_loader

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

def load_binary_dataloader(capture_layer, batch_size):
    base_dir = f"{PATH_PREFIX}/src_clean/binding_id/44_complete/layer{capture_layer}"

    saved_data = torch.load(os.path.join(base_dir, "balanced_data.pt"))
    balanced_train_dataset = BalancedBindingDataset(saved_data["class_0_data_train"], saved_data["class_1_data_train"])
    balanced_val_dataset = BalancedBindingDataset(saved_data["class_0_data_val"], saved_data["class_1_data_val"])
    balanced_test_dataset = BalancedBindingDataset(saved_data["class_0_data_test"], saved_data["class_1_data_test"])

    train_loader = DataLoader(balanced_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(balanced_val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(balanced_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_binary_probe(
    capture_layer,
    project,
    embedding_dim=128,
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=100,
    patience=10,
    device='cuda' if torch.cuda.is_available() else 'cpu'
): 
    # Load the binary dataset
    print("Loading binary dataset")
    train_loader, val_loader, test_loader = load_binary_dataloader(capture_layer, batch_size=batch_size)
    print("Binary dataset loaded")

    # Initialize model, optimizer, and move to device
    model = BinaryProbe(embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize wandb
    wandb.init(
        project=project,
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "embedding_dim": embedding_dim,
            "capture_layer": capture_layer,
        },
        name=f"layer{capture_layer}"
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

    base_dir = f"{PATH_PREFIX}/src_clean/binding_id/44_complete/layer{capture_layer}"
    torch.save(model.state_dict(), 
               os.path.join(base_dir, "binary_probe.pt"))
    wandb.finish()
    return model



# def get_binding_classifier_accuracy(X, y, model_path, input_dim=128):
#     """Test a trained binding classifier model."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Initialize the same model architecture
#     model = nn.Sequential(
#         nn.Linear(input_dim, 1)
#     ).to(device)
    
#     # Load the best model weights
#     checkpoint = torch.load(model_path, weights_only=False)
#     model.load_state_dict(checkpoint["model"])
    
#     if not X.is_cuda and device.type == "cuda":
#         X = X.to(device)

#     if not y.is_cuda and device.type == "cuda":
#         y = y.to(device)

#     # Evaluation
#     model.eval()
#     with torch.no_grad():
#         outputs = model(X).squeeze()
#         # preds = (torch.sigmoid(outputs) > 0.5).float()  # Added sigmoid here
#         preds = (outputs > 0.5).float()  # Added sigmoid here
#         acc = (preds == y.float()).float().mean()
        
#     print(f"Accuracy: {acc.item()*100:.2f}%")
    
#     return acc.item()

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    config = GPTConfig44_Complete()

    # for capture_layer in range(4):
    #     # construct_binding_id_dataset(
    #     #     config, 
    #     #     capture_layer)
    #     # initialize_binding_dataset(capture_layer)
    #     train_binary_probe(
    #         capture_layer=capture_layer,
    #         project="acc-binding-id",
    #         num_epochs=1,
    #     )

    capture_layer = 0
    # initialize_binding_dataset(capture_layer)
    # print("Finished initializing dataset")
    train_binary_probe(
        capture_layer=capture_layer,
        project="acc-binding-id",
    )

    # capture_layer = 1
    # initialize_binding_dataset(capture_layer)
    # print("Finished initializing dataset")
    # train_binary_probe(
    #     capture_layer=capture_layer,
    #     project="acc-binding-id",
    # )

    # capture_layer = 2
    # initialize_binding_dataset(capture_layer)
    # print("Finished initializing dataset")
    # train_binary_probe(
    #     capture_layer=capture_layer,
    #     project="acc-binding-id",
    # )

    # capture_layer = 3
    # initialize_binding_dataset(capture_layer)
    # print("Finished initializing dataset")
    # train_binary_probe(
    #     capture_layer=capture_layer,
    #     project="acc-binding-id",
    # )
