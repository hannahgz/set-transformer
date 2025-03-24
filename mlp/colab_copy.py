from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import itertools
import os
from sklearn.model_selection import train_test_split
import wandb

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

class BalancedDataset(Dataset):
    def __init__(self, class_0_data, class_1_data):
        """
        Args:
            class_0_data (Tensor): Data where y = 0.
            class_1_data (Tensor): Data where y = 1.
        """
        self.class_0_data = class_0_data
        self.class_1_data = class_1_data
        self.length = min(len(class_0_data), len(
            class_1_data)) * 2  # Ensure 50/50 split
        print(f"Length of class 0 data: {len(class_0_data)}, Length of class 1 data: {len(class_1_data)}")
        print("Length of dataset: ", self.length)

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


class SetNet(nn.Module):
    def __init__(self):
        super(SetNet, self).__init__()
        self.fc1 = nn.Linear(36, 24)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(24, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Helper function to one-hot encode a card


def one_hot_encode_card(shape, color, number, shading):
    encoding = [0]*12
    encoding[shape] = 1
    encoding[3 + color] = 1
    encoding[6 + number] = 1
    encoding[9 + shading] = 1
    return encoding

# Function to check if three cards form a set


def is_set(triplet):
    for i in range(4):  # For each feature
        feature_values = [triplet[0][i*3:(i+1)*3].index(1),
                          triplet[1][i*3:(i+1)*3].index(1),
                          triplet[2][i*3:(i+1)*3].index(1)]
        if len(set(feature_values)) not in [1, 3]:
            return False
    return True


def generate_data(batch_size=16):

    # Generate all possible cards
    shapes = [0, 1, 2]  # 0: oval, 1: squiggle, 2: diamond
    colors = [0, 1, 2]  # 0: red, 1: green, 2: purple
    numbers = [0, 1, 2]  # 0: one, 1: two, 2: three
    shadings = [0, 1, 2]  # 0: solid, 1: striped, 2: open

    cards = list(itertools.product(shapes, colors, numbers, shadings))

    # Encode all cards
    encoded_cards = [one_hot_encode_card(*card) for card in cards]

    # Generate all possible triplets of cards
    triplets = list(itertools.permutations(encoded_cards, 3))

    # Generate dataset
    X = []
    y = []

    for index, triplet in enumerate(triplets):
        print(f"Processing triplet {index}/{len(triplets)}")
        X.append(np.concatenate(triplet))
        y.append(1 if is_set(triplet) else 0)

    X = np.array(X)
    y = np.array(y)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    # Step 2: Convert the dataset to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Shuffle the training data
    permutation_train = torch.randperm(X_train_tensor.size()[0])
    X_train_tensor = X_train_tensor[permutation_train]
    y_train_tensor = y_train_tensor[permutation_train]

    # Split X_train_tensor into two groups based on y_train_tensor
    class_0_data_train = X_train_tensor[y_train_tensor.squeeze() == 0]
    class_1_data_train = X_train_tensor[y_train_tensor.squeeze() == 1]

    permutation_val = torch.randperm(X_test_tensor.size()[0])
    X_val_tensor = X_test_tensor[permutation_val]
    y_val_tensor = y_test_tensor[permutation_val]
    class_0_data_val = X_val_tensor[y_val_tensor.squeeze() == 0]
    class_1_data_val = X_val_tensor[y_val_tensor.squeeze() == 1]

    # Step 3: Create a BalancedDataset for batch processing
    train_dataset = BalancedDataset(class_0_data_train, class_1_data_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = BalancedDataset(class_0_data_val, class_1_data_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Print train_dataset and val_dataset lengths/sizes
    print("Train dataset size: ", len(train_dataset))
    print("Val dataset size: ", len(val_dataset))

    save_data_loader_path = f"{PATH_PREFIX}/colab"
    os.makedirs(save_data_loader_path, exist_ok=True)

    torch.save({
        "train_loader": train_loader,
        "val_loader": val_loader
    }, f"{save_data_loader_path}/data_loader.pth")

def load_binary_dataloader():
    dataset_save_path = f"{PATH_PREFIX}/colab/data_loader.pth"
    saved_data = torch.load(dataset_save_path)
    return saved_data['train_loader'], saved_data['val_loader']

class SetNet(nn.Module):
    def __init__(self):
        super(SetNet, self).__init__()
        self.fc1 = nn.Linear(36, 24)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(24, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.sigmoid(self.fc2(x))
        return x
    
    def predict(self, x, threshold=0.5):
        # Returns binary predictions (0 or 1)
        probs = self.forward(x)
        return (probs >= threshold).float()
    
    def compute_loss(self, outputs, targets):
        # Binary cross entropy loss
        return nn.functional.binary_cross_entropy(outputs, targets)
    
def train_model(
    project,
    batch_size=16,
    learning_rate=1e-3,
    num_epochs=100,
    patience=10,
    val_split=0.1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
): 
    # Load the binary dataset
    train_loader, val_loader = load_binary_dataloader()

    # Initialize model, optimizer, and move to device
    model = SetNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize wandb
    wandb.init(
        project=project,
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "val_split": val_split
        },
        name=f"colab_set_example"
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

    save_path = f"{PATH_PREFIX}/{project}"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), 
               f"{PATH_PREFIX}/{project}/model.pt")
    wandb.finish()
    return model

def get_layer_activations(model, layer_name, data_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach())

    # Register a hook to the desired layer
    hook = dict([*model.named_modules()])[layer_name].register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for batch_embeddings, batch_targets in data_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_embeddings).squeeze()

    # Unregister the hook
    hook.remove()

    return torch.cat(activations, dim=0)

import matplotlib.pyplot as plt
if __name__ == "__main__":
    # generate_data()
    # train_model(project="setnet")
    
    # Load saved model
    model = SetNet()
    model.load_state_dict(torch.load(f"{PATH_PREFIX}/setnet/model.pt"))

    train_loader, val_loader = load_binary_dataloader()
    train_activations = get_layer_activations(model, "fc2", train_loader)

    plt.figure(figsize=(36, 18))
    for i in range(train_activations.shape[1]):
        plt.subplot(6, 4, i+1)  # Adjust the grid size based on the number of neurons
        plt.hist(train_activations[:, i], bins=50, alpha=0.7)
        plt.title(f'Neuron {i}')
        plt.xlabel('Activation')
        plt.ylabel('Frequency')
    plt.suptitle('Activation Distributions for Training Set Neurons', fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_save_path = f"COMPLETE_FIGS/setnet"
    os.makedirs(fig_save_path, exist_ok=True)
    plt.savefig(f"{fig_save_path}/train_activations_24.png", bbox_inches="tight")
    plt.show()

    # get_layer_activations

# # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# train_loader = DataLoader(train_dataset, batch_size=32, sampler = sampler)
