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
import matplotlib.pyplot as plt

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'
FIG_SAVE_PATH = "COMPLETE_FIGS/setnet"


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
        print(
            f"Length of class 0 data: {len(class_0_data)}, Length of class 1 data: {len(class_1_data)}")
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
    def __init__(self, hidden_size=24):
        """
        Initialize the SetNet model with customizable hidden layer size.

        Args:
            hidden_size (int): Number of neurons in the hidden layer
        """
        super(SetNet, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(36, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
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


def create_analysis_dataloader(original_dataloader, batch_size=16):
    """
    Create a non-shuffled DataLoader from a shuffled one for consistent analysis.
    """
    # First, collect all data from the original loader
    all_data = []
    all_targets = []
    
    for data, target in original_dataloader:
        all_data.append(data)
        all_targets.append(target)
    
    # Concatenate all batches
    all_data = torch.cat(all_data, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Create a TensorDataset
    dataset = TensorDataset(all_data, all_targets)
    
    # Create a new DataLoader without shuffling
    if batch_size is None:
        batch_size = len(dataset)  # Use a single batch for simplicity
        
    non_shuffled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    torch.save(non_shuffled_loader, f"{PATH_PREFIX}/colab/non_shuffled_train_loader.pth")

def train_model(
    project,
    hidden_size=24,
    batch_size=16,
    learning_rate=1e-3,
    num_epochs=200,
    patience=10,
    val_split=0.1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Load the binary dataset
    train_loader, val_loader = load_binary_dataloader()

    # Initialize model, optimizer, and move to device
    model = SetNet(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize wandb
    run_name = f"setnet_hidden_{hidden_size}"
    wandb.init(
        project=project,
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "val_split": val_split,
            "hidden_size": hidden_size
        },
        name=run_name
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
        print(
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(
            f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

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

    # Save model with hidden size in path
    save_path = f"{PATH_PREFIX}/{project}/hidden_{hidden_size}"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(),
               f"{save_path}/model.pt")
    wandb.finish()
    return model


def get_layer_activations(model, layer_name, data_loader, device='cuda' if torch.cuda.is_available() else 'cpu', save_activations_path=None):
    """
    Get activations from a specific layer in the model.

    Args:
        model: The trained model
        layer_name: Name of the layer to extract activations from 
        data_loader: DataLoader containing samples
        device: Device for computation

    Returns:
        Tensor containing activations
    """
    activations = []

    def hook_fn(module, input, output):
        # Move the tensor to CPU before storing
        activations.append(output.detach().cpu())

    # Register a hook to the desired layer
    hook = dict([*model.named_modules()]
                )[layer_name].register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for batch_embeddings, batch_targets in data_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_embeddings).squeeze()

    # Unregister the hook
    hook.remove()

    activations = torch.cat(activations, dim=0)

    if save_activations_path is not None:
        os.makedirs(os.path.dirname(save_activations_path), exist_ok=True)
        torch.save(activations, f"{save_activations_path}")
    return activations


def generate_activations_hist(activations, hidden_size, layer_name):
    """
    Generate histogram of activations.

    Args:
        activations: Tensor of activations
        hidden_size: Number of neurons (for plotting layout)
        layer_name: Name of the layer (for title)

    Returns:
        matplotlib figure
    """
    activations_numpy = activations.numpy()

    # Calculate appropriate subplot grid size based on hidden_size
    grid_size = determine_grid_size(hidden_size)
    rows, cols = grid_size

    plt.figure(figsize=(cols * 6, rows * 4))
    for i in range(activations_numpy.shape[1]):
        plt.subplot(rows, cols, i+1)
        plt.hist(activations_numpy[:, i], bins=50, alpha=0.7)
        plt.title(f'Neuron {i}')
        plt.xlabel('Activation')
        plt.ylabel('Frequency')
    plt.suptitle(
        f'{layer_name} Activation Distributions ({hidden_size} neurons)', fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return plt.gcf()


def determine_grid_size(num_neurons):
    """
    Determine appropriate grid size for plotting based on number of neurons.

    Args:
        num_neurons: Number of neurons

    Returns:
        Tuple (rows, cols) for subplot grid
    """
    # Find factors close to square root
    sqrt_n = int(np.sqrt(num_neurons))

    # Find the smallest factor >= sqrt_n that divides num_neurons
    while num_neurons % sqrt_n != 0 and sqrt_n > 0:
        sqrt_n -= 1

    if sqrt_n == 0:  # Prime number case
        sqrt_n = 1

    rows = sqrt_n
    cols = num_neurons // sqrt_n

    return rows, cols


def load_model(project, hidden_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load the model
    model_path = f"{PATH_PREFIX}/{project}/hidden_{hidden_size}/model.pt"
    if not os.path.exists(model_path):
        print(
            f"Model with hidden size {hidden_size} not found at {model_path}")
        return

    model = SetNet(hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    return model


def analyze_model(project, hidden_size, layer_names=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Analyze a trained model by visualizing activations of different layers.

    Args:
        project: Project name
        hidden_size: Hidden layer size of the model
        layer_names: List of layer names to analyze (default: all layers)
        device: Device for computation
    """
    if layer_names is None:
        layer_names = ["fc1", "relu1"]  # Default layers to analyze

    # Load the model
    model = load_model(project, hidden_size, device)

    # Load data
    train_loader, val_loader = load_binary_dataloader()

    # Create output directory for figures
    fig_save_path = f"{FIG_SAVE_PATH}/hidden_{hidden_size}"
    os.makedirs(fig_save_path, exist_ok=True)

    # Analyze each layer
    for layer_name in layer_names:
        print(f"Analyzing {layer_name} activations...")
        activations = get_layer_activations(
            model,
            layer_name,
            train_loader,
            device,)

        fig = generate_activations_hist(activations, hidden_size, layer_name)
        fig.savefig(
            f"{fig_save_path}/{layer_name}_activations.png", bbox_inches="tight")
        plt.close(fig)

        print(f"Saved activation histogram for {layer_name}")


def plot_weight_heatmaps(model, hidden_size, project="setnet"):
    """
    Plot heatmaps of model weight matrices with feature grouping.

    Args:
        model: The trained SetNet model
        hidden_size: Number of neurons in hidden layer
        project: Project name for file paths
    """
    # Load model if needed
    if model is None:
        model_path = f"{PATH_PREFIX}/{project}/hidden_{hidden_size}/model.pt"
        model = SetNet(hidden_size=hidden_size)
        model.load_state_dict(torch.load(model_path))
        model.cpu()  # Ensure model is on CPU for data extraction

    # Get weights
    # Shape: [hidden_size, 36]
    fc1_weights = model.fc1.weight.data.cpu().numpy()
    # Shape: [1, hidden_size]
    fc2_weights = model.fc2.weight.data.cpu().numpy()

    # Create figure directory
    fig_save_path = f"{FIG_SAVE_PATH}/hidden_{hidden_size}"
    os.makedirs(fig_save_path, exist_ok=True)

    # Plot FC1 weights (36 x hidden_size) with feature group delineation
    plt.figure(figsize=(12, 8))

    # Create the heatmap
    im = plt.imshow(fc1_weights, cmap='viridis')
    plt.colorbar(im)

    # Add vertical lines to separate each group of 12 features (each card)
    for i in range(1, 3):
        plt.axvline(x=i*12-0.5, color='red', linestyle='-', linewidth=2)

    # Customize title and labels
    plt.title(f'FC1 Weights (Input → Hidden Layer) - {hidden_size} neurons')
    plt.xlabel('Input Feature (grouped by cards)')
    plt.ylabel('Hidden Neuron')

    # Add card labels below the x-axis
    plt.text(6, hidden_size+1, 'Card 1', ha='center')
    plt.text(18, hidden_size+1, 'Card 2', ha='center')
    plt.text(30, hidden_size+1, 'Card 3', ha='center')

    # Extend the bottom margin to fit the card labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    plt.savefig(f"{fig_save_path}/fc1_weights_heatmap.png",
                bbox_inches="tight")
    plt.close()

    # Plot FC2 weights (hidden_size x 1)
    plt.figure(figsize=(10, 4))
    # Reshape to make it a proper heatmap
    fc2_weights_reshaped = fc2_weights.T  # [hidden_size, 1]
    im = plt.imshow(fc2_weights_reshaped, cmap='viridis', aspect='auto')
    plt.colorbar(im)
    plt.title(f'FC2 Weights (Hidden → Output Layer) - {hidden_size} neurons')
    plt.xlabel('Output Neuron')
    plt.ylabel('Hidden Neuron')
    plt.savefig(f"{fig_save_path}/fc2_weights_heatmap.png",
                bbox_inches="tight")
    plt.close()

    print(f"Weight heatmaps saved to {fig_save_path}")


# def is_triplet_set(key_list, index):
#     return (
#         np.all(np.sum(key_list[index], axis=0) == [1, 1, 1], axis=0) or
#         np.all(np.sum(key_list[index], axis=0) == [3, 0, 0], axis=0) or
#         np.all(np.sum(key_list[index], axis=0) == [0, 3, 0], axis=0) or
#         np.all(np.sum(key_list[index], axis=0) == [0, 0, 3], axis=0))
# Alternative implementation using the original approach but with proper conversion
def is_triplet_set_alt(category_to_triplet, index):
    """
    Alternative implementation of is_triplet_set using the original approach
    but with proper conversion to numpy arrays.
    """
    try:
        # Convert the tuple of tuples to a numpy array
        triplet = np.array([[int(val) for val in card] for card in category_to_triplet[index]])
        
        # Sum along axis 0 (across the 3 cards) for each attribute position
        sums = np.sum(triplet, axis=0)
        
        # Check the conditions:
        # - Sum is [3,0,0] or [0,3,0] or [0,0,3] -> all cards have same value
        # - Sum is [1,1,1] -> all cards have different values
        return (
            np.array_equal(sums, [1, 1, 1]) or  # All different
            np.array_equal(sums, [3, 0, 0]) or  # All same, first value
            np.array_equal(sums, [0, 3, 0]) or  # All same, second value
            np.array_equal(sums, [0, 0, 3])     # All same, third value
        )
    except Exception as e:
        print(f"Error in is_triplet_set_alt: {e}")
        print(f"category_to_triplet[{index}]: {category_to_triplet[index]}")
        return False

# Define a method to get attribute triplets


# def categorize_triplet(sequence, index):
#     # Extract last 3 elements of each card in the triplet
#     sequence = sequence.astype(int)
#     card1 = tuple(sequence[0:12][index*3: (index + 1) * 3])
#     card2 = tuple(sequence[12:24][index*3: (index + 1) * 3])
#     card3 = tuple(sequence[24:36][index*3: (index + 1) * 3])

#     triplet_type = (card1, card2, card3)

#     # Convert the tuple into a category index (hashable representation)
#     return triplet_type

def categorize_triplet(sequence, index):
    # Extract last 3 elements of each card in the triplet
    sequence = sequence.astype(int)
    # Convert np.int64 to regular Python int
    card1 = tuple(int(x) for x in sequence[0:12][index*3: (index + 1) * 3])
    card2 = tuple(int(x) for x in sequence[12:24][index*3: (index + 1) * 3])
    card3 = tuple(int(x) for x in sequence[24:36][index*3: (index + 1) * 3])

    triplet_type = (card1, card2, card3)

    # Convert the tuple into a category index (hashable representation)
    return triplet_type

def triplet_type_to_labels(triplet_type, attribute_index):
    """
    Convert a triplet type to a list of labels for each card in the triplet.

    Args:
        triplet_type: Tuple representing the triplet type

    Returns:
        List of labels for each card in the triplet
    """
    card_labels = ""
    for index, card in enumerate(triplet_type):
        card_labels += attr_id_to_name_dict[attribute_index][card.index(1)]
        if index < 2:
            card_labels += " | "
        # card_labels.append(attr_id_to_name_dict[attribute_index][card.index(1)])
    return card_labels

    
def assign_triplet_categories(dataloader, index):
    # Dictionary to store each unique triplet type and its corresponding category ID
    categories = {}

    category_to_triplet = {}
    category_counter = 0

    # Array to store the category assignment for each triplet
    triplet_categories = []
    # labels = []
    # Iterate through all triplets in X_train
    # for i in range(dataset.shape[0]):
    for batch_embeddings, batch_targets in dataloader:
        for sequence in batch_embeddings:
            sequence = sequence.numpy()
            triplet_type = categorize_triplet(sequence, index)

            # Assign a category ID to the triplet type if not already assigned
            if triplet_type not in categories:
                categories[triplet_type] = category_counter
                category_to_triplet[category_counter] = triplet_type
                category_counter += 1

            # Append the category ID to the result list
            triplet_categories.append(categories[triplet_type])
            # labels.append(triplet_type_to_labels(triplet_type, index))
            # breakpoint()

    # Convert to a NumPy array for further use
    triplet_categories = np.array(triplet_categories) # [0, 1, 2, 1, 5, 6, ...]

    return triplet_categories, category_to_triplet


attribute_map = {0: "shape", 1: "color", 2: "number", 3: "shading"}

attr_id_to_name_dict = {
    0: ["oval", "squiggle", "diamond"],
    1: ["red", "green", "purple"],
    2: ["one", "two", "three"],
    3: ["solid", "striped", "open"]
}
# shapes = [0, 1, 2]  # 0: oval, 1: squiggle, 2: diamond
#     colors = [0, 1, 2]  # 0: red, 1: green, 2: purple
#     numbers = [0, 1, 2]  # 0: one, 1: two, 2: three
#     shadings = [0, 1, 2]  # 0: solid, 1: striped, 2: open


def plot_activations_by_triplet_category(activations, neuron_index, dataloader, attribute_index, hidden_size, savefig=False):
    # Create a color map to distinguish between the categories
    triplet_categories, category_to_triplet  = assign_triplet_categories(
        dataloader, attribute_index)

    colors = plt.cm.get_cmap('tab20', 27)
    # colors = plt.cm.get_cmap('tab20', 9)

    # Plot histograms for each category
    plt.figure(figsize=(10, 8))

    # for category in range(20, 21):
    for category in range(27):
        # Filter the activations that belong to the current category
        category_activations = activations[:,
                                           neuron_index][triplet_categories == category]

        # Plot the histogram for the current category
        curr_label = triplet_type_to_labels(category_to_triplet[category], attribute_index)
        plt.hist(category_activations, bins=30, alpha=0.5,
                 label=curr_label, color=colors(category))

    # Add labels and a legend
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.title(
        f'Activations for Neuron {neuron_index} Categorized by Attribute {attribute_index} ({attribute_map[attribute_index]})')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

    # Show the plot
    plt.tight_layout()
    if savefig:
        save_fig_path = f"{FIG_SAVE_PATH}/hidden_{hidden_size}"
        os.makedirs(save_fig_path, exist_ok=True)
        plt.savefig(
            f"{save_fig_path}/activations_neuron{neuron_index}_index{attribute_index}.png", bbox_inches="tight")
    plt.show()


def plot_activation_grid_by_triplet_category(activations, neuron_index, dataloader, attribute_index, hidden_size, savefig=False):
    triplet_categories, category_to_triplet = assign_triplet_categories(
        dataloader, attribute_index)

    # Create a color map to distinguish between the categories
    colors = plt.cm.get_cmap('tab20', 27)

    # Set up a 9x3 grid for subplots
    fig, axes = plt.subplots(9, 3, figsize=(10, 20))  # 9 rows, 3 columns
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    for category in range(27):
        # for category in range(20, 21):
        # Filter the activations that belong to the current category
        category_activations = activations[:,
                                           neuron_index][triplet_categories == category]

        # Plot the histogram in the respective subplot
        ax = axes[category]
        ax.hist(category_activations, bins=30,
                alpha=0.5, color=colors(category))

        # ax.set_xlim(-0.1, 24)
        # ax.set_ylim(0, 3000)

        # Add title and labels for the individual subplot
        if is_triplet_set_alt(category_to_triplet, category):
            curr_label = triplet_type_to_labels(category_to_triplet[category], attribute_index)
            ax.set_title(f'SET: {curr_label}')
        else:
            ax.set_title(f'{curr_label}')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Frequency')

    # Adjust layout to prevent overlap
    plt.suptitle(
        f'Activations for Neuron {neuron_index} Categorized by Attribute {attribute_index} ({attribute_map[attribute_index]})')
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save the figure
    if savefig:
        save_fig_path = f"{FIG_SAVE_PATH}/hidden_{hidden_size}"
        os.makedirs(save_fig_path, exist_ok=True)
        plt.savefig(
            f"{save_fig_path}/activations_grid_neuron{neuron_index}_index{attribute_index}.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Create overall figure directory
    # os.makedirs(FIG_SAVE_PATH, exist_ok=True)

    
    hidden_size = 16
    model = load_model(project="setnet", hidden_size=hidden_size)
    # train_loader, val_loader = load_binary_dataloader()
    # create_analysis_dataloader(train_loader, batch_size=16)
    layer_name = "fc1"

    analysis_loader = torch.load(f"{PATH_PREFIX}/colab/non_shuffled_train_loader.pth")
    fig_save_path = f"{FIG_SAVE_PATH}/hidden_{hidden_size}"
    activations = get_layer_activations(
        model,
        layer_name,
        analysis_loader,
        save_activations_path=f"{fig_save_path}/{layer_name}_non_shuffled_train_activations.pth")

    activations = torch.load(f"{fig_save_path}/{layer_name}_non_shuffled_train_activations.pth")

    neuron_indices = [10, 15]
    for neuron_index in neuron_indices:
        plot_activations_by_triplet_category(
            activations,
            neuron_index=neuron_index,
            dataloader=analysis_loader,
            attribute_index=2,
            hidden_size=hidden_size,
            savefig=True
        )
        plot_activation_grid_by_triplet_category(
            activations,
            neuron_index=neuron_index,
            dataloader=analysis_loader,
            attribute_index=2,
            hidden_size=hidden_size,
            savefig=True)
    # Example usage:
    # 1. Generate data (run once)
    # generate_data()

    # 2. Train models with different hidden sizes
    # hidden_sizes = [8, 16, 24, 32, 64]
    # for hidden_size in hidden_sizes:
    #     print(f"Training model with {hidden_size} hidden neurons...")
    #     train_model(project="setnet", hidden_size=hidden_size)

    # # 3. Analyze trained models
    # hidden_sizes = [8, 16, 24, 32, 64]
    # for hidden_size in hidden_sizes:
    #     print(f"Analyzing model with {hidden_size} hidden neurons...")
    #     analyze_model(project="setnet", hidden_size=hidden_size)

    # plot_weight_heatmaps(model=None, hidden_size=16, project="setnet")
