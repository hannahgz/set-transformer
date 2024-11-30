import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pickle
from sklearn.preprocessing import LabelEncoder
from data_utils import split_data
from tokenizer import load_tokenizer

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Linear layer (16 -> 12)

    def forward(self, x):
        # No activation, as we will use CrossEntropyLoss which applies softmax internally
        return self.fc(x)


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        # self.fc1 = nn.Linear(input_dim, hidden_dim)  # First hidden layer
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer

        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Second hidden layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # x = self.relu2(x)
        # x = self.fc3(x)
        return x


def evaluate_model(model, X, y, model_name, predict_dim, continuous_to_original_path=None, tokenizer_path=None):
    """Evaluates the model on data and prints correct predictions."""

    checkpoint = torch.load(
        f'{PATH_PREFIX}/classify/{model_name}.pt', weights_only=False)
    model.load_state_dict(checkpoint["model"])

    model.eval()  # Set the model to evaluation mode
    correct_predictions = []

    # Dictionary to count frequency of correct predictions for each class
    class_correct_counts = {i: 0 for i in range(predict_dim)}
    # Dictionary to count total occurrences of each class
    class_total_counts = {i: 0 for i in range(predict_dim)}

    # input_correct_counts = {i: 0 for i in range(input_dim)}
    # # Dictionary to count total occurrences of each class
    # input_total_counts = {i: 0 for i in range(input_dim)}

    # these mod 20 counts are not accurate because the order is changing for X so we don't actually know which position in the sequence corresponds to which attribute

    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)

        for idx, (pred, true) in enumerate(zip(predicted, y)):
            true_label = true.item()
            class_total_counts[true_label] += 1

            if pred == true:
                correct_predictions.append((idx, pred.item()))
                class_correct_counts[true_label] += 1

        accuracy = len(correct_predictions) / y.size(0)

    print("\nPer-class statistics:")

    if continuous_to_original_path and tokenizer_path:
        log_per_class_statistics(continuous_to_original_path, tokenizer_path, model_name, predict_dim, class_total_counts, class_correct_counts)
        # with open(continuous_to_original_path, 'rb') as f:
        #     continuous_to_original = pickle.load(f)
        # tokenizer = load_tokenizer(tokenizer_path)
        # for class_idx in range(predict_dim):
        #     total = class_total_counts[class_idx]
        #     correct = class_correct_counts[class_idx]
        #     accuracy_per_class = (correct / total) if total > 0 else 0
        #     original_token = continuous_to_original.get(
        #         class_idx, f"Unknown token {class_idx}")
        #     print(
        #         f"\nClass {class_idx}, Original Token: ({original_token})/{tokenizer.decode([original_token])}:")
        #     print(f"  Accuracy: {accuracy_per_class:.4f} ({correct}/{total})")

    # print("Correctly predicted values and their indices:")
    # for idx, value in correct_predictions:
    #     print(f"Index: {idx}, mod {idx % 20}, Predicted Value: {value}")

    print(f"\nTotal correct predictions: {len(correct_predictions)}")
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy

def log_per_class_statistics(continuous_to_original_path, tokenizer_path, model_name, predict_dim, class_total_counts, class_correct_counts):
    with open(continuous_to_original_path, 'rb') as f:
        continuous_to_original = pickle.load(f)
    tokenizer = load_tokenizer(tokenizer_path)
    # Save the printed logs into a txt file
    log_file_path = f'{PATH_PREFIX}/classify/{model_name}_evaluation_log.txt'
    print("log_file_path: ", log_file_path)
    with open(log_file_path, 'w') as log_file:
        log_file.write("\nPer-class statistics:\n")

        if continuous_to_original_path and tokenizer_path:
            for class_idx in range(predict_dim):
                total = class_total_counts[class_idx]
                correct = class_correct_counts[class_idx]
                accuracy_per_class = (correct / total) if total > 0 else 0
                original_token = continuous_to_original.get(
                    class_idx, f"Unknown token {class_idx}")
                
                print(
                    f"\nClass {class_idx}, Original Token: ({original_token})/{tokenizer.decode([original_token])}:")
                print(f"  Accuracy: {accuracy_per_class:.4f} ({correct}/{total})")

                log_file.write(
                    f"\nClass {class_idx}, Original Token: ({original_token})/{tokenizer.decode([original_token])}:\n")
                log_file.write(f"  Accuracy: {accuracy_per_class:.4f} ({correct}/{total})\n")

def train_model(model, train_data, val_data, criterion, optimizer, num_epochs=100, batch_size=32, patience=5, model_name=None):
    """Trains the model using validation accuracy for early stopping."""
    X_train, y_train = train_data
    X_val, y_val = val_data

    wandb.init(
        project="classify-card",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "patience": patience,
        },
        name=model_name
    )

    counter = 0
    best_val_loss = 1e10

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        permutation = torch.randperm(X_train.size(0))
        train_loss = 0

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / (X_train.size(0) // batch_size)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
        })

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            counter = 0
            best_val_loss = val_loss

            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch_num": epoch,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, f'{PATH_PREFIX}/classify/{model_name}.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break


def run_classify(X, y, model_name, input_dim, output_dim, num_epochs=100, batch_size=32, lr=0.001, model_type="linear", continuous_to_original_path=None, tokenizer_path=None):
    """Main function to run the model training and evaluation."""
    # Prepare data with validation split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss function, and optimizer
    if model_type == "linear":
        model = LinearModel(input_dim, output_dim).to(device)
    elif model_type == "mlp":
        model = MLPModel(input_dim=input_dim, hidden_dim=32,
                         output_dim=output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model with validation data
    train_model(
        model,
        (X_train, y_train),
        (X_val, y_val),
        criterion,
        optimizer,
        num_epochs,
        batch_size,
        model_name=f"{model_name}_{model_type}"
    )

    # Evaluate the model
    train_accuracy = evaluate_model(
        model, 
        X_train, 
        y_train, 
        model_name=f"{model_name}_{model_type}",
        predict_dim=5)
    val_accuracy = evaluate_model(
        model, 
        X_val, 
        y_val, 
        model_name=f"{model_name}_{model_type}", 
        predict_dim=5)
    test_accuracy = evaluate_model(
        model, 
        X_test,
        y_test,
        model_name=f"{model_name}_{model_type}",
        continuous_to_original_path=continuous_to_original_path,
        tokenizer_path=tokenizer_path,
        predict_dim=5)

    wandb.log({
        "final_train_accuracy": train_accuracy,
        "final_val_accuracy": val_accuracy,
        "final_test_accuracy": test_accuracy
    })
    wandb.finish()

    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


def analyze_weights(model_path, input_dim=64, output_dim=5):
    model = LinearModel(input_dim, output_dim)

    # Load the model's state_dict
    model.load_state_dict(torch.load(f"{PATH_PREFIX}/classify/{model_path}.pt")["model"])
    model.eval()

    # Access weights and biases
    weights = model.fc.weight.data
    biases = model.fc.bias.data

    print("Weights:\n", weights)
    print("Biases:\n", biases)

    # Iterate through the weight tensor and print the index and the value
    for i in range(weights.size(0)):  # Iterate over the output_dim (rows)
        for j in range(weights.size(1)):  # Iterate over the input_dim (columns)
            print(f"Weight at index ({i}, {j}): {weights[i, j].item()}")

    return weights, biases

import matplotlib.pyplot as plt
import seaborn as sns

def plot_weights_as_heatmap(weights, savefig_path=None):
    """
    Visualizes the weight matrix for a single layer as a heatmap.
    
    Parameters:
    weights (torch.Tensor): The weight matrix of shape [output_dim, input_dim].
    """
    plt.figure(figsize=(12, 4))  # Adjust the aspect ratio to fit the wide matrix
    sns.heatmap(weights.numpy(), annot=False, cmap='coolwarm', cbar=True, center=0)
    plt.title("Weight Matrix Heatmap")
    plt.xlabel("Input Features (64)")
    plt.ylabel("Output Dimensions (5)")
    plt.xticks(ticks=range(0, weights.size(1), 8), labels=range(0, weights.size(1)))  # Tick every 8th input
    plt.yticks(ticks=range(weights.size(0)), labels=[f"{i}" for i in range(weights.size(0))])
    if savefig_path:
        plt.savefig(savefig_path, bbox_inches="tight")
    plt.show()


# Example usage:
# X = [...]  # Your input data as a list of vectors (num_samples, 16)
# y = [...]  # Your target labels (num_samples,) where values are 0 to 11
# main(X, y)


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# # Assuming your dataset looks like this:
# # X - List of input vectors, each of shape (16,)
# # y - Corresponding class labels from 0 to 11

# # Example data (replace this with your actual data)
# X = [...]  # Shape: (num_samples, 16)
# y = [...]  # Shape: (num_samples,) with values from 0 to 11

# # Convert to torch tensors
# X_tensor = torch.tensor(X, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.long)

# # Split the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)

# # Define the Logistic Regression Model
# class LinearModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearModel, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)  # Linear layer (16 -> 12)

#     def forward(self, x):
#         return self.fc(x)  # No activation, as we will use CrossEntropyLoss which applies softmax internally

# # Initialize the model
# input_dim = 16  # Number of features per input vector
# output_dim = 12  # Number of classes
# model = LinearModel(input_dim, output_dim)

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()  # This loss function applies softmax internally
# optimizer = optim.SGD(model.parameters(), lr=0.01)  # Use SGD optimizer

# # Training the model
# num_epochs = 100  # Number of epochs to train
# batch_size = 32  # Batch size for training

# for epoch in range(num_epochs):
#     model.train()  # Set the model to training mode

#     # Shuffle and iterate over the dataset in batches
#     permutation = torch.randperm(X_train.size(0))
#     for i in range(0, X_train.size(0), batch_size):
#         indices = permutation[i:i+batch_size]
#         batch_X, batch_y = X_train[indices], y_train[indices]

#         # Forward pass
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     # Print loss every 10 epochs
#     if (epoch+1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# # Evaluate the model
# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():
#     outputs = model(X_test)
#     _, predicted = torch.max(outputs, 1)
#     accuracy = (predicted == y_test).sum().item() / y_test.size(0)
#     print(f"Test Accuracy: {accuracy*100:.2f}%")
