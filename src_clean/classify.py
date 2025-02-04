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

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Linear layer (64 -> 5)

    def forward(self, x):
        # No activation, as we will use CrossEntropyLoss which applies softmax internally
        return self.fc(x)


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
        log_per_class_statistics(continuous_to_original_path, tokenizer_path,
                                 model_name, predict_dim, class_total_counts, class_correct_counts)

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
                print(
                    f"  Accuracy: {accuracy_per_class:.4f} ({correct}/{total})")

                log_file.write(
                    f"\nClass {class_idx}, Original Token: ({original_token})/{tokenizer.decode([original_token])}:\n")
                log_file.write(
                    f"  Accuracy: {accuracy_per_class:.4f} ({correct}/{total})\n")


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
    # elif model_type == "mlp":
    #     model = MLPModel(input_dim=input_dim, hidden_dim=32,
    #                      output_dim=output_dim).to(device)
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
    model.load_state_dict(torch.load(
        f"{PATH_PREFIX}/classify/{model_path}.pt")["model"])
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


def map_non_continuous_vals_to_continuous(data):
    # Get the unique values in input data
    unique_values = torch.unique(data)

    # Create a mapping from unique values to a continuous sequence
    value_to_continuous = {v.item(): i for i, v in enumerate(unique_values)}

    # Reverse the mapping: continuous values back to the original values
    continuous_to_original = {v: k for k, v in value_to_continuous.items()}

    # Map the values in combined_target_tokens (data) to the continuous sequence
    mapped_target_to_continuous = torch.tensor(
        [value_to_continuous[val.item()] for val in data])

    return mapped_target_to_continuous, continuous_to_original


def init_card_attr_binding_dataset(config, capture_layer, pred_card_from_attr=True):
    dataset = torch.load(config.dataset_path)
    _, val_loader = initialize_loaders(config, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT(config).to(device)
    checkpoint = torch.load(config.filename, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    all_flattened_input_embeddings = []
    all_flattened_target_tokens = []

    if pred_card_from_attr:
        print("Predicting card from attribute")
        dataset_name = "card_from_attr"
    else:
        print("Predicting attribute from card")
        dataset_name = "attr_from_card"

    for index, batch in enumerate(val_loader):
        print(f"Batch {index + 1}/{len(val_loader)}")
        batch = batch.to(device)
        _, _, _, captured_embedding = model(batch, True, capture_layer)

        if pred_card_from_attr:
            input_start_index = 1
            target_start_index = 0
        else:
            input_start_index = 0
            target_start_index = 1

        # Get every other embedding in the input sequence starting from input_start_index, representing either all attribute or card embeddings
        input_embeddings = captured_embedding[:, input_start_index:(
            config.input_size-1):2, :]
        flattened_input_embeddings = input_embeddings.reshape(-1, 64)

        # Get every other token in the input starting from index target_start_index, representing either all the card tokens or attribute tokens
        target_tokens = batch[:, target_start_index:(config.input_size - 1):2]
        flattened_target_tokens = target_tokens.reshape(-1)

        # Append the flattened tensors to the respective lists
        all_flattened_input_embeddings.append(flattened_input_embeddings)
        all_flattened_target_tokens.append(flattened_target_tokens)
        breakpoint()

    combined_input_embeddings = torch.cat(
        all_flattened_input_embeddings, dim=0)

    combined_target_tokens = torch.cat(all_flattened_target_tokens, dim=0)
    mapped_target_tokens, continuous_to_original = map_non_continuous_vals_to_continuous(
        combined_target_tokens)

    # Create the directory structure if it doesn't exist
    base_dir = f"{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}"
    os.makedirs(base_dir, exist_ok=True)

    input_embeddings_path = f"{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}/input_embeddings.pt"
    mapped_target_tokens_path = f"{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}/continuous_target_tokens.pt"
    continuous_to_original_path = f"{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}/continuous_to_original.pkl"

    # Save the combined_input_embeddings tensor
    torch.save(combined_input_embeddings, input_embeddings_path)

    # Save the mapped_target_attributes tensor
    torch.save(mapped_target_tokens, mapped_target_tokens_path)

    with open(continuous_to_original_path, "wb") as f:
        pickle.dump(continuous_to_original, f)

    breakpoint()

    return combined_input_embeddings, mapped_target_tokens, continuous_to_original


def plot_weights_as_heatmap(weights, savefig_path=None):
    """
    Visualizes the weight matrix for a single layer as a heatmap.

    Parameters:
    weights (torch.Tensor): The weight matrix of shape [output_dim, input_dim].
    """
    plt.figure(figsize=(12, 4))  # Adjust the aspect ratio to fit the wide matrix
    sns.heatmap(weights.numpy(), annot=False,
                cmap='coolwarm', cbar=True, center=0)
    plt.title("Weight Matrix Heatmap")
    plt.xlabel("Input Features (64)")
    plt.ylabel("Output Dimensions (5)")
    plt.xticks(ticks=range(0, weights.size(1)), labels=range(
        0, weights.size(1)), fontsize=5, ha="center")  # Tick every 8th input
    plt.yticks(ticks=range(weights.size(0)), labels=[
               f"{i}" for i in range(weights.size(0))], rotation=0, ha="center")
    if savefig_path:
        plt.savefig(savefig_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    config = GPTConfig44_Complete()
    capture_layer = 2
    init_card_attr_binding_dataset(
        config=config,
        capture_layer=capture_layer,
        pred_card_from_attr=True)
