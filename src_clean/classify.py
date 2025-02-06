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

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Linear layer (64 -> 5)

    def forward(self, x):
        # No activation, as we will use CrossEntropyLoss which applies softmax internally
        return self.fc(x)


@dataclass
class LinearProbeBindingCardAttrConfig:
    capture_layer: int = 2
    pred_card_from_attr: bool = True
    model_type: str = "linear"
    input_dim: int = 64
    output_dim: int = 5


def evaluate_model(
        model,
        X,
        y,
        model_name,
        output_dim,
        continuous_to_original_path=None,
        dataset_name=None,
        capture_layer=None,
        tokenizer_path=None):
    """Evaluates the model on data and prints correct predictions."""

    checkpoint = torch.load(
        f'{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}/{model_name}.pt', weights_only=False)
    model.load_state_dict(checkpoint["model"])

    model.eval()  # Set the model to evaluation mode
    correct_predictions = []

    # Dictionary to count frequency of correct predictions for each class
    class_correct_counts = {i: 0 for i in range(output_dim)}
    # Dictionary to count total occurrences of each class
    class_total_counts = {i: 0 for i in range(output_dim)}

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
        log_per_class_statistics(
            continuous_to_original_path,
            tokenizer_path,
            model_name,
            output_dim,
            class_total_counts,
            class_correct_counts,
            dataset_name,
            capture_layer)

    print(f"\nTotal correct predictions: {len(correct_predictions)}")
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


def log_per_class_statistics(
        continuous_to_original_path,
        tokenizer_path,
        model_name,
        output_dim,
        class_total_counts,
        class_correct_counts,
        dataset_name,
        capture_layer):
    with open(continuous_to_original_path, 'rb') as f:
        continuous_to_original = pickle.load(f)
    tokenizer = load_tokenizer(tokenizer_path)
    # Save the printed logs into a txt file
    log_file_path = f'{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}/{model_name}_evaluation_log.txt'
    print("log_file_path: ", log_file_path)
    with open(log_file_path, 'w') as log_file:
        log_file.write("\nPer-class statistics:\n")

        if continuous_to_original_path and tokenizer_path:
            for class_idx in range(output_dim):
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


def train_model(model, train_data, val_data, criterion, optimizer, num_epochs=100, batch_size=32, patience=5, dataset_name=None, capture_layer=None, model_name=None):
    """Trains the model using validation accuracy for early stopping."""
    X_train, y_train = train_data
    X_val, y_val = val_data

    wandb.init(
        project="complete-classify-card",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "patience": patience,
        },
        name=f"{dataset_name}_{model_name}_layer{capture_layer}"
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
            torch.save(
                checkpoint, f'{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}/{model_name}.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break


def run_classify(
        input_dim,
        output_dim,
        capture_layer,
        num_epochs=100,
        batch_size=32,
        lr=0.001,
        model_type="linear",
        tokenizer_path=None,
        pred_card_from_attr=True):
    """Main function to run the model training and evaluation."""

    if pred_card_from_attr:
        dataset_name = "card_from_attr"
    else:
        dataset_name = "attr_from_card"

    input_embeddings_path = f"{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}/input_embeddings.pt"
    mapped_target_tokens_path = f"{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}/continuous_target_tokens.pt"
    continuous_to_original_path = f"{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}/continuous_to_original.pkl"

    X = torch.load(input_embeddings_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.load(mapped_target_tokens_path).to(device)

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
        dataset_name=dataset_name,
        capture_layer=capture_layer,
        model_name=model_type,
    )

    # Evaluate the model

    train_accuracy = evaluate_model(
        model,
        X_train,
        y_train,
        model_name=model_type,
        output_dim=output_dim,
        continuous_to_original_path=continuous_to_original_path,
        dataset_name=dataset_name,
        capture_layer=capture_layer,
        tokenizer_path=tokenizer_path)

    val_accuracy = evaluate_model(
        model,
        X_val,
        y_val,
        model_name=model_type,
        output_dim=output_dim,
        continuous_to_original_path=continuous_to_original_path,
        dataset_name=dataset_name,
        capture_layer=capture_layer,
        tokenizer_path=tokenizer_path)

    test_accuracy = evaluate_model(
        model,
        X_test,
        y_test,
        model_name=model_type,
        output_dim=output_dim,
        continuous_to_original_path=continuous_to_original_path,
        dataset_name=dataset_name,
        capture_layer=capture_layer,
        tokenizer_path=tokenizer_path)

    wandb.log({
        "final_train_accuracy": train_accuracy,
        "final_val_accuracy": val_accuracy,
        "final_test_accuracy": test_accuracy
    })
    wandb.finish()

    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


def analyze_weights(capture_layer, pred_card_from_attr, model_type="linear", input_dim=64, output_dim=5):
    model = LinearModel(input_dim, output_dim)

    if pred_card_from_attr:
        dataset_name = "card_from_attr"
    else:
        dataset_name = "attr_from_card"

    # Load the model's state_dict
    model.load_state_dict(torch.load(
        f'{PATH_PREFIX}/complete/classify/{dataset_name}/layer{capture_layer}/{model_type}.pt')["model"])
    model.eval()

    # Access weights and biases
    weights = model.fc.weight.data
    biases = model.fc.bias.data

    print("Weights shape:", weights.shape)
    print("Weights:\n", weights)
    print("Biases:\n", biases)

    # Iterate through the weight tensor and print the index and the value
    for i in range(weights.size(0)):  # Iterate over the output_dim (rows)
        for j in range(weights.size(1)):  # Iterate over the input_dim (columns)
            print(f"Weight at index ({i}, {j}): {weights[i, j].item()}")

    return weights, biases


def load_model_from_config(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT(config).to(device)
    checkpoint = torch.load(config.filename, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_linear_probe_from_config(config):
    if config.pred_card_from_attr:
        dataset_name = "card_from_attr"
    else:
        dataset_name = "attr_from_card"

    model = LinearModel(config.input_dim, config.output_dim)
    model.load_state_dict(torch.load(
        f'{PATH_PREFIX}/complete/classify/{dataset_name}/layer{config.capture_layer}/{config.model_type}.pt')["model"])
    model.eval()
    return model


def load_continuous_to_original_from_config(config):
    if config.pred_card_from_attr:
        dataset_name = "card_from_attr"
    else:
        dataset_name = "attr_from_card"

    continuous_to_original_path = f"{PATH_PREFIX}/complete/classify/{dataset_name}/layer{config.capture_layer}/continuous_to_original.pkl"
    with open(continuous_to_original_path, 'rb') as f:
        continuous_to_original = pickle.load(f)

    return continuous_to_original


def linear_probe_vector_analysis(model_config, probe_config, input_sequence):
    model = load_model_from_config(model_config)
    probe = load_linear_probe_from_config(probe_config)
    continuous_to_original = load_continuous_to_original_from_config(
        probe_config)
    tokenizer = load_tokenizer(model_config.tokenizer_path)

    print("continuous_to_original: ", continuous_to_original)
    tokenized_input_sequence = torch.tensor(
        tokenizer.encode(input_sequence)).unsqueeze(0)
    print("tokenized_input_sequence shape: ", tokenized_input_sequence.shape)

    # Get embeddings at specific layer
    _, _, _, layer_embedding, _ = model(
        tokenized_input_sequence, capture_layer=probe_config.capture_layer)
    print("layer_embedding shape: ", layer_embedding.shape)

    # Get probe weights
    # Shape: [5, 64] for your 5-class probe
    probe_weights = probe.weight.detach()

    for pos in range(0, input_sequence.shape[1], 2):
        token_embedding = layer_embedding[0, pos, :]  # Shape: [64]
        current_card = input_sequence[pos]
        print("current_card: ", current_card)

        # Compare with each probe dimension
        for probe_dim in range(probe_weights.shape[0]):
            probe_vector = probe_weights[probe_dim]  # Shape: [64]
            probe_dim_card = tokenizer.decode(continuous_to_original[probe_dim])
            print(f"probe dim: {probe_dim}, corresponds to card {probe_dim_card}")

            # Analysis metrics
            cosine_sim = F.cosine_similarity(
                token_embedding.unsqueeze(0), probe_vector.unsqueeze(0))
            dot_product = torch.dot(token_embedding, probe_vector)

            print(f"Position {pos}, Card {current_card}, Probe dim {probe_dim}, Probe dim card {probe_dim_card}:")
            print(f"Cosine similarity: {cosine_sim:.3f}")
            print(f"Dot product: {dot_product:.3f}")
            breakpoint()


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
        # input_embeddings.shape, torch.Size([512, 20, 64]), [batch size, sequence length, embedding size]
        input_embeddings = captured_embedding[:, input_start_index:(
            config.input_size-1):2, :]
        # flattened_input_embeddings.shape, torch.Size([10240, 64]), [batch size * sequence length, embedding size]
        flattened_input_embeddings = input_embeddings.reshape(-1, 64)

        # Get every other token in the input starting from index target_start_index, representing either all the card tokens or attribute tokens
        # target_tokens.shape, torch.Size([512, 20])
        target_tokens = batch[:, target_start_index:(config.input_size - 1):2]
        # flattened_target_tokens.shape, torch.Size([10240])
        flattened_target_tokens = target_tokens.reshape(-1)

        # Append the flattened tensors to the respective lists
        all_flattened_input_embeddings.append(flattened_input_embeddings)
        all_flattened_target_tokens.append(flattened_target_tokens)

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

    input_sequence = [
        "B", "oval", "C", "green", "B", "one", "B", "green", "D", "one", "E", "green", "E", "one",
        "A", "green", "A", "one", "D", "open", "C", "one", "A", "solid", "D", "oval", "B", "striped",
        "D", "green", "A", "oval", "E", "diamond", "C", "solid", "C", "squiggle", "E", "solid",
        "A", "B", "D", "A", "C", "E", "."
    ]

    linear_probe_vector_analysis(
        model_config=config,
        probe_config=LinearProbeBindingCardAttrConfig(),
        input_sequence=input_sequence
    )
    
    # analyze_weights(
    #     capture_layer=1,
    #     pred_card_from_attr=True,
    #     model_type="linear",
    #     input_dim=64,
    #     output_dim=5)

    # capture_layer = 2

    # init_card_attr_binding_dataset(
    #     config=config,
    #     capture_layer=capture_layer,
    #     pred_card_from_attr=False)

    # run_classify(
    #     input_dim=64,
    #     output_dim=5,
    #     capture_layer=capture_layer,
    #     num_epochs=5,
    #     batch_size=32,
    #     lr=0.001,
    #     model_type="linear",
    #     tokenizer_path=config.tokenizer_path,
    #     pred_card_from_attr=True)

    # pred_card_from_attr = True
    # for capture_layer in [0, 1, 3]:
    #     print(f"Predicting card from attribute, capture layer: {capture_layer}")
    #     init_card_attr_binding_dataset(
    #         config=config,
    #         capture_layer=capture_layer,
    #         pred_card_from_attr=pred_card_from_attr)

    #     run_classify(
    #         input_dim=64,
    #         output_dim=5,
    #         capture_layer=capture_layer,
    #         num_epochs=5,
    #         batch_size=32,
    #         lr=0.001,
    #         model_type="linear",
    #         tokenizer_path=config.tokenizer_path,
    #         pred_card_from_attr=pred_card_from_attr)

    # pred_card_from_attr = False

    # run_classify(
    #     input_dim=64,
    #     output_dim=12,
    #     capture_layer=0,
    #     num_epochs=5,
    #     batch_size=32,
    #     lr=0.001,
    #     model_type="linear",
    #     tokenizer_path=config.tokenizer_path,
    #     pred_card_from_attr=pred_card_from_attr)

    # for capture_layer in range(1,4):
    #     print(f"Predicting attribute from card, capture layer: {capture_layer}")
    #     init_card_attr_binding_dataset(
    #         config=config,
    #         capture_layer=capture_layer,
    #         pred_card_from_attr=pred_card_from_attr)

    #     run_classify(
    #         input_dim=64,
    #         output_dim=12,
    #         capture_layer=capture_layer,
    #         num_epochs=5,
    #         batch_size=32,
    #         lr=0.001,
    #         model_type="linear",
    #         tokenizer_path=config.tokenizer_path,
    #         pred_card_from_attr=pred_card_from_attr)
