"""set_transformer_small.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bvaQwDegrF9u6At559hZo2VhyR0SKcVa

# Fixed Dataset Construction
"""

import os
import torch
from torch import optim
import wandb
from model import GPT
from model import GPTConfig24, GPTConfig42, GPTConfig44, GPTConfig
from data_utils import initialize_datasets, initialize_loaders, plot_attention_heatmap
import random
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def wandb_log(config, avg_train_loss, avg_val_loss, epoch=None):
    print(
        f"Epoch {epoch+1}/{config.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
    )

    wandb.log(
        {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }
    )


# Update accuracy calculation
# TODO: add alternate measure for computing accuracy (set prediction, but wrong order of cards)
@torch.no_grad()
def calculate_accuracy(model, dataloader, config, save_incorrect=False):
    model.eval()
    correct = 0
    total = 0

    for index, sequences in enumerate(dataloader):
        inputs = sequences[:, : config.input_size].to(device)
        targets = sequences[:, config.input_size:].to(device)

        outputs = model.generate(
            inputs, max_new_tokens=config.target_size)
        predictions = outputs[:, config.input_size:]

        mask = targets != config.padding_token  # Create a mask to ignore padding
        matches = ((predictions == targets) | ~mask).all(dim=1)

        # if print_incorrect:
        #     # Print incorrect predictions and corresponding targets
        #     for i in range(len(matches)):
        #         if not matches[i].item():
        #             print(f"Incorrect Prediction, sequence {index}, batch {i}:")
        #             print(f"  Inputs: {inputs[i].cpu().numpy()}")
        #             print(f"  Target: {targets[i].cpu().numpy()}")
        #             print(f"  Prediction: {predictions[i].cpu().numpy()}")
        if save_incorrect:
            with open("incorrect_predictions.txt", "w") as f:
                # Print incorrect predictions and corresponding targets to file
                for i in range(len(matches)):
                    if not matches[i].item():
                        f.write(
                            f"Incorrect Prediction, sequence {index}, batch {i}:\n")
                        f.write(f"  Inputs: {inputs[i].cpu().numpy()}\n")
                        f.write(f"  Target: {targets[i].cpu().numpy()}\n")
                        f.write(
                            f"  Prediction: {predictions[i].cpu().numpy()}\n\n")

        correct += matches.sum().item()
        total += mask.any(dim=1).sum().item()

    return correct / total


@torch.no_grad()
def evaluate_val_loss(
    model,
    val_loader,
    optimizer,
    counter,
    best_val_loss,
    val_losses,
    config,
    epoch=None,
):
    model.eval()
    total_val_loss = 0
    avg_val_loss = 0

    for inputs in val_loader:
        inputs = inputs.to(device)
        _, loss, _ = model(inputs, True)
        total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # if avg_val_loss < best_val_loss or always_save_checkpoint:
    if avg_val_loss < best_val_loss:
        counter = 0
        best_val_loss = avg_val_loss

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch_num": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
        }
        print(f"saving checkpoint to {config.out_dir}")
        torch.save(checkpoint, os.path.join(config.out_dir, config.filename))
    else:
        counter += 1

    return avg_val_loss, best_val_loss, counter


def run(config, load_model=False):

    # dataset = initialize_datasets(config, save_dataset=True)

    dataset = torch.load(
        '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/balanced_set_dataset_random.pth')
    train_loader, val_loader = initialize_loaders(config, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)

    wandb.init(
        project="set-prediction-small",
        config={
                "learning_rate": config.lr,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "patience": config.patience,
                "eval_freq": config.eval_freq,
        },
    )

    if not load_model:

        optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=0.01)

        # Training loop (remains mostly the same)
        train_losses = []
        val_losses = []

        best_val_loss = 1e9

        counter = 0

        for epoch in range(config.epochs):
            if counter >= config.patience:
                break
            model.train()
            total_train_loss = 0
            for inputs in train_loader:  # train_loader, 364 batches (11664/32)
                model.train()
                inputs = inputs.to(device)
                optimizer.zero_grad()
                _, loss, _ = model(inputs, True)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            avg_val_loss, best_val_loss, counter = evaluate_val_loss(
                model,
                val_loader,
                optimizer,
                counter,
                best_val_loss,
                val_losses,
                config,
                epoch=epoch,
            )

            wandb_log(config, avg_train_loss, avg_val_loss, epoch=epoch)

    # Restore the model state dict
    checkpoint = torch.load(os.path.join(
        config.out_dir, config.filename), weights_only=False)
    model.load_state_dict(checkpoint["model"])

    train_accuracy = calculate_accuracy(
        model, train_loader, config)
    val_accuracy = calculate_accuracy(
        model, val_loader, config)

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    wandb.log({"train_accuracy": train_accuracy, "val_accuracy": val_accuracy})

    wandb.finish()


def generate_heatmap(config, dataset_index):
    print(f"Generating heatmap for index {dataset_index}")
    # dataset = initialize_datasets(config, save_dataset=True)

    dataset = torch.load(
        '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/balanced_set_dataset_random.pth')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)
    print("Loaded dataset")

    breakpoint()
    # Restore the model state dict
    checkpoint = torch.load(os.path.join(
        config.out_dir, config.filename), weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print("Loaded model")

    _, _, attention_weights = model(
        dataset[dataset_index].unsqueeze(0).to(device), False)
    print("Got attention weights")

    labels = dataset[dataset_index].tolist()
    print("labels: ", labels)

    layers = range(config.n_layer)
    heads = range(config.n_head)
    for layer in layers:
        for head in heads:
            plot_attention_heatmap(
                attention_weights[layer][0][head],
                labels,
                title=f"Attention Weights: Layer {layer}, Head {head}",
                savefig=f"attention_heatmap_index_{dataset_index}_layer_{layer}_head_{head}.png")


if __name__ == "__main__":
    # small_combinations = run()
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # generate_heatmap(GPTConfig(), 0)
    # generate_heatmap(GPTConfig(), 1)
    # generate_heatmap(GPTConfig(), 2)
    # generate_heatmap(GPTConfig(), 4)
    # run(GPTConfig(), load_model=True)

    # run(GPTConfig24, load_model=False)
    # run(GPTConfig42, load_model=False)
    # run(GPTConfig44, load_model=False)

    dataset = initialize_datasets(GPTConfig(), save_dataset=False, save_tokenizer_path = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/balanced_set_dataset_random_tokenizer.pkl')
