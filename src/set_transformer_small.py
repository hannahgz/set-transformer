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
from model import GPT, GPTConfig
from data_utils import initialize_datasets


device = "cuda" if torch.cuda.is_available() else "cpu"


def wandb_log(avg_train_loss, avg_val_loss, epoch=None):
    print(
        f"Epoch {epoch+1}/{GPTConfig.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
    )

    wandb.log(
        {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }
    )


# Update accuracy calculation
# TODO: check that this is implemented correctly
# TODO: add alternate measure for computing accuracy (set prediction, but wrong order of cards)
@torch.no_grad()
def calculate_accuracy(model, dataloader, padding_token):
    model.eval()
    correct = 0
    total = 0

    for sequences in dataloader:
        inputs = sequences[:, : GPTConfig().input_size].to(device)
        targets = sequences[:, GPTConfig().input_size :].to(device)

        outputs = model.generate(inputs, max_new_tokens=GPTConfig().target_size)
        predictions = outputs[:, GPTConfig().input_size :]

        print("targets: ", targets)
        print("predictions: ", predictions)

        # breakpoint()

        mask = targets != padding_token  # Create a mask to ignore padding
        correct += ((predictions == targets) | ~mask).all(dim=1).sum().item()
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
        _, loss = model(inputs, True)
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


def run(load_model = False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTConfig()
    train_loader, val_loader = initialize_datasets(config)
    model = GPT(config).to(device)
    print("device: ", device)
    breakpoint()

    if not load_model:

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
        
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

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
            for inputs in train_loader: #train_loader, 364 batches (11664/32)
                model.train()
                inputs = inputs.to(device)
                optimizer.zero_grad()
                _, loss = model(inputs, True)
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

            wandb_log(avg_train_loss, avg_val_loss, epoch = epoch)

        
    # # Comment out if not loading already trained model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # checkpoint_path = "zxcv.pt"
    # checkpoint_path = "ckpt_small_patience (1).pt"
    # config = GPTConfig(vocab_size=len(tokenizer.token_to_id))
    # model = GPT(config).to(device)
    # Restore the model state dict
    checkpoint = torch.load(os.path.join(config.out_dir, config.filename))
    model.load_state_dict(checkpoint["model"])

    train_accuracy = calculate_accuracy(model, train_loader)
    val_accuracy = calculate_accuracy(model, val_loader)

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    wandb.log({"train_accuracy": train_accuracy, "val_accuracy": val_accuracy})

    wandb.finish()


if __name__ == "__main__":
    # small_combinations = run()
    run()
