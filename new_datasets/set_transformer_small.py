import os
import torch
from torch import optim
import wandb
from model import GPT
from model import add_causal_masking
from data_utils import initialize_loaders, initialize_triples_datasets
import random
import numpy as np
from tokenizer import load_tokenizer
import pickle
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

device = "cuda" if torch.cuda.is_available() else "cpu"

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

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


def categorize_predict(target_num_sets, set_accuracy_dict, prediction_np, two_set_id, zero_set_id):
    if zero_set_id in prediction_np:
        set_accuracy_dict[target_num_sets]["incorrect"][0] += 1
    elif two_set_id in prediction_np:
        set_accuracy_dict[target_num_sets]["incorrect"][2] += 1
    else:
        set_accuracy_dict[target_num_sets]["incorrect"][1] += 1
    
# Update accuracy calculation
# TODO: add alternate measure for computing accuracy (set prediction, but wrong order of cards)
@torch.no_grad()
def calculate_accuracy(model, dataloader, config, tokenizer_path=None, save_incorrect_path=None, breakdown=False):
    print("Calculating accuracy")
    model.eval()
    correct = 0
    total = 0

    set_accuracy_dict = {
        0: {
            "incorrect": {
                0: 0,
                1: 0,
                2: 0
            },
            "total": 0},
        1: {
            "incorrect": {
                0: 0,
                1: 0,
                2: 0
            },
            "total": 0},
        2: {
            "incorrect": {
                0: 0,
                1: 0,
                2: 0
            },
            "total": 0},
    }

    for index, sequences in enumerate(dataloader):
        if index % 100 == 0:
            print(f"Input: {index}/{len(dataloader)}")
            
        inputs = sequences[:, :config.input_size].to(device)
        targets = sequences[:, config.input_size:].to(device)

        # Add autocast for the forward pass (model.generate)
        with autocast():
            outputs = model.generate(
                inputs, max_new_tokens=config.target_size)
            
        predictions = outputs[:, config.input_size:]

        mask = targets != config.padding_token  # Create a mask to ignore padding
        matches = ((predictions == targets) | ~mask).all(dim=1)

        if save_incorrect_path:
            with open(save_incorrect_path, "a") as f:
                tokenizer = load_tokenizer(tokenizer_path)
                two_set_id = tokenizer.token_to_id["/"]
                for i in range(len(matches)):
                    target_np = targets[i].cpu().numpy()
                    if not matches[i].item():
                        f.write(
                            f"Incorrect Prediction, sequence {index}, batch {i}:\n")
                        f.write(
                            f"  Inputs: {tokenizer.decode(inputs[i].cpu().numpy())}\n")
                        f.write(
                            f"  Target: {tokenizer.decode(targets[i].cpu().numpy())}\n")
                        f.write(
                            f"  Prediction: {tokenizer.decode(predictions[i].cpu().numpy())}\n\n")
                    # else:
                    #     if two_set_id in target_np:
                    #         target_np = targets[i].cpu().numpy()
                    #         print("Correct prediction")
                    #         print("Inputs: ", tokenizer.decode(inputs[i].cpu().numpy()))
                    #         print("Target: ", tokenizer.decode(targets[i].cpu().numpy()))
                    #         print("Prediction: ", tokenizer.decode(predictions[i].cpu().numpy()))
                        
        if breakdown:
            tokenizer = load_tokenizer(tokenizer_path)

            two_set_id = tokenizer.token_to_id["/"]
            zero_set_id = tokenizer.token_to_id["*"]

            for i in range(len(matches)):
                target_np = targets[i].cpu().numpy()
                prediction_np = predictions[i].cpu().numpy()
                if not matches[i].item():
                    if two_set_id in target_np:
                        categorize_predict(
                            target_num_sets=2, 
                            set_accuracy_dict=set_accuracy_dict, 
                            prediction_np=prediction_np,
                            two_set_id=two_set_id,
                            zero_set_id=zero_set_id)
                    elif zero_set_id in target_np:
                        categorize_predict(
                            target_num_sets=0, 
                            set_accuracy_dict=set_accuracy_dict, 
                            prediction_np=prediction_np,
                            two_set_id=two_set_id,
                            zero_set_id=zero_set_id)
                    else:
                        categorize_predict(
                            target_num_sets=1, 
                            set_accuracy_dict=set_accuracy_dict, 
                            prediction_np=prediction_np,
                            two_set_id=two_set_id,
                            zero_set_id=zero_set_id)
                
                if two_set_id in target_np:
                    set_accuracy_dict[2]["total"] += 1
                elif zero_set_id in target_np:
                    set_accuracy_dict[0]["total"] += 1
                else:
                    set_accuracy_dict[1]["total"] += 1


        correct += matches.sum().item()
        total += mask.any(dim=1).sum().item()

        if index % 100 == 0:
            print("Accuracy: ", correct / total)
            if breakdown:
                print("Set Accuracy Dict: ", set_accuracy_dict)
                for i in range(3):
                    total_incorrect = sum(set_accuracy_dict[i]["incorrect"].values())
                    if set_accuracy_dict[i]["total"] != 0:
                        print("Percentage of incorrect predictions for set size", i, ": ", total_incorrect / set_accuracy_dict[i]["total"])
                        for j in range(3):
                            if total_incorrect != 0:
                                print("\t Predicted incorrectly with set size", j, ": ", set_accuracy_dict[i]["incorrect"][j] / total_incorrect)
                

    return correct / total


@torch.no_grad()
def evaluate_val_loss(model, val_loader, optimizer, best_val_loss, val_losses, config, epoch):
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(device)
            with autocast():
                _, loss, _, _, _ = model(inputs, True)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch_num": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
        }
        
        # torch.save(checkpoint, f"{PATH_PREFIX}/{config.filename}")
        torch.save(checkpoint, config.filename)

    return avg_val_loss, best_val_loss


def model_accuracy(config, model, train_loader, val_loader):
     # Restore the model state dict
    checkpoint = torch.load(f"{PATH_PREFIX}/{config.filename}", weights_only=False)
    model.load_state_dict(checkpoint["model"])

    train_accuracy = calculate_accuracy(
        model, train_loader, config)
    val_accuracy = calculate_accuracy(
        model, val_loader, config)
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    return train_accuracy, val_accuracy

def run(config, dataset_path, load_model=False, should_wandb_log=True):
    dataset = torch.load(dataset_path)
    train_loader, val_loader = initialize_loaders(config, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)

    if should_wandb_log:
        wandb.init(
            project="equal-set-prediction-balanced",
            config={
                    "learning_rate": config.lr,
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "n_layer": config.n_layer,
                    "n_head": config.n_head,
                    "n_embd": config.n_embd,
                    "patience": config.patience,
            },
            name=config.filename
        )

    if not load_model:

        optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=0.01)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,  # Reduce LR by half when plateauing
            patience=2,   # Wait 2 epochs before reducing LR
            min_lr=1e-6,
            verbose=True  # Print when LR changes
        )

        scaler = GradScaler()
        
        train_losses = []
        val_losses = []

        best_val_loss = 1e9

        counter = 0

        for epoch in range(config.epochs):
            if counter >= config.patience:
                break
            
            model.train()
            total_train_loss = 0
            
            for index, inputs in enumerate(train_loader):
                if index % 10000 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch: {epoch + 1}/{config.epochs}, Batch: {index}/{len(train_loader)}, LR: {current_lr:.2e}")
                
                inputs = inputs.to(device)
                optimizer.zero_grad()
                
                # Automatic mixed precision
                with autocast():
                    _, loss, _, _, _ = model(inputs, True)
                
                # Scale the loss and call backward
                scaler.scale(loss).backward()
                
                # Unscale gradients and clip them
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Step optimizer and update scaler
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step()
                
                total_train_loss += loss.item()

                # Evaluate every eval_freq steps
                if index > 0 and index % config.eval_freq == 0:
                    current_avg_train_loss = total_train_loss / index
                    
                    print("Evaluating validation loss")
                    avg_val_loss, best_val_loss = evaluate_val_loss(
                        model,
                        val_loader,
                        optimizer,
                        best_val_loss,
                        val_losses,
                        config,
                        epoch=epoch,
                    )
                    
                    # Step the scheduler with validation loss
                    scheduler.step(avg_val_loss)

                    if scheduler.num_bad_epochs >= scheduler.patience:
                        counter += 1
                    else:
                        counter = 0
                    
                    # Log metrics
                    current_lr = optimizer.param_groups[0]['lr']
                    wandb.log({
                        "step": index + epoch * len(train_loader),
                        "train_loss": current_avg_train_loss,
                        "val_loss": avg_val_loss,
                        "learning_rate": current_lr
                    })
                    print(
                        f"Epoch {epoch+1}/{config.epochs}, Step: {index + epoch * len(train_loader)}, Train Loss: {current_avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                    )
                    # Return to training mode
                    model.train()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

    print("Calculating accuracy")
    train_accuracy, val_accuracy = model_accuracy(config, model, train_loader, val_loader)

    if should_wandb_log:
        wandb.log({"train_accuracy": train_accuracy, "val_accuracy": val_accuracy})
        wandb.finish()


def analyze_embeddings(config, dataset_name, model_path, capture_layer):
    dataset_path = f"{PATH_PREFIX}/{dataset_name}.pth"
    dataset = torch.load(dataset_path)
    train_loader, val_loader = initialize_loaders(config, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = GPT(config).to(device)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    all_flattened_input_embeddings = []
    all_flattened_target_cards = []

    for index, batch in enumerate(val_loader):
        print(f"Batch {index + 1}/{len(val_loader)}")
        batch = batch.to(device)
        _, _, _, captured_embedding, _ = model(batch, True, capture_layer)

        input_embeddings = captured_embedding[:, 1:(config.input_size-1):2, :]
        flattened_input_embeddings = input_embeddings.reshape(-1, 64)

        # Get every other element in the input starting from index 0, representing all the card tokens
        target_cards = batch[:, :(config.input_size - 1):2]
        flattened_target_cards = target_cards.reshape(-1)

        # Append the flattened tensors to the respective lists
        all_flattened_input_embeddings.append(flattened_input_embeddings)
        all_flattened_target_cards.append(flattened_target_cards)
    
    combined_input_embeddings = torch.cat(all_flattened_input_embeddings, dim=0)
    combined_target_cards = torch.cat(all_flattened_target_cards, dim=0)

    # Get the unique values in combined_target_attributes
    unique_values = torch.unique(combined_target_cards)

    # Create a mapping from unique values to a continuous sequence
    value_to_continuous = {v.item(): i for i, v in enumerate(unique_values)}

    # Reverse the mapping: continuous values back to the original values
    continuous_to_original = {v: k for k, v in value_to_continuous.items()}

    # Map the values in combined_target_attributes to the continuous sequence
    mapped_target_cards = torch.tensor([value_to_continuous[val.item()] for val in combined_target_cards])

    # Create the directory structure if it doesn't exist
    base_dir = f"{PATH_PREFIX}/classify/{dataset_name}/layer{capture_layer}"
    os.makedirs(base_dir, exist_ok=True)

    embeddings_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{capture_layer}/real_model_input_embeddings.pt"
    mapped_cards_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{capture_layer}/real_model_mapped_target_attributes.pt"
    continuous_to_original_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{capture_layer}/real_model_continuous_to_original.pkl"

    # Save the combined_input_embeddings tensor
    torch.save(combined_input_embeddings, embeddings_path)

    # Save the mapped_target_attributes tensor
    torch.save(mapped_target_cards, mapped_cards_path)

    with open(continuous_to_original_path, "wb") as f:
        pickle.dump(continuous_to_original, f)

    return combined_input_embeddings, mapped_target_cards, continuous_to_original


def opp_analyze_embeddings(config, dataset_name, capture_layer):
    # TODO: this needs to be fixed to load the actual model in
    dataset_path = f"{PATH_PREFIX}/{dataset_name}.pth"
    dataset = torch.load(dataset_path)
    train_loader, val_loader = initialize_loaders(config, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)

    all_flattened_input_embeddings = []
    all_flattened_target_attributes = []

    for index, batch in enumerate(val_loader):
        batch = batch.to(device)
        _, _, _, captured_embedding, _ = model(batch, True, capture_layer)

        # input_embeddings = captured_embedding[:, 1:(config.input_size-1):2, :]
        input_embeddings = captured_embedding[:, :(config.input_size-1):2, :]
        flattened_input_embeddings = input_embeddings.reshape(-1, 64)

        # Get every other element in the input starting from index 1, representing all the card tokens
        target_attributes = batch[:, 1:(config.input_size - 1):2]
        flattened_target_attributes = target_attributes.reshape(-1)

        # Append the flattened tensors to the respective lists
        all_flattened_input_embeddings.append(flattened_input_embeddings)
        all_flattened_target_attributes.append(flattened_target_attributes)
    
    combined_input_embeddings = torch.cat(all_flattened_input_embeddings, dim=0)
    combined_target_attributes = torch.cat(all_flattened_target_attributes, dim=0)

    # Get the unique values in combined_target_attributes
    unique_values = torch.unique(combined_target_attributes)

    # Create a mapping from unique values to a continuous sequence
    value_to_continuous = {v.item(): i for i, v in enumerate(unique_values)}

    # Reverse the mapping: continuous values back to the original values
    continuous_to_original = {v: k for k, v in value_to_continuous.items()}

    # Map the values in combined_target_attributes to the continuous sequence
    mapped_target_attributes = torch.tensor([value_to_continuous[val.item()] for val in combined_target_attributes])

    # Create the directory structure if it doesn't exist
    base_dir = f"{PATH_PREFIX}/classify/opp_{dataset_name}/layer{capture_layer}"
    os.makedirs(base_dir, exist_ok=True)

    embeddings_path = f"{PATH_PREFIX}/classify/opp_{dataset_name}/layer{capture_layer}/input_embeddings.pt"
    mapped_attributes_path = f"{PATH_PREFIX}/classify/opp_{dataset_name}/layer{capture_layer}/mapped_target_attributes.pt"
    continuous_to_original_path = f"{PATH_PREFIX}/classify/opp_{dataset_name}/layer{capture_layer}/continuous_to_original.pkl"

    # Save the combined_input_embeddings tensor
    torch.save(combined_input_embeddings, embeddings_path)

    # Save the mapped_target_attributes tensor
    torch.save(mapped_target_attributes, mapped_attributes_path)

    with open(continuous_to_original_path, "wb") as f:
        pickle.dump(continuous_to_original, f)

    return combined_input_embeddings, mapped_target_attributes, continuous_to_original

    
def get_raw_input_embeddings(config, dataset_name, capture_layer):
    dataset_path = f"{PATH_PREFIX}/{dataset_name}.pth"
    dataset = torch.load(dataset_path)
    train_loader, val_loader = initialize_loaders(config, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)

    all_flattened_embeddings = []

    for index, batch in enumerate(val_loader):
        batch = batch.to(device)
        _, _, _, captured_embedding, _ = model(batch, True, capture_layer)
        # torch.Size([64, 49, 64])
        # [batch_size, seq_len, embedding_dim]

        flattened_embeddings = captured_embedding.reshape(-1, 64)
        # torch.Size([3136, 64])

        all_flattened_embeddings.append(flattened_embeddings)

    base_dir = f"{PATH_PREFIX}/classify/{dataset_name}/layer{capture_layer}"
    os.makedirs(base_dir, exist_ok=True)

    embeddings_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{capture_layer}/all_val_raw_embeddings.pt"
    
    final_embeddings = torch.cat(all_flattened_embeddings)

    torch.save(final_embeddings, embeddings_path)

    return final_embeddings
