"""set_transformer_small.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bvaQwDegrF9u6At559hZo2VhyR0SKcVa
"""

import os
import torch
from torch import optim
import wandb
from model import GPT
# from model import GPTConfig24, GPTConfig42, GPTConfig44, GPTConfig, add_causal_masking, GPTConfig48, GPTConfig44_Patience20, GPTConfig44_AttrFirst
from model import GPTConfig44, GPTConfig44TriplesEmbdDrop, GPTConfig44_AttrFirst
from data_utils import initialize_datasets, initialize_loaders, initialize_triples_datasets
import random
import numpy as np
from tokenizer import load_tokenizer
from graph import lineplot_specific
import pickle
from classify import LinearModel, evaluate_model, run_binary_classify, prepare_data
from sklearn.model_selection import train_test_split
from dimension_reduce import run_pca_analysis, run_umap_analysis

from classify import run_classify

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


# Update accuracy calculation
# TODO: add alternate measure for computing accuracy (set prediction, but wrong order of cards)
@torch.no_grad()
def calculate_accuracy(model, dataloader, config, save_incorrect_path=None):
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
        if save_incorrect_path:
            with open(save_incorrect_path, "a") as f:
                # Print incorrect predictions and corresponding targets to file
                tokenizer = load_tokenizer(
                    '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/balanced_set_dataset_random_tokenizer.pkl')
                for i in range(len(matches)):
                    if not matches[i].item():
                        f.write(
                            f"Incorrect Prediction, sequence {index}, batch {i}:\n")
                        f.write(
                            f"  Inputs: {tokenizer.decode(inputs[i].cpu().numpy())}\n")
                        f.write(
                            f"  Target: {tokenizer.decode(targets[i].cpu().numpy())}\n")
                        f.write(
                            f"  Prediction: {tokenizer.decode(predictions[i].cpu().numpy())}\n\n")

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
        _, loss, _, _ = model(inputs, True)
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



def model_accuracy(config, model, train_loader, val_loader):
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

    return train_accuracy, val_accuracy

def run(config, dataset_path, load_model=False, should_wandb_log=True):
    dataset = torch.load(dataset_path)
    train_loader, val_loader = initialize_loaders(config, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)

    if should_wandb_log:
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
            },
            name=config.filename
        )

    if not load_model:

        optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=0.01)

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
                _, loss, _, _ = model(inputs, True)
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

    train_accuracy, val_accuracy = model_accuracy(config, model, train_loader, val_loader)

    if should_wandb_log:
        wandb.log({"train_accuracy": train_accuracy, "val_accuracy": val_accuracy})
        wandb.finish()


def analyze_embeddings(config, dataset_name, capture_layer):
    dataset_path = f"{PATH_PREFIX}/{dataset_name}.pth"
    dataset = torch.load(dataset_path)
    train_loader, val_loader = initialize_loaders(config, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)

    all_flattened_input_embeddings = []
    all_flattened_target_attributes = []

    for index, batch in enumerate(val_loader):
        batch = batch.to(device)
        _, _, _, captured_embedding = model(batch, True, capture_layer)

        input_embeddings = captured_embedding[:, 1:(config.input_size-1):2, :]
        flattened_input_embeddings = input_embeddings.reshape(-1, 64)

        # Get every other element in the input starting from index 0, representing all the card tokens
        target_attributes = batch[:, :(config.input_size - 1):2]
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
    base_dir = f"{PATH_PREFIX}/classify/{dataset_name}/layer{capture_layer}"
    os.makedirs(base_dir, exist_ok=True)

    embeddings_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{capture_layer}/input_embeddings.pt"
    mapped_attributes_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{capture_layer}/mapped_target_attributes.pt"
    continuous_to_original_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{capture_layer}/continuous_to_original.pkl"

    # Save the combined_input_embeddings tensor
    torch.save(combined_input_embeddings, embeddings_path)

    # Save the mapped_target_attributes tensor
    torch.save(mapped_target_attributes, mapped_attributes_path)

    with open(continuous_to_original_path, "wb") as f:
        pickle.dump(continuous_to_original, f)

    return combined_input_embeddings, mapped_target_attributes, continuous_to_original


def opp_analyze_embeddings(config, dataset_name, capture_layer):
    dataset_path = f"{PATH_PREFIX}/{dataset_name}.pth"
    dataset = torch.load(dataset_path)
    train_loader, val_loader = initialize_loaders(config, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)

    all_flattened_input_embeddings = []
    all_flattened_target_attributes = []

    for index, batch in enumerate(val_loader):
        batch = batch.to(device)
        _, _, _, captured_embedding = model(batch, True, capture_layer)

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
        _, _, _, captured_embedding = model(batch, True, capture_layer)
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

if __name__ == "__main__":
    # small_combinations = run()
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # dataset_name = "attr_first_balanced_set_dataset_random"
    # config = GPTConfig44_AttrFirst

    # for layer in range(4):
    #     embeddings_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{layer}/input_embeddings.pt"
    #     mapped_attributes_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{layer}/mapped_target_attributes.pt"

    #     X = torch.load(embeddings_path)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     y = torch.load(mapped_attributes_path).to(device)

    #     run_classify(X, y, model_name=f"{dataset_name}_layer{layer}", input_dim=64, output_dim=12)
    #     run_classify(X, y, model_name=f"{dataset_name}_layer{layer}", input_dim=64, output_dim=12, model_type="mlp")

    dataset_name = "attr_first_balanced_set_dataset_random"
    config = GPTConfig44_AttrFirst

    layer = 0
    embeddings_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{layer}/input_embeddings.pt"
    mapped_attributes_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{layer}/mapped_target_attributes.pt"
    continuous_to_original_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{layer}/continuous_to_original.pkl"
    tokenizer_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/attr_first_balanced_set_dataset_random_tokenizer.pkl'
    with open(continuous_to_original_path, 'rb') as f:
        continuous_to_original = pickle.load(f)

    X = torch.load(embeddings_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.load(mapped_attributes_path).to(device)

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(X, y)

    model = LinearModel(input_dim=64, output_dim=12).to(device)
    tokenizer = load_tokenizer(tokenizer_path)
    evaluate_model(model, X_test, y_test, model_name=f"{dataset_name}_layer{layer}_linear", continuous_to_original=continuous_to_original, tokenizer=tokenizer)

    # dataset_name = "balanced_set_dataset_random"
    # config = GPTConfig44

    # for layer in range(3, 4):
    #     embeddings_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{layer}/input_embeddings.pt"
    #     mapped_attributes_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{layer}/mapped_target_attributes.pt"

    #     X = torch.load(embeddings_path)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     y = torch.load(mapped_attributes_path).to(device)

    #     run_classify(X, y, model_name=f"{dataset_name}_layer{layer}", input_dim=64, output_dim=5)
    #     run_classify(X, y, model_name=f"{dataset_name}_layer{layer}", input_dim=64, output_dim=5, model_type="mlp")

    # dataset_name = "balanced_set_dataset_random"
    # for layer in range(4):
    #     print(f"Layer {layer}")
    #     get_raw_input_embeddings(GPTConfig44, dataset_name, capture_layer=layer)

    # for layer in range(4):
    #     embeddings_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{layer}/all_val_raw_embeddings.pt"
    #     X = torch.load(embeddings_path)
    #     for i in X:


    # dataset_name = "balanced_set_dataset_random"
    # get_raw_input_embeddings(GPTConfig44, dataset_name, capture_layer=0)

    # dataset_name = "balanced_set_dataset_random"
    # for layer in range(4):
    #     embeddings_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{layer}/input_embeddings.pt"
    #     mapped_attributes_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{layer}/mapped_target_attributes.pt"

    #     X = torch.load(embeddings_path)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     y = torch.load(mapped_attributes_path).to(device)

    #     # run_classify(X, y, model_name=f"{dataset_name}_layer{layer}", input_dim=64, output_dim=5)
    #     run_classify(X, y, model_name=f"{dataset_name}_layer{layer}", input_dim=64, output_dim=5, model_type="mlp")


    # for layer in range(0, 4):
    #     dataset_name = "balanced_set_dataset_random"
    #     embeddings_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{layer}/input_embeddings.pt"
    #     mapped_attributes_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{layer}/mapped_target_attributes.pt"

    #     X = torch.load(embeddings_path)
    #     y = torch.load(mapped_attributes_path)

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     y = y.to(device)

    #     # run_pca_analysis(X, y, layer)
    #     run_umap_analysis(X, y, layer)



    # layer = 0
    # dataset_name = "balanced_set_dataset_random"
    # embeddings_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{layer}/input_embeddings.pt"
    # mapped_attributes_path = f"{PATH_PREFIX}/classify/{dataset_name}/layer{layer}/mapped_target_attributes.pt"
    # X = torch.load(embeddings_path)
    # y = torch.load(mapped_attributes_path)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # y = y.to(device)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model = LinearModel(input_dim=64, output_dim=5).to(device)

    # evaluate_model(model, X_train, y_train, model_name=f"{dataset_name}_layer{layer}_linear")
    # evaluate_model(model, X_test, y_test, model_name=f"{dataset_name}_layer{layer}_linear")

    # embeddings_path = f"{PATH_PREFIX}/classify/full_combined_input_embeddings.pt"
    # mapped_attributes_path = f"{PATH_PREFIX}/classify/full_mapped_target_attributes.pt"
    # continuous_to_original_path = f"{PATH_PREFIX}/classify/full_continuous_to_original.pkl"

    # X = torch.load(embeddings_path)
    # y = torch.load(mapped_attributes_path)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # y = y.to(device)

    # # # run_classify(X, y, model_name="full_test.pt", input_dim=64, output_dim=5)
    # # run_classify(X, y, model_name="full_mlp.pt", input_dim=64, output_dim=5, model_type="mlp")

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # model = LinearModel(input_dim=64, output_dim=5).to(device)
    # checkpoint = torch.load(f'{PATH_PREFIX}/classify/full_test.pt')
    # model.load_state_dict(checkpoint["model"])

    # evaluate_model(model, X_test, y_test)



    # run_classify(X, y, model_name="mlp_adam.pt", model_type-"mlp")
    # run_classify(X, y, model_name="adam_lr_0.001.pt")
    # run_classify(X, y, batch_size=64, model_name="adam_batch_size_64.pt")
    # run_classify(X, y, lr=0.01, model_name="adam_reg.pt")

    # dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/balanced_set_dataset_random.pth'


    # dataset_name = "attr_first_balanced_set_dataset_random"
    # config = GPTConfig44_AttrFirst

    # for layer in range(4):
    #     print(f"Layer {layer}")
    #     analyze_embeddings(config, dataset_name, capture_layer=layer)


    # dataset_name = "attr_first_balanced_set_dataset_random"
    # config = GPTConfig44_AttrFirst

    # for layer in range(4):
    #     print(f"Layer {layer}")
    #     analyze_embeddings(config, dataset_name, capture_layer=layer)
    
    # run(
    #     GPTConfig44TriplesEmbdDrop,
    #     dataset_path=f'{PATH_PREFIX}/triples_balanced_set_dataset_random.pth'
    # )

    # dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/triples_balanced_set_dataset_random.pth',
    # dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/balanced_set_dataset_random.pth'
    # config = GPTConfig44Triples

    # dataset = torch.load(dataset_path)
    # train_loader, val_loader = initialize_loaders(config, dataset)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = GPT(config).to(device)

    # model_accuracy(config, model, train_loader, val_loader)
    
    # run(
    #     GPTConfig44Triples,
    #     dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/triples_balanced_set_dataset_random.pth',
    #     load_model=True,
    #     wandb_log=False
    # )

    # generate_heatmap(
    #     GPTConfig44Triples,
    #     [0, 1, 2],
    #     dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/triples_balanced_set_dataset_random.pth',
    #     tokenizer_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/triples_balanced_set_dataset_random_tokenizer.pkl',
    #     use_labels=True,
    #     threshold=0.05,
    #     get_prediction=True)

    # run(
    #     GPTConfig48Triples,
    #     dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/triples_balanced_set_dataset_random.pth'
    # )

    # run(
    #     GPTConfig88Triples,
    #     dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/triples_balanced_set_dataset_random.pth'
    # )

    # run(
    #     GPTConfig44TriplesLR,
    #     dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/triples_balanced_set_dataset_random.pth'
    # )

    # run(
    #     GPTConfig44TriplesEmbd,
    #     dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/triples_balanced_set_dataset_random.pth'
    # )


    # dataset = initialize_triples_datasets(
    #     GPTConfig44(),
    #     save_dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/triples_balanced_set_dataset_random.pth',
    #     save_tokenizer_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/triples_balanced_set_dataset_random_tokenizer.pkl'
    # )

    # train_loader, val_loader = initialize_loaders(GPTConfig44, dataset)


    # dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/balanced_set_dataset_random.pth'
    # dataset = torch.load(dataset_path)
    # train_loader, val_loader = initialize_loaders(GPTConfig44, dataset)
    # for input in train_loader:
    #     breakpoint()

    # lineplot_specific(
    #     config=GPTConfig48,
    #     input=[
    #         "A", "green", "B", "green", "C", "blue", "D", "green", "E", "pink", 
    #         "A", "one",  "B", "two",  "C", "one",  "D", "three",  "E", "one",
    #         "A", "squiggle", "B", "squiggle", "C", "squiggle", "D", "squiggle", "E", "squiggle",
    #         "A", "striped", "B", "striped", "C", "striped", "D", "striped", "E", "striped", 
    #         ">", "A", "B", "D", "/", "A", "C", "E", "."
    #     ],
    #     get_prediction=True,
    #     filename_prefix="test"
    # )

    # lineplot_specific(
    #     config=GPTConfig48,
    #     input=[
    #         "A", "green", "B", "green", "C", "blue", "D", "green", "E", "pink", 
    #         "A", "one",  "B", "two",  "C", "one",  "D", "three",  "E", "one",
    #         "A", "diamond", "B", "diamond", "C", "diamond", "D", "diamond", "E", "diamond",
    #         "A", "striped", "B", "striped", "C", "striped", "D", "striped", "E", "striped", 
    #         ">", "A", "B", "D", "/", "A", "C", "E", "."
    #     ],
    #     get_prediction=True,
    #     filename_prefix="test"
    # )


    # lineplot_specific(
    #     config=GPTConfig44,
    #     input=[
    #         "A", "squiggle", "B", "squiggle", "C", "squiggle", "D", "squiggle", "E", "squiggle",
    #         "A", "striped", "B", "striped", "C", "striped", "D", "striped", "E", "striped", 
    #         "A", "one",  "B", "two",  "C", "one",  "D", "three",  "E", "one",
    #         "A", "green", "B", "green", "C", "blue", "D", "green", "E", "pink", 
    #         ">", "A", "B", "D", "/", "A", "C", "E", "."
    #     ],
    #     get_prediction=True,
    #     filename_prefix="test"
    # )


    # lineplot_specific(
    #     config=GPTConfig44,
    #     input=[
    #         "A", "squiggle", "B", "squiggle", "C", "squiggle", "D", "squiggle", "E", "squiggle",
    #         "A", "striped", "B", "striped", "C", "solid", "D", "striped", "E", "striped", 
    #         "A", "one",  "B", "two",  "C", "one",  "D", "three",  "E", "one",
    #         "A", "green", "B", "green", "C", "blue", "D", "green", "E", "pink", 
    #         ">", "A", "B", "D", ".", ".", ".", ".", "."
    #     ],
    #     get_prediction=True,
    #     filename_prefix="test2"
    # )

    # lineplot_specific(
    #     config=GPTConfig44,
    #     input=[
    #         "A", "squiggle",  "D", "squiggle", "E", "squiggle",
    #         "A", "striped", "B", "striped", "C", "solid", 
    #         "A", "one",  "B", "two",  "C", "one",  "D", "three",  "E", "one",
    #         "A", "green", "B", "green", "C", "blue", "D", "green", "E", "pink", "B", "squiggle", "C", "squiggle", "D", "striped", "E", "striped", 
    #         ">", "A", "B", "D", ".", ".", ".", ".", "."
    #     ],
    #     get_prediction=True,
    #     filename_prefix="test3"
    # )


    # generate_heatmap(
    #     config=GPTConfig44,
    #     dataset_indices=[1, 0, 4],
    #     dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/balanced_set_dataset_random.pth',
    #     tokenizer_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/balanced_set_dataset_random_tokenizer.pkl',
    #     use_labels=True,
    #     threshold=0.1,
    #     get_prediction=True)











    # OLD RUNS - leaving in case want to reference later



    # add_causal_masking(GPTConfig(), "causal_full_run_random.pt")
    # add_causal_masking(GPTConfig24(), "causal_full_run_random_layers_2_heads_4.pt")
    # add_causal_masking(GPTConfig42(), "causal_full_run_random_layers_4_heads_2.pt")
    # add_causal_masking(GPTConfig44(), "causal_full_run_random_layers_4_heads_4.pt")

    # run(
    #     GPTConfig44_AttrFirst,
    #     dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/attr_first_balanced_set_dataset_random.pth',
    #     load_model=False)

    # New model configs
    # run(GPTConfig48, load_model=False)
    # run(GPTConfig44_Patience20, load_model=False)

    # run(
    #     GPTConfig44,
    #     dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/balanced_set_dataset_random.pth',
    #     load_model=True)

    # generate_heatmap(GPTConfig(), [1, 0, 4], use_labels=True)
    # generate_heatmap(GPTConfig24(), [1, 0, 4], use_labels=True)
    # generate_heatmap(GPTConfig42(), [1, 0, 4], use_labels=True)
    # generate_heatmap(GPTConfig44(), [1, 0, 4], use_labels=True)
    # generate_heatmap(GPTConfig48, [1, 0, 4], use_labels=True, threshold=0.05)
    # generate_heatmap(GPTConfig48, [1, 0, 4], use_labels=True, threshold=0.1)
    # generate_heatmap(
    #     config=GPTConfig44_AttrFirst,
    #     dataset_indices=[1, 0, 4],
    #     dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/attr_first_balanced_set_dataset_random.pth',
    #     tokenizer_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/attr_first_balanced_set_dataset_random_tokenizer.pkl',
    #     use_labels=True,
    #     threshold=0.1)

    # generate_heatmap(
    #     config=GPTConfig44,
    #     dataset_indices=[1, 0, 4],
    #     dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/balanced_set_dataset_random.pth',
    #     tokenizer_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/balanced_set_dataset_random_tokenizer.pkl',
    #     use_labels=True,
    #     threshold=0.1,
    #     get_prediction=True)

    # dataset = initialize_datasets(
    #     GPTConfig(),
    #     save_dataset_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/attr_first_balanced_set_dataset_random.pth',
    #     save_tokenizer_path='/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/attr_first_balanced_set_dataset_random_tokenizer.pkl',
    #     attr_first=True
    # )
