import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
from data_utils import pretty_print_input
from model import GPT
from tokenizer import load_tokenizer
import tkinter as tk
from tkinter import simpledialog


PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'


def plot_attention_heatmap(att_weights, labels, title="Attention Pattern", savefig=None):
    # Convert attention weights to numpy array
    att_weights_np = att_weights.detach().cpu().numpy()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the heatmap
    sns.heatmap(att_weights_np, ax=ax, cmap='rocket',
                cbar_kws={'label': 'Attention Weight'})

    # Set x-ticks to match the number of labels
    ax.set_xticks(range(len(labels)))
    # Set y-ticks to match the number of labels
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Key')
    ax.set_ylabel('Query')

    # Adjust layout and display
    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close(fig)


def plot_attention_heads_layer_horizontal(attention_weights, labels, layer, n_heads, title_prefix="Attention Pattern", savefig=None):
    # Set up a single row of subplots
    # Adjust width to fit all heads horizontally
    fig, axes = plt.subplots(1, n_heads, figsize=(n_heads * 10, 10))

    for head in range(n_heads):
        ax = axes[head] if n_heads > 1 else axes  # In case there's only 1 head

        # Extract attention weights for the current head
        att_weights_np = attention_weights[layer][0][head].detach(
        ).cpu().numpy()

        # Plot the heatmap for this head in its respective subplot
        sns.heatmap(att_weights_np, ax=ax, cmap='rocket', cbar=(head == n_heads - 1),
                    cbar_kws={'label': 'Attention Weight'} if head == n_heads - 1 else None)
        ax.set_title(f"Head {head}")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels, rotation=0)

    # Adjust layout and main title
    fig.suptitle(f"{title_prefix} Layer {layer}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for main title

    # Save or display the figure
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close(fig)


def plot_attention_pattern_all(attention_weights, labels, n_layers, n_heads, title_prefix="Attention Pattern", savefig=None):
    # Create grid layout with extra space for color bar
    fig = plt.figure(figsize=(n_heads * 10, n_layers * 10))
    gs = gridspec.GridSpec(n_layers, n_heads + 1,
                           width_ratios=[1] * n_heads + [0.05], wspace=0.3)

    axes = [[fig.add_subplot(gs[layer, head]) for head in range(
        n_heads)] for layer in range(n_layers)]
    # Dedicated color bar axis spanning all rows
    cbar_ax = fig.add_subplot(gs[:, -1])

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer][head]

            # Extract attention weights for the current head
            att_weights_np = attention_weights[layer][0][head].detach(
            ).cpu().numpy()

            # Plot the heatmap for this head in its respective subplot
            sns.heatmap(att_weights_np, ax=ax, cmap='rocket', cbar=(layer == 0 and head == n_heads - 1),
                        cbar_ax=cbar_ax if (
                            layer == 0 and head == n_heads - 1) else None,
                        cbar_kws={'label': 'Attention Weight'} if (layer == 0 and head == n_heads - 1) else None)

            ax.set_title(f"Layer {layer} Head {head}")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels, rotation=0)

    # Adjust layout and main title
    fig.suptitle(
        f"{title_prefix}: {n_layers} Layers, {n_heads} Heads", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for main title

    # Save or display the figure
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close(fig)

def plot_attention_pattern_lines(attention_weights, labels, n_layers, n_heads, title_prefix="Attention Line Pattern", savefig=None, threshold=None):
    fig, axes = plt.subplots(
        n_layers, n_heads, figsize=(n_heads * 4, n_layers * 8))

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]

            # Extract attention weights for the current head
            att_weights_np = attention_weights[layer][0][head].detach(
            ).cpu().numpy()

            # Plot the attention lines
            for i, label_start in enumerate(labels):
                for j, label_end in enumerate(labels):
                    weight = att_weights_np[i, j]
                    if threshold:
                        if weight < threshold:
                            continue
                    # Define line opacity based on attention weight
                    ax.plot([0, 1], [i, j], color='blue', alpha=weight)

            # Set labels for the columns
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Query", "Key"], fontsize=12, ha='center')
            ax.set_yticks(range(len(labels)))
            # Apply to left side
            ax.set_yticklabels(labels, fontsize=10, va='center')
            
            # Mirror labels on the right side
            ax.tick_params(axis='y', labelright=True,
                           right=True, labelleft=True, left=True)
            # ax.yaxis.set_tick_params(pad=-10)  # Adjust padding for both sides

            ax.invert_yaxis()  # Keep labels top-to-bottom
            ax.set_title(f"Layer {layer} Head {head}", fontsize=14)

    # Adjust layout and main title
    if threshold:
        fig.suptitle(
            f"{title_prefix}: {n_layers} Layers, {n_heads} Heads, weights ≥ {threshold} Threshold", fontsize=12)
    else:
        fig.suptitle(
            f"{title_prefix}: {n_layers} Layers, {n_heads} Heads", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    formatted_labels = pretty_print_input(labels)
    plt.figtext(0.15, 0.95, formatted_labels, ha="center", fontsize=10, family="monospace")

    # Save or display the figure
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close(fig)


def generate_lineplot(
        config,
        dataset_indices,
        dataset_path,
        tokenizer_path,
        use_labels=False,
        threshold=0.05,
        get_prediction=False):
    dataset = torch.load(dataset_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)
    print("Loaded dataset")

    # Restore the model state dict
    # checkpoint = torch.load(os.path.join(
    #     config.out_dir, config.filename), weights_only=False)
    # Restore the model state dict
    checkpoint = torch.load(os.path.join(
        PATH_PREFIX, config.filename), weights_only=False)

    model.load_state_dict(checkpoint["model"])
    print("Loaded model")

    if use_labels:
        tokenizer = load_tokenizer(tokenizer_path)

    for dataset_index in dataset_indices:
        print(f"Generating heatmap for index {dataset_index}")

        sequences = dataset[dataset_index].unsqueeze(0)

        if get_prediction:
            inputs = sequences[:, : config.input_size].to(device)
            targets = sequences[:, config.input_size:].to(device)

            outputs = model.generate(
                inputs,
                max_new_tokens=config.target_size)

            predictions = outputs[:, config.input_size:]

            mask = targets != config.padding_token  # Create a mask to ignore padding
            matches = ((predictions == targets) | ~mask).all(dim=1)

            print("matches: ", matches)
            print("correct: ", matches.sum().item())

            print("full output: ", tokenizer.decode(outputs[0].tolist()))
            print("predictions: ", tokenizer.decode(predictions[0].tolist()))
            print("target: ", tokenizer.decode(targets[0].tolist()))

        _, _, attention_weights, _ = model(
            sequences.to(device), False)
        print("Got attention weights")

        labels = dataset[dataset_index].tolist()

        if use_labels:
            labels = tokenizer.decode(labels)
            if "/" in labels:
                number_set = "two"
            elif "*" in labels:
                number_set = "zero"
            else:
                number_set = "one"

        # print("labels: ", labels)

        dir_path = f"figs/attention_pattern_layers_{config.n_layer}_heads_{config.n_head}"
        filename = f"larger_lineplot_sets_{number_set}_index_{dataset_index}_threshold_{threshold}.png"
        plot_attention_pattern_lines(
            attention_weights,
            labels,
            config.n_layer,
            config.n_head,
            title_prefix=f"Attention Pattern: {number_set.capitalize()} Set(s)",
            savefig=f"{dir_path}/{filename}",
            threshold=threshold)

        # plot_attention_pattern_lines(
        #     attention_weights,
        #     labels,
        #     config.n_layer,
        #     config.n_head,
        #     title_prefix=f"Attention Pattern: {number_set.capitalize()} Set(s)",
        #     savefig=None,
        #     threshold=threshold)


def make_prediction_given_input(
        config,
        input,
        tokenizer,
        model,
        get_prediction=False):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenized_input = torch.tensor(tokenizer.encode(input))
    sequences = tokenized_input.unsqueeze(0)

    if get_prediction:
        inputs = sequences[:, : config.input_size].to(device)
        targets = sequences[:, config.input_size:].to(device)

        outputs = model.generate(
            inputs,
            max_new_tokens=config.target_size)

        predictions = outputs[:, config.input_size:]

        mask = targets != config.padding_token  # Create a mask to ignore padding
        matches = ((predictions == targets) | ~mask).all(dim=1)
        print("correct: ", matches.sum().item())

        print("full output: ", tokenizer.decode(outputs[0].tolist()))
        print("predictions: ", tokenizer.decode(predictions[0].tolist()))
        print("target: ", tokenizer.decode(targets[0].tolist()))

    _, _, attention_weights, _ = model(
        sequences.to(device), False)
    
    return attention_weights

def lineplot_specific(
        config,
        input,
        # tokenizer_path=f"{PATH_PREFIX}/balanced_set_dataset_random_tokenizer.pkl",
        tokenizer_path=f"balanced_set_dataset_random_tokenizer.pkl",
        threshold=0.1,
        get_prediction=False,
        filename_prefix=""):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)
    print("Loaded dataset")

    # Restore the model state dict
    # checkpoint = torch.load(os.path.join(
    #     PATH_PREFIX, config.filename), weights_only=False)
    
    checkpoint = torch.load(os.path.join(
        config.filename), weights_only=False, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint["model"])
    print("Loaded model")

    tokenizer = load_tokenizer(tokenizer_path)
    input = torch.tensor(tokenizer.encode(input))
    sequences = input.unsqueeze(0)

    if get_prediction:
        inputs = sequences[:, : config.input_size].to(device)
        targets = sequences[:, config.input_size:].to(device)

        outputs = model.generate(
            inputs,
            max_new_tokens=config.target_size)

        predictions = outputs[:, config.input_size:]

        mask = targets != config.padding_token  # Create a mask to ignore padding
        matches = ((predictions == targets) | ~mask).all(dim=1)
        print("correct: ", matches.sum().item())

        print("full output: ", tokenizer.decode(outputs[0].tolist()))
        print("predictions: ", tokenizer.decode(predictions[0].tolist()))
        print("target: ", tokenizer.decode(targets[0].tolist()))

    # _, _, attention_weights, _ = model(
    #     sequences.to(device), False)

    # labels = input.tolist()
    # labels = tokenizer.decode(labels)
    # if "/" in labels:
    #     number_set = "two"
    # elif "*" in labels:
    #     number_set = "zero"
    # else:
    #     number_set = "one"

    # # print("labels: ", labels)

    # dir_path = f"figs/attention_pattern_layers_{config.n_layer}_heads_{config.n_head}/synthetic_larger"
    # filename = f"{filename_prefix}_lineplot_sets_{number_set}_threshold_{threshold}.png"
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)

    # plot_attention_pattern_lines(
    #     attention_weights,
    #     labels,
    #     config.n_layer,
    #     config.n_head,
    #     title_prefix=f"Attention Pattern: {number_set.capitalize()} Set(s)",
    #     savefig=f"{dir_path}/{filename}",
    #     threshold=threshold)
    

def plot_attention_pattern_lines_comparison(
        attention_weights1,
        attention_weights2,
        labels1,
        labels2,
        n_layers=4, 
        n_heads=4, 
        title_prefix="Attention Line Pattern Differences", 
        savefig=None, 
        threshold=None
    ):
    fig, axes = plt.subplots(
        n_layers, n_heads, figsize=(n_heads * 4, n_layers * 8))

    
    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]

            attention_weights_diff = attention_weights1[layer][0][head] - attention_weights2[layer][0][head]

            # breakpoint()
            # Extract attention weights for the current head
            att_weights_np = attention_weights_diff.detach().cpu().numpy()
            # att_weights_np = attention_weights_diff.numpy()

            # Plot the attention lines
            for i in range(len(labels1)):
                for j in range(len(labels1)):
                    weight = att_weights_np[i, j]
                    if threshold:
                        if weight < threshold:
                            continue
                    # Define line opacity based on attention weight
                    ax.plot([0, 1], [i, j], color='blue', alpha=weight)
            
            # Set labels for the columns
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Query", "Key"], fontsize=12, ha='center')
            ax.set_yticks(range(len(labels1)))
            ax.set_yticklabels([])
            ax.tick_params(axis='y', right=True, left=True)
            
            for index in range(len(labels1)):
                label1 = labels1[index]
                label2 = labels2[index]

                ax.text(-0.3, index, f"{label1}", fontsize=10, va='center_baseline', ha='right', transform=ax.transData)
                ax.text(-0.2, index, f"{label2}", fontsize=10, va='center_baseline', ha='right', transform=ax.transData)

                ax.text(1.2, index, f"{label1}", fontsize=10, va='center_baseline', ha='left', transform=ax.transData)
                ax.text(1.3, index, f"{label2}", fontsize=10, va='center_baseline', ha='left', transform=ax.transData)

            # ax.yaxis.set_tick_params(pad=-10)  # Adjust padding for both sides

            ax.invert_yaxis()  # Keep labels top-to-bottom
            ax.set_title(f"Layer {layer} Head {head}", fontsize=14)

    # Adjust layout and main title
    if threshold:
        fig.suptitle(
            f"{title_prefix}: {n_layers} Layers, {n_heads} Heads, weights ≥ {threshold} Threshold", fontsize=12)
    else:
        fig.suptitle(
            f"{title_prefix}: {n_layers} Layers, {n_heads} Heads", fontsize=12)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    formatted_labels = pretty_print_input(labels1)
    plt.figtext(0.15, 0.95, formatted_labels, ha="center", fontsize=10, family="monospace")

    # Save or display the figure
    if savefig is not None:
        plt.savefig(savefig)

    plt.show()
    plt.close(fig)


def lineplot_difference_inputs(
        config,
        input1,
        input2,
        # tokenizer_path=f"{PATH_PREFIX}/balanced_set_dataset_random_tokenizer.pkl",
        tokenizer_path=f"balanced_set_dataset_random_tokenizer.pkl",
        threshold=0.1,
        get_prediction=False,
        filename_prefix=""):
    print("Generating lineplot for difference in attention weights")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)

    # Restore the model state dict
    # checkpoint = torch.load(os.path.join(
    #     PATH_PREFIX, config.filename), weights_only=False)
    checkpoint = torch.load(config.filename, weights_only=False, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint["model"])
    print("Loaded model")

    tokenizer = load_tokenizer(tokenizer_path)
    print("Loaded tokenizer")

    attention_weights1 = make_prediction_given_input(
        config=config,
        input=input1,
        tokenizer=tokenizer,
        model=model,
        get_prediction=get_prediction)
    print("Got attention weights for input 1")
    
    attention_weights2 = make_prediction_given_input(
        config=config,
        input=input2,
        tokenizer=tokenizer,
        model=model,
        get_prediction=get_prediction)
    print("Got attention weights for input 2")


    dir_path = f"figs/attention_pattern_layers_{config.n_layer}_heads_{config.n_head}/differences"
    filename = f"{filename_prefix}_lineplot_threshold_{threshold}.png"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print("Plotting attention weights")
    plot_attention_pattern_lines_comparison(
        attention_weights1,
        attention_weights2,
        input1,
        input2,
        n_layers=4, 
        n_heads=4, 
        title_prefix="Attention Line Pattern", 
        savefig=f"{dir_path}/{filename}",
        threshold=threshold
    )

def interactive_lineplot_difference_inputs(config, tokenizer_path=f"{PATH_PREFIX}/balanced_set_dataset_random_tokenizer.pkl", threshold=0.1, get_prediction=False, filename_prefix=""):
    def get_input(prompt):
        return simpledialog.askstring("Input", prompt)

    root = tk.Tk()
    root.withdraw()  # Hide the main window

    input1 = get_input("Enter the first sequence (comma-separated):")
    input2 = get_input("Enter the second sequence (comma-separated):")

    if input1 and input2:
        input1 = input1.split(",")
        input2 = input2.split(",")

        lineplot_difference_inputs(
            config,
            input1,
            input2,
            tokenizer_path=tokenizer_path,
            threshold=threshold,
            get_prediction=get_prediction,
            filename_prefix=filename_prefix
        )
    else:
        print("Inputs cannot be empty.")

    root.destroy()

