import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple
import itertools
from itertools import chain
import random
from tokenizer import Tokenizer, save_tokenizer
from set_dataset import SetDataset, BalancedSetDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.gridspec as gridspec

n_cards = 3
card_vectors = ["A", "B", "C", "D", "E"]

# Define attribute categories
shapes = ["oval", "squiggle", "diamond"]
colors = ["green", "blue", "pink"]
numbers = ["one", "two", "three"]
shadings = ["solid", "striped", "open"]


# Function to get the card encoding with attribute vectors
def get_card_attributes(shape, color, number, shading):
    return (shapes[shape], colors[color], numbers[number], shadings[shading])


def flatten_tuple(t):
    return list(chain.from_iterable(x if isinstance(x, tuple) else (x,) for x in t))


def is_set(card1, card2, card3):
    return all((a + b + c) % 3 == 0 for a, b, c in zip(card1, card2, card3))


def find_sets_with_cards(combination: Tuple) -> List[Tuple]:
    sets = []
    n_cards = len(combination)
    # print("combination: ", combination)
    for i in range(n_cards - 2):
        for j in range(i + 1, n_cards - 1):
            for k in range(j + 1, n_cards):
                if is_set(combination[i], combination[j], combination[k]):
                    # print("appending: ", (card_vectors[i], card_vectors[j], card_vectors[k]))
                    sets.extend(
                        [card_vectors[i], card_vectors[j], card_vectors[k]])

    if len(sets) == 6:
        sets.insert(3, "/")

    if len(sets) == 0:
        sets.append("*")

    return sets


def get_cards():
    # Define attribute categories
    shapes_range = range(3)
    colors_range = range(3)
    numbers_range = range(3)
    shadings_range = range(3)

    # Generate all possible cards
    cards = list(
        itertools.product(shapes_range, colors_range,
                          numbers_range, shadings_range)
    )
    return cards


def get_target_seq(combination, target_size, pad_symbol):
    target_seq = find_sets_with_cards(combination)

    target_seq.append(".")

    for _ in range(target_size - len(target_seq)):
        target_seq.append(pad_symbol)

    return target_seq

# Generate continuous combinations (A x y z B i j k ...) where all attributes directly follow card


def generate_cont_combinations(block_size, pad_symbol, n_cards=3):

    target_size = 4

    if n_cards == 5:
        target_size = 8

    cards = get_cards()

    for combination in itertools.permutations(cards, n_cards):
        card_tuples = [
            (card_vectors[i], get_card_attributes(*card)) for i, card in enumerate(combination)
        ]

        # Randomize the array
        # random.shuffle(array_of_3)

        # Flatten the array to 40 elements using the new flatten_tuple function
        flattened_array = flatten_tuple(flatten_tuple(card_tuples))
        flattened_array.append(">")
        flattened_array.extend(get_target_seq(
            combination, target_size, pad_symbol))

        yield flattened_array


def generate_combinations(target_size, pad_symbol, n_cards, random_order=False):

    cards = get_cards()

    for combination in itertools.combinations(cards, n_cards):
        # Create the initial array of 20 tuples
        tuple_array = [
            (card_vectors[i], attr)
            for i, card in enumerate(combination)
            for attr in get_card_attributes(*card)
        ]

        # Randomize the array
        if random_order:
            random.shuffle(tuple_array)

        # Flatten the array to 40 elements using the new flatten_tuple function
        flattened_array = flatten_tuple(tuple_array)
        flattened_array.append(">")

        flattened_array.extend(get_target_seq(
            combination, target_size, pad_symbol))
        yield flattened_array


def separate_sets_non_sets(tokenized_combinations, no_set_token, expected_pos):
    set_sequences = []
    non_set_sequences = []
    for tokenized_combo in tokenized_combinations:
        if tokenized_combo[expected_pos] == no_set_token:
            non_set_sequences.append(tokenized_combo)
        else:
            set_sequences.append(tokenized_combo)
    return set_sequences, non_set_sequences


def initialize_datasets(config, save_dataset=False, save_tokenizer_path=None):
    optimized_combinations = generate_combinations(
        config.target_size, config.pad_symbol, config.n_cards, random_order=True
    )

    small_combinations = list(optimized_combinations)

    # Create tokenizer and tokenize all sequences
    tokenizer = Tokenizer()
    tokenized_combinations = [tokenizer.encode(
        seq) for seq in small_combinations]

    if save_tokenizer_path:
        save_tokenizer(tokenizer, save_tokenizer_path)

    breakpoint()

    end_of_seq_token = -1
    padding_token = -1
    no_set_token = -1

    for i in range(len(small_combinations)):
        if config.pad_symbol in small_combinations[i]:
            padding_token_pos = small_combinations[i].index(config.pad_symbol)
            padding_token = tokenized_combinations[i][padding_token_pos]

        if "*" in small_combinations[i]:
            no_set_token_pos = small_combinations[i].index("*")
            no_set_token = tokenized_combinations[i][no_set_token_pos]

        if "." in small_combinations[i]:
            end_of_seq_token_pos = small_combinations[i].index(".")
            end_of_seq_token = tokenized_combinations[i][end_of_seq_token_pos]

        if no_set_token >= 0 and padding_token >= 0 and end_of_seq_token >= 0:
            break

    print("padding token: ", padding_token)
    print("end of seq token: ", end_of_seq_token)
    print("no set token: ", no_set_token)

    config.end_of_seq_token = end_of_seq_token
    config.padding_token = padding_token

    # Separate out sets from non sets in the tokenized representation
    set_sequences, non_set_sequences = separate_sets_non_sets(
        tokenized_combinations, no_set_token, -config.target_size)

    # Create dataset and dataloaders
    # dataset = SetDataset(tokenized_combinations)
    # train_size = int(0.95 * len(dataset))
    dataset = BalancedSetDataset(set_sequences, non_set_sequences)

    if save_dataset:
        torch.save(
            dataset, '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp/balanced_set_dataset_random.pth')
    return dataset


def initialize_loaders(config, dataset):
    train_size = int(0.95 * len(dataset))  # makes val size 1296

    # make validation set a lot smaller TODO, revisit how large val set this leaves us with
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print("train_size: ", train_size, " val_size: ", val_size)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


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


# def plot_attention_pattern_all(attention_weights, labels, n_layers, n_heads, title_prefix="Attention Pattern", savefig=None):
#     fig, axes = plt.subplots(
#         n_layers, n_heads, figsize=(n_heads * 10, n_layers * 10))

#     for layer in range(n_layers):
#         for head in range(n_heads):
#             ax = axes[layer, head]

#             # Extract attention weights for the current head
#             att_weights_np = attention_weights[layer][0][head].detach(
#             ).cpu().numpy()

#             # Plot the heatmap for this head in its respective subplot
#             sns.heatmap(att_weights_np, ax=ax, cmap='rocket', cbar=(head == n_heads - 1),
#                         cbar_kws={'label': 'Attention Weight'} if head == n_heads - 1 else None)
#             ax.set_title(f"Layer {layer} Head {head}")
#             ax.set_xticks(range(len(labels)))
#             ax.set_yticks(range(len(labels)))
#             ax.set_xticklabels(labels, rotation=45, ha="right")
#             ax.set_yticklabels(labels, rotation=0)

#     # Adjust layout and main title
#     fig.suptitle(f"{title_prefix}", fontsize=18)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for main title

#     # Save or display the figure
#     if savefig is not None:
#         plt.savefig(savefig)
#     plt.show()
#     plt.close(fig)

def plot_attention_pattern_all(attention_weights, labels, n_layers, n_heads, title_prefix="Attention Pattern", savefig=None):
    # Create grid layout with extra space for color bar
    fig = plt.figure(figsize=(n_heads * 10, n_layers * 10))
    gs = gridspec.GridSpec(n_layers, n_heads + 1, width_ratios=[1] * n_heads + [0.05], wspace=0.3)
    
    axes = [[fig.add_subplot(gs[layer, head]) for head in range(n_heads)] for layer in range(n_layers)]
    cbar_ax = fig.add_subplot(gs[:, -1])  # Dedicated color bar axis spanning all rows

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer][head]

            # Extract attention weights for the current head
            att_weights_np = attention_weights[layer][0][head].detach().cpu().numpy()

            # Plot the heatmap for this head in its respective subplot
            sns.heatmap(att_weights_np, ax=ax, cmap='rocket', cbar=(layer == 0 and head == n_heads - 1),
                        cbar_ax=cbar_ax if (layer == 0 and head == n_heads - 1) else None,
                        cbar_kws={'label': 'Attention Weight'} if (layer == 0 and head == n_heads - 1) else None)
            
            ax.set_title(f"Layer {layer} Head {head}")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels, rotation=0)

    # Adjust layout and main title
    fig.suptitle(f"{title_prefix}: {n_layers} Layers, {n_heads} Heads", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for main title

    # Save or display the figure
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close(fig)

def plot_attention_pattern_lines(attention_weights, labels, n_layers, n_heads, title_prefix="Attention Line Pattern", savefig=None):
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(n_heads * 8, n_layers * 8))
    if n_layers == 1 and n_heads == 1:
        axes = np.array([[axes]])

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]

            # Extract attention weights for the current head
            # att_weights_np = attention_weights[layer][0][head].detach().cpu().numpy()
            att_weights_np = attention_weights[layer][0][head]

            # Plot the attention lines
            for i, label_start in enumerate(labels):
                for j, label_end in enumerate(labels):
                    weight = att_weights_np[i, j]
                    # Define line thickness based on attention weight
                    # linewidth = max(0.1, weight * 5)
                    # linewidth = weight
                    # ax.plot([0, 1], [i, j], linewidth=linewidth, color='blue', alpha=0.7)
                    ax.plot([0, 1], [i, j], color='blue', alpha=weight)

            # Set labels for the columns
            ax.set_xlim(-0.1, 1.1)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Query", "Key"])
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.invert_yaxis()

            ax.set_title(f"Layer {layer} Head {head}")

    # Adjust layout and main title
    fig.suptitle(f"{title_prefix}: {n_layers} Layers, {n_heads} Heads", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save or display the figure
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close(fig)
