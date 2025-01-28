from typing import List, Tuple
import itertools
from itertools import chain
import random
from tokenizer import Tokenizer, save_tokenizer, load_tokenizer
from set_dataset import SetDataset, BalancedSetDataset, BalancedTriplesSetDataset
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time
import pickle
import numpy as np
from model import GPTConfig44_Complete

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

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

def num_sets(combination):
    n_cards = len(combination)
    num_sets = 0
    for i in range(n_cards - 2):
        for j in range(i + 1, n_cards - 1):
            for k in range(j + 1, n_cards):
                if is_set(combination[i], combination[j], combination[k]):
                    num_sets += 1
    
    return num_sets

def find_sets_with_cards(combination: Tuple, shuffled_card_vectors: List):
    sets = []
    n_cards = len(combination)
    # print("combination: ", combination)
    for i in range(n_cards - 2):
        for j in range(i + 1, n_cards - 1):
            for k in range(j + 1, n_cards):
                if is_set(combination[i], combination[j], combination[k]):
                    # print("appending: ", (card_vectors[i], card_vectors[j], card_vectors[k]))
                    # sets.extend(
                    #     [card_vectors[i], card_vectors[j], card_vectors[k]])
                    
                    sets.extend(
                        [shuffled_card_vectors[i], shuffled_card_vectors[j], shuffled_card_vectors[k]])

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


def get_target_seq(combination, target_size, pad_symbol, shuffled_card_vectors):
    target_seq = find_sets_with_cards(combination, shuffled_card_vectors)

    target_seq.append(".")

    for _ in range(target_size - len(target_seq)):
        target_seq.append(pad_symbol)

    return target_seq


def generate_combinations(target_size, pad_symbol, n_cards, random_order=False, attr_first=False, balance_sets=False):

    cards = get_cards()

    for combination in itertools.combinations(cards, n_cards):
        # Create the initial array of 20 tuples

        shuffled_card_vectors = random.sample(card_vectors, n_cards)
        if not attr_first:
            shuffled_tuple_array = [
                (shuffled_card_vectors[i], attr)
                for i, card in enumerate(combination)
                for attr in get_card_attributes(*card)
            ]
            # tuple_array = [
            #     (card_vectors[i], attr)
            #     for i, card in enumerate(combination)
            #     for attr in get_card_attributes(*card)
            # ]
        else:
            print("Attribute First Dataset")
            # tuple_array = [
            #     (attr, card_vectors[i])
            #     for i, card in enumerate(combination)
            #     for attr in get_card_attributes(*card)
            # ]


        random_iterations = 1
        if num_sets(combination) == 0 and balance_sets:
            # random_iterations = 3 # actual 0 set size = 22441536
            random_iterations = 1
        elif num_sets(combination) == 1 and balance_sets:
            # random_iterations = 25 # actual 1 set size = 77922000/25 = 3116880
            random_iterations = 7
        elif num_sets(combination) == 2 and balance_sets:
            # random_iterations = 250 # actual 2 set size = 15795000/250 = 63180
            random_iterations = 345

        target_seq = get_target_seq(
                combination, target_size, pad_symbol, shuffled_card_vectors)
        # random.seed(42)
        for i in range(random_iterations):
            # random.seed(42 + i)
            # Randomize the array
            if random_order:
                random.shuffle(shuffled_tuple_array)

            # Flatten the array to 40 elements using the new flatten_tuple function
            flattened_array = flatten_tuple(shuffled_tuple_array)
            flattened_array.append(">")

            flattened_array.extend(target_seq)
            
            yield flattened_array


def separate_all_sets(tokenized_combinations, no_set_token, separate_token):
    no_set_sequences = []
    one_set_sequences = []
    two_set_sequences = []
    for tokenized_combo in tokenized_combinations:
        if no_set_token in tokenized_combo:
            no_set_sequences.append(tokenized_combo)
        elif separate_token in tokenized_combo:
            two_set_sequences.append(tokenized_combo)
        else:
            one_set_sequences.append(tokenized_combo)

    random.shuffle(no_set_sequences)
    random.shuffle(one_set_sequences)
    random.shuffle(two_set_sequences)

    return no_set_sequences, one_set_sequences, two_set_sequences


def initialize_triples_datasets(config, save_dataset_path=None, attr_first=False):
    optimized_combinations = generate_combinations(
        target_size=config.target_size,
        pad_symbol=config.pad_symbol,
        n_cards=config.n_cards, 
        random_order=True, 
        attr_first=False,
        balance_sets=True
    )

    small_combinations = list(optimized_combinations)

    # Create tokenizer and tokenize all sequences
    tokenizer = load_tokenizer(f"{PATH_PREFIX}/all_tokenizer.pkl")
    tokenized_combinations = [tokenizer.encode(
        seq) for seq in small_combinations]

    
    no_set_token = tokenizer.token_to_id["*"]
    separate_token = tokenizer.token_to_id["/"]

    breakpoint()
    # Separate out sets from non sets in the tokenized representation
    no_set_sequences, one_set_sequences, two_set_sequences = separate_all_sets(
        tokenized_combinations, no_set_token, separate_token
    )

    print("len(no_set_sequences): ", len(no_set_sequences))
    print("len(one_set_sequences): ", len(one_set_sequences))
    print("len(two_set_sequences): ", len(two_set_sequences))

    # Create dataset and dataloaders
    # dataset = SetDataset(tokenized_combinations)
    # train_size = int(0.95 * len(dataset))
    dataset = BalancedTriplesSetDataset(
        no_set_sequences, one_set_sequences, two_set_sequences)

    if save_dataset_path:
        torch.save(dataset, save_dataset_path)
    return dataset


def initialize_loaders(config, dataset):
    train_size = int(0.99 * len(dataset))

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

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Splits the data into train, validation and test sets."""
    # First split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Then split train+val into train and val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_ratio, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def pretty_print_input(input):
    grid = [["" for _ in range(4)] for _ in range(5)]
    card_indices = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    
    # Create a mapping from attribute values to their types
    attr_mapping = {}
    for shape in shapes:
        attr_mapping[shape] = "shape"
    for color in colors:
        attr_mapping[color] = "color"
    for number in numbers:
        attr_mapping[number] = "number"
    for shading in shadings:
        attr_mapping[shading] = "shading"
    
    i = 0
    while i < 40:
        card = input[i]
        attr = input[i+1]

        attr_type = attr_mapping[attr]

        if attr_type == "shape":
            grid[card_indices[card]][0] = attr
        elif attr_type == "shading":
            grid[card_indices[card]][1] = attr
        elif attr_type == "number":
            grid[card_indices[card]][2] = attr
        elif attr_type == "color":
            grid[card_indices[card]][3] = attr

        i += 2
        
    # print("Card\tShape\tShading\tNumber\tColor")
    # for card, attrs in zip(card_indices.keys(), grid):
    #     print(f"{card}\t" + "\t".join(attrs))

    # Define column widths
    col_widths = [10, 10, 10, 10, 10]

    # Create header
    headers = ["Card", "Shape", "Shading", "Number", "Color"]
    header_line = "".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))

    # Create rows
    rows = []
    for card, attrs in zip(card_indices.keys(), grid):
        row = [card] + attrs
        row_line = "".join(attr.ljust(col_widths[i]) for i, attr in enumerate(row))
        rows.append(row_line)
    
    return header_line + "\n" + "\n".join(rows)

    # # Define column widths
    # col_widths = [5, 10, 10, 10, 10]

    # # Print header
    # headers = ["Card", "Shape", "Shading", "Number", "Color"]
    # header_line = "".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
    # print(header_line)

    # # Print rows
    # for card, attrs in zip(card_indices.keys(), grid):
    #     row = [card] + attrs
    #     row_line = "".join(attr.ljust(col_widths[i]) for i, attr in enumerate(row))
    #     print(row_line)


def generate_base_combinations(n_cards = 5):
    cards = get_cards()

    for combination in itertools.combinations(cards, n_cards):
        # Create the initial array of 20 tuples

        # tuple_array = [
        #     (card_vectors[i], attr)
        #     for i, card in enumerate(combination)
        #     for attr in get_card_attributes(*card)
        # ]
        shuffled_card_vectors = random.sample(card_vectors, n_cards)
        
        shuffled_tuple_array = [
            (shuffled_card_vectors[i], attr)
            for i, card in enumerate(combination)
            for attr in get_card_attributes(*card)
        ]

        target_seq = get_target_seq(
                combination, 8, "_", shuffled_card_vectors)

        random.shuffle(shuffled_tuple_array)

        # Flatten the array to 40 elements using the new flatten_tuple function
        flattened_array = flatten_tuple(shuffled_tuple_array)
        flattened_array.append(">")

        flattened_array.extend(target_seq)
        
        yield flattened_array

def initialize_base_dataset(save_dataset_path=None):
    optimized_combinations = generate_base_combinations()

    small_combinations = list(optimized_combinations)

    # Create tokenizer and tokenize all sequences
    tokenizer = load_tokenizer(f"{PATH_PREFIX}/all_tokenizer.pkl")
    tokenized_combinations = [tokenizer.encode(
        seq) for seq in small_combinations]
    breakpoint()

    dataset = SetDataset(tokenized_combinations)

    if save_dataset_path:
        torch.save(dataset, save_dataset_path)
    return dataset

def find_paired_sequence(dataset, tokenizer_path, target_sequence):
    """
    Search for any ordering of paired elements in the dataset.
    
    Args:
        dataset: List of sequences
        target_sequence: List with paired elements to search for
        
    Returns:
        List of tuples containing (index, matching_subsequence) where found
    """
    
    # Convert target sequence into pairs
    target_pairs = []

    tokenizer = load_tokenizer(tokenizer_path)
    target_sequence = tokenizer.encode(target_sequence)
    for i in range(0, len(target_sequence) - 8 - 1, 2):
        target_pairs.append((target_sequence[i], target_sequence[i + 1]))
    
    target_pairs_set = set(target_pairs)
    print(f"Number of target pairs: {len(target_pairs_set)}")

    # Iterate through dataset
    for idx, sequence in enumerate(dataset):
        print(f"Id {idx}/{len(dataset)}")
        if torch.is_tensor(sequence):
            sequence = sequence.tolist()
        # Convert sequence into pairs, excluding special tokens
        sequence_pairs = []
        for i in range(0, len(sequence) - 8 - 1, 2):
            # Skip special tokens like ">", ".", "_"
            sequence_pairs.append((sequence[i], sequence[i + 1]))

        sequence_pairs_set = set(sequence_pairs)
        print("Number of sequence pairs: ", len(sequence_pairs_set))
        # Print what pairs are different
        missing_pairs = target_pairs_set - sequence_pairs_set
        extra_pairs = sequence_pairs_set - target_pairs_set
        if missing_pairs:
            print(f"Missing pairs: {missing_pairs}")
            print(f"Num missing pairs: {len(missing_pairs)}")
        if extra_pairs:
            print(f"Extra pairs: {extra_pairs}")
            print(f"Num extra pairs: {len(extra_pairs)}")

        breakpoint()
        if set(sequence_pairs) == target_pairs_set:
            print("Found match at index:", idx)
            return (idx, sequence)
                
    return None

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # test = generate_base_combinations()
    # print(next(test))
    # print(next(test))
    # print(next(test))
    # print(next(test))
    # print(next(test))
    # print(next(test))
    # print(next(test))

    # breakpoint()

    # print("Initializing base dataset")
    # initialize_base_dataset(
    #     save_dataset_path=f"{PATH_PREFIX}/base_card_randomization_tuple_randomization_dataset.pth"
    # )

    print("Initializing triples dataset")
    initialize_triples_datasets(
        config = GPTConfig44_Complete(),
        save_dataset_path=f"{PATH_PREFIX}/triples_card_randomization_tuple_randomization_dataset.pth"
    )

