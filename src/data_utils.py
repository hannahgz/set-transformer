from typing import List, Tuple
import itertools
from itertools import chain
import random
from tokenizer import Tokenizer, save_tokenizer
from set_dataset import SetDataset, BalancedSetDataset, BalancedTriplesSetDataset
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time
import pickle

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

def contains_set(combination):
    n_cards = len(combination)
    for i in range(n_cards - 2):
        for j in range(i + 1, n_cards - 1):
            for k in range(j + 1, n_cards):
                if is_set(combination[i], combination[j], combination[k]):
                    return True
    return False

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


def generate_combinations(target_size, pad_symbol, n_cards, random_order=False, attr_first=False, balance_sets=False):

    cards = get_cards()

    for combination in itertools.combinations(cards, n_cards):
        # Create the initial array of 20 tuples

        if not attr_first:
            tuple_array = [
                (card_vectors[i], attr)
                for i, card in enumerate(combination)
                for attr in get_card_attributes(*card)
            ]
        else:
            tuple_array = [
                (attr, card_vectors[i])
                for i, card in enumerate(combination)
                for attr in get_card_attributes(*card)
            ]

        random_iterations = 1
        if contains_set(combination) and balance_sets:
            random_iterations = 7

        target_seq = get_target_seq(
                combination, target_size, pad_symbol)
        random.seed(42)
        for i in range(random_iterations):
            random.seed(42 + i)
            # Randomize the array
            if random_order:
                random.shuffle(tuple_array)

            # Flatten the array to 40 elements using the new flatten_tuple function
            flattened_array = flatten_tuple(tuple_array)
            flattened_array.append(">")

            flattened_array.extend(target_seq)
            
            yield flattened_array

# def generate_combinations(target_size, pad_symbol, n_cards, random_order=False, attr_first=False):

#     cards = get_cards()

#     for combination in itertools.combinations(cards, n_cards):
#         # Create the initial array of 20 tuples

#         if not attr_first:
#             tuple_array = [
#                 (card_vectors[i], attr)
#                 for i, card in enumerate(combination)
#                 for attr in get_card_attributes(*card)
#             ]
#         else:
#             tuple_array = [
#                 (attr, card_vectors[i])
#                 for i, card in enumerate(combination)
#                 for attr in get_card_attributes(*card)
#             ]

#         # Randomize the array
#         if random_order:
#             random.shuffle(tuple_array)

#         # Flatten the array to 40 elements using the new flatten_tuple function
#         flattened_array = flatten_tuple(tuple_array)
#         flattened_array.append(">")

#         flattened_array.extend(get_target_seq(
#             combination, target_size, pad_symbol))
#         yield flattened_array

def separate_sets_non_sets(tokenized_combinations, no_set_token, expected_pos, randomize=False):
    set_sequences = []
    non_set_sequences = []
    for tokenized_combo in tokenized_combinations:
        if tokenized_combo[expected_pos] == no_set_token:
            non_set_sequences.append(tokenized_combo)
        else:
            set_sequences.append(tokenized_combo)

    if randomize:
        random.shuffle(set_sequences)
        random.shuffle(non_set_sequences)

    return set_sequences, non_set_sequences


def initialize_datasets(config, save_dataset_path=None, save_tokenizer_path=None, attr_first=False, randomize_sequence_order=False):
    # optimized_combinations = generate_combinations(
    #     config.target_size, config.pad_symbol, config.n_cards, random_order=True, attr_first=attr_first
    # )

    optimized_combinations = generate_combinations(
        config.target_size, config.pad_symbol, config.n_cards, random_order=True, attr_first=attr_first, balance_sets=True
    )

    start_time = time.time()
    small_combinations = list(optimized_combinations)
    print("Len small_combinations: ", len(small_combinations))
    end_time = time.time()

    print("Time taken to construct small_combinations: ", end_time - start_time, "seconds")

    with open(f'{PATH_PREFIX}/small_combo.pkl', 'wb') as f:
        print("Saving small_combinations to pickle file")
        pickle.dump(small_combinations, f)

    # Create tokenizer and tokenize all sequences
    tokenizer = Tokenizer()
    tokenized_combinations = [tokenizer.encode(
        seq) for seq in small_combinations]

    if save_tokenizer_path:
        save_tokenizer(tokenizer, save_tokenizer_path)

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
        tokenized_combinations, no_set_token, -config.target_size, randomize_sequence_order)
    print("len set_sequences: ", len(set_sequences))
    print("len non_set_sequences: ", len(non_set_sequences))

    # Create dataset and dataloaders
    dataset = BalancedSetDataset(set_sequences, non_set_sequences)

    if save_dataset_path:
        torch.save(dataset, save_dataset_path)
    return dataset


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


def initialize_triples_datasets(config, save_dataset_path=None, save_tokenizer_path=None, attr_first=False):
    optimized_combinations = generate_combinations(
        config.target_size, config.pad_symbol, config.n_cards, random_order=True, attr_first=attr_first
    )

    small_combinations = list(optimized_combinations)

    # Create tokenizer and tokenize all sequences
    tokenizer = Tokenizer()
    tokenized_combinations = [tokenizer.encode(
        seq) for seq in small_combinations]

    if save_tokenizer_path:
        save_tokenizer(tokenizer, save_tokenizer_path)

    separate_token = -1
    no_set_token = -1

    for i in range(len(small_combinations)):
        if "/" in small_combinations[i]:
            separate_token_pos = small_combinations[i].index("/")
            separate_token = tokenized_combinations[i][separate_token_pos]

        if "*" in small_combinations[i]:
            no_set_token_pos = small_combinations[i].index("*")
            no_set_token = tokenized_combinations[i][no_set_token_pos]

        if no_set_token >= 0 and separate_token >= 0:
            break

    print("separate token: ", separate_token)
    print("no set token: ", no_set_token)

    # Separate out sets from non sets in the tokenized representation
    no_set_sequences, one_set_sequences, two_set_sequences = separate_all_sets(
        tokenized_combinations, no_set_token, separate_token
    )

    # Create dataset and dataloaders
    # dataset = SetDataset(tokenized_combinations)
    # train_size = int(0.95 * len(dataset))
    dataset = BalancedTriplesSetDataset(
        no_set_sequences, one_set_sequences, two_set_sequences)
    breakpoint()

    if save_dataset_path:
        torch.save(dataset, save_dataset_path)
    return dataset


def initialize_loaders(config, dataset):
    train_size = int(0.95 * len(dataset))

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


if __name__ == "__main__":
    cards = get_cards()

    for combination in itertools.combinations(cards, 5):
        breakpoint()

