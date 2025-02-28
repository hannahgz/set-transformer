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
from model import GPTConfig44_Seeded
import os

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
                    
                    # sets.extend(
                    #     [shuffled_card_vectors[i], shuffled_card_vectors[j], shuffled_card_vectors[k]])
                    sorted_triplet = sorted([shuffled_card_vectors[i], shuffled_card_vectors[j], shuffled_card_vectors[k]])
                    sets.extend(sorted_triplet)
                    
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
        
        for i in range(random_iterations):
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


def initialize_triples_datasets(config):
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
    tokenizer = load_tokenizer(config.tokenizer_path)
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

    breakpoint()
    
    if config.dataset_path:
        torch.save(dataset, config.dataset_path)
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


if __name__ == "__main__":
    # seed = 42
    

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

    for curr_seed in [1, 2, 3, 4]:
        torch.manual_seed(curr_seed)
        random.seed(curr_seed)
        np.random.seed(curr_seed)

        config = GPTConfig44_Seeded(seed = curr_seed)
        dataset_path = config.dataset_path

        if not os.path.exists(os.path.dirname(dataset_path)):
            os.makedirs(os.path.dirname(dataset_path))

        print("Initializing triples dataset")
        initialize_triples_datasets(
            config = config
        )

