from data_utils import get_cards, get_card_attributes, contains_set, get_target_seq, flatten_tuple
import itertools
import random
import torch
from tokenizer import load_tokenizer
from model import GPT, GPTConfig44_BalancedSets
import os

card_vectors = ["A", "B", "C", "D", "E"]

# Define attribute categories
shapes = ["oval", "squiggle", "diamond"]
colors = ["green", "blue", "pink"]
numbers = ["one", "two", "three"]
shadings = ["solid", "striped", "open"]

PATH_PREFIX = ""

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_target(input_sequence):
    card_attribute_dict = {
        "A": {"shape": 0, "color": 0, "number": 0, "shading": 0}, 
        "B": {"shape": 0, "color": 0, "number": 0, "shading": 0}, 
        "C": {"shape": 0, "color": 0, "number": 0, "shading": 0}, 
        "D": {"shape": 0, "color": 0, "number": 0, "shading": 0}, 
        "E": {"shape": 0, "color": 0, "number": 0, "shading": 0}}

    for index in range(len(input_sequence) - 1):
        card = input_sequence[index]
        attribute = input_sequence[index + 1]

        if attribute in shapes:
            card_attribute_dict[card]["shape"] = shapes.index(attribute)
        elif attribute in colors:
            card_attribute_dict[card]["color"] = colors.index(attribute)
        elif attribute in numbers:
            card_attribute_dict[card]["number"] = numbers.index(attribute)
        elif attribute in shadings:
            card_attribute_dict[card]["shading"] = shadings.index(attribute)

    combination = []
    for card, attribute_dict in card_attribute_dict.items():
        combination.append((attribute_dict["shape"], attribute_dict["color"], attribute_dict["number"], attribute_dict["shading"]))
    combination = tuple(combination)

    target_seq = get_target_seq(combination, target_size=8, pad_symbol="_")
    return target_seq

def predict_sequence(input_seq, model, config, tokenizer_path="larger_balanced_set_dataset_random_tokenizer.pkl"):
    tokenizer = load_tokenizer(tokenizer_path)
    
    full_seq = input_seq.copy()
    full_seq.append(">")

    target_seq = generate_target(input_seq)
    full_seq.extend(target_seq)

    tokenized_full_seq = torch.tensor(tokenizer.encode(full_seq))
    tokenized_full_seq = tokenized_full_seq.unsqueeze(0)

    tokenized_inputs = tokenized_full_seq[:, : config.input_size].to(device)
    tokenized_targets = tokenized_full_seq[:, config.input_size:].to(device)

    outputs = model.generate(
        tokenized_inputs,
        max_new_tokens=config.target_size)

    predictions = outputs[:, config.input_size:]

    mask = tokenized_targets != config.padding_token  # Create a mask to ignore padding
    matches = ((predictions == tokenized_targets) | ~mask).all(dim=1)
    is_correct = bool(matches.sum().item())

    decoded_predictions = tokenizer.decode(predictions[0].tolist())
    decoded_targets = tokenizer.decode(tokenized_targets[0].tolist())


    print("full seq: ", full_seq)
    print("correct: ", is_correct)
        # print("full output: ", tokenizer.decode(outputs[0].tolist()))
    print("predictions: ", decoded_predictions)
    print("target: ", decoded_targets)

    return is_correct


def test_combinations(n_cards=5, random_order=True, config=GPTConfig44_BalancedSets):
    # random.seed(42)
    cards = get_cards()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)
    print("Loaded dataset")

    # Restore the model state dict
    checkpoint = torch.load(os.path.join(
        PATH_PREFIX, config.filename), weights_only=False, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint["model"])
    print("Loaded model")


    correct = 0
    total = 0

    for index, combination in enumerate(itertools.combinations(cards, n_cards)):
        # Create the initial array of 20 tuples

        tuple_array = [
            (card_vectors[i], attr)
            for i, card in enumerate(combination)
            for attr in get_card_attributes(*card)
        ]

        random_iterations = 1

        # if contains_set(combination) and balance_sets:
        #     random_iterations = 7
        
        random.seed(42)
        for i in range(random_iterations):
            # random.seed(42 + i)
            # Randomize the array
            if random_order:
                random.shuffle(tuple_array)

            # Flatten the array to 40 elements using the new flatten_tuple function
            flattened_array = flatten_tuple(tuple_array)
            
            is_correct = predict_sequence(flattened_array, model, config)

            correct += is_correct
            total += 1
        
        if index % 100 == 0:
            print(f"Accuracy: {correct / total}")
            breakpoint()

if __name__ == "__main__":
    # random.seed(42)
    test_combinations()


            
            
