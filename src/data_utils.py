from typing import List, Tuple
import itertools
from itertools import chain

n_cards = 3
card_vectors = ["A", "B", "C"]

# Define attribute categories
shapes = ["oval", "squiggle", "diamond"]
colors = ["green", "blue", "pink"]
numbers = ["one", "two", "three"]
shadings = ["solid", "striped", "open"]


# Function to get the card encoding with attribute vectors
def get_card(shape, color, number, shading):
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
                    sets.extend([card_vectors[i], card_vectors[j], card_vectors[k]])

    if len(sets) == 0:
        sets.append("*")
    return sets


# Generate combinations of cards on-the-fly
def generate_combinations(block_size, pad_symbol):

    # Use arrays instead of lists for card and attribute vectors
    # Define attribute categories
    shapes_range = range(3)
    colors_range = range(3)
    numbers_range = range(3)
    shadings_range = range(3)

    # Generate all possible cards
    cards = list(
        itertools.product(shapes_range, colors_range, numbers_range, shadings_range)
    )

    for combination in itertools.permutations(cards, n_cards):
        # Create the initial array of 20 tuples
        array_of_3 = [
            (card_vectors[i], get_card(*card)) for i, card in enumerate(combination)
        ]

        ## Randomize the array
        # random.shuffle(array_of_3)

        # Flatten the array to 40 elements using the new flatten_tuple function
        flattened_array = flatten_tuple(flatten_tuple(array_of_3))

        flattened_array.append(">")

        flattened_array.extend(find_sets_with_cards(combination))

        curr_len = len(flattened_array)

        # Padding with "_" (padding symbol)
        for i in range(curr_len, block_size - 1):
            flattened_array.append(pad_symbol)

        flattened_array.append(".")
        yield flattened_array
