from graph import make_prediction_given_input, generate_lineplot, lineplot_specific
from model import GPT, GPTConfig44_Complete
import random
import torch
from tokenizer import load_tokenizer

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'
# PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'
if __name__ == "__main__":

    # input1 = ["A", "oval", "A", "green", "A", "one", "A", "solid", "B", "oval", "B", "blue", "B", "one", "B", "solid", "C", "oval", "C", "pink", "C", "one", "C", "solid", "D", "oval", "D", "green", "D", "two", "D", "solid", "E", "oval", "E", "green", "E", "three", "E", "solid", ">", "A", "B", "C", "/", "A", "D", "E", "."]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig44_Complete()

    model = GPT(config).to(device)
    checkpoint = torch.load(f"{PATH_PREFIX}/{config.filename}", weights_only=False)
    model.load_state_dict(checkpoint["model"])

    tokenizer = load_tokenizer(config.tokenizer_path)
    # make_prediction_given_input(
    #     GPTConfig44_Equal,
    #     input1,
    #     tokenizer=tokenizer,
    #     model=model,
    #     get_prediction=True
    # )

    test_input = [
        "A", "oval", "A", "green", "A", "one", "A", "solid",
        "B", "oval", "B", "blue", "B", "one", "B", "solid",
        "C", "oval", "C", "pink", "C", "one", "C", "solid",
        "D", "oval", "D", "green", "D", "two", "D", "solid",
        "E", "oval", "E", "green", "E", "three", "E", "solid",
        ">", "A", "D", "E", "/", "A", "B", "C", "."
    ]

    lineplot_specific(
        config=config,
        input=test_input,
        tokenizer_path=config.tokenizer_path,
        get_prediction=True,
        filename_prefix="input0"
    )

    # generate_lineplot(
    #     config,
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    #     dataset_path,
    #     tokenizer_path,
    #     use_labels=True,
    #     threshold=0.05,
    #     get_prediction=True)

    # input1 = [
    #     "A", "diamond",      "B", "diamond",      "C", "diamond",   "D", "oval",    "E", "oval",
    #     "A", "striped",      "B", "striped",      "C", "striped",   "D", "open",    "E", "open",
    #     "A", "one",          "B", "one",          "C", "one",       "D", "three",   "E", "three",
    #     "A", "green",        "B", "green",        "C", "green",     "D", "green",   "E", "pink",
    #     ">", "A", "B", "C", ".", "_", "_", "_", "_"
    # ]

    # input2 = [
    #     "A", "diamond",      "B", "diamond",      "C", "diamond",   "D", "oval",    "E", "oval",
    #     "A", "striped",      "B", "striped",      "C", "striped",   "D", "open",    "E", "open",
    #     "A", "one",          "B", "one",          "C", "one",       "D", "three",   "E", "three",
    #     "A", "green",        "B", "green",        "C", "pink",     "D", "green",   "E", "pink",
    #     ">", "*", ".", "_", "_", "_", "_", "_", "_"
    # ]

    # input1 = [
    #     "E", "striped", "B", "green", "D", "two", "B", "oval", "C", "green", "D", "green", "D", "solid", "E", "two", "B", "one", "D", "oval", "E", "green", "C", "one", "A", "green", "C", "open", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
    #     ">", "A", "B", "C", ".", "_", "_", "_", "_"
    # ]

    # input2 = [
    #     "E", "striped", "B", "pink", "D", "two", "B", "oval", "C", "green", "D", "green", "D", "solid", "E", "two", "B", "one", "D", "oval", "E", "green", "C", "one", "A", "green", "C", "open", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
    #     ">", "*", ".", "_", "_", "_", "_", "_", "_"
    # ]

    # lineplot_difference_inputs(
    #     GPTConfig44_BalancedSets,
    #     input1,
    #     input2,
    #     # tokenizer_path=f'{PATH_PREFIX}/larger_balanced_set_dataset_random_tokenizer.pkl',
    #     tokenizer_path=f'larger_balanced_set_dataset_random_tokenizer.pkl',
    #     get_prediction=True,
    #     filename_prefix="NEW_input1_input2_allsame"
    # )

    # generate_lineplot(
    #     GPTConfig44_BalancedSets,
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     dataset_path=f'{PATH_PREFIX}/larger_balanced_set_dataset_random.pth',
    #     tokenizer_path=f'{PATH_PREFIX}/larger_balanced_set_dataset_random_tokenizer.pkl',
    #     use_labels=True,
    #     threshold=0.05,
    #     get_prediction=True)

    # # PREVIOUS CODE BEFORE 1/6/2025

    # test_input = [
    #     "E", "striped", "B", "green", "D", "two", "B", "oval", "C", "green", "D", "green", "D", "solid", "E", "two", "B", "one", "D", "oval", "E", "green", "C", "one", "A", "green", "C", "open", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
    #     ">", "A", "B", "C", ".", "_", "_", "_", "_"
    # ]

    # lineplot_specific(
    #     config=GPTConfig44_BalancedSets,
    #     input=test_input,
    #     # tokenizer_path=f'{PATH_PREFIX}/larger_balanced_set_dataset_random_tokenizer.pkl',
    #     tokenizer_path=f'larger_balanced_set_dataset_random_tokenizer.pkl',
    #     get_prediction=True,
    #     filename_prefix="input0"
    # )

    # # Step 1: Extract the first 40 elements and group into pairs
    # pairs = [(test_input[i], test_input[i + 1]) for i in range(0, 40, 2)]

    # # Step 2: Sort the pairs alphabetically by the first element of each pair
    # pairs = sorted(pairs, key=lambda x: x[0])
    # random.shuffle(pairs)

    # # Step 3: Flatten the sorted pairs back into a single list
    # sorted_flattened = [item for pair in pairs for item in pair]

    # # Step 4: Replace the first 40 elements with the sorted sequence
    # test_input[:40] = sorted_flattened

    # print("Sorted test input")

    # # test_input = [
    # #     "E", "striped", "B", "pink", "D", "one", "B", "squiggle", "C", "green", "D", "green", "D", "solid", "E", "two", "B", "one", "D", "oval", "E", "green", "C", "two", "A", "pink", "C", "solid", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
    # #     ">", "*", ".", "_", "_", "_", "_", "_", "_"
    # # ]

    # lineplot_specific(
    #     config=GPTConfig44_BalancedSets,
    #     input=test_input,
    #     # tokenizer_path=f'{PATH_PREFIX}/larger_balanced_set_dataset_random_tokenizer.pkl',
    #     tokenizer_path=f'larger_balanced_set_dataset_random_tokenizer.pkl',
    #     get_prediction=True,
    #     filename_prefix="input0_modified_shape"
    # )

    # # Step 1: Extract the first 40 elements and group into pairs
    # pairs = [(test_input[i], test_input[i + 1]) for i in range(0, 40, 2)]

    # # Step 2: Sort the pairs alphabetically by the first element of each pair
    # pairs = sorted(pairs, key=lambda x: x[0])
    # random.shuffle(pairs)

    # # Step 3: Flatten the sorted pairs back into a single list
    # sorted_flattened = [item for pair in pairs for item in pair]

    # # Step 4: Replace the first 40 elements with the sorted sequence
    # test_input[:40] = sorted_flattened

    # print("Sorted test input")

    # # test_input = [
    # #     "E", "striped", "B", "pink", "D", "one", "B", "squiggle", "C", "green", "D", "green", "D", "solid", "E", "two", "B", "one", "D", "oval", "E", "green", "C", "two", "A", "pink", "C", "solid", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
    # #     ">", "*", ".", "_", "_", "_", "_", "_", "_"
    # # ]

    # lineplot_specific(
    #     config=GPTConfig44_BalancedSets,
    #     input=test_input,
    #     # tokenizer_path=f'{PATH_PREFIX}/larger_balanced_set_dataset_random_tokenizer.pkl',
    #     tokenizer_path=f'larger_balanced_set_dataset_random_tokenizer.pkl',
    #     get_prediction=True,
    #     filename_prefix="input0_modified_shape"
    # )

    # # Step 1: Extract the first 40 elements and group into pairs
    # pairs = [(test_input[i], test_input[i + 1]) for i in range(0, 40, 2)]

    # # Step 2: Sort the pairs alphabetically by the first element of each pair
    # pairs = sorted(pairs, key=lambda x: x[0])
    # random.shuffle(pairs)

    # # Step 3: Flatten the sorted pairs back into a single list
    # sorted_flattened = [item for pair in pairs for item in pair]

    # # Step 4: Replace the first 40 elements with the sorted sequence
    # test_input[:40] = sorted_flattened

    # print("Sorted test input")

    # # test_input = [
    # #     "E", "striped", "B", "pink", "D", "one", "B", "squiggle", "C", "green", "D", "green", "D", "solid", "E", "two", "B", "one", "D", "oval", "E", "green", "C", "two", "A", "pink", "C", "solid", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
    # #     ">", "*", ".", "_", "_", "_", "_", "_", "_"
    # # ]

    # lineplot_specific(
    #     config=GPTConfig44_BalancedSets,
    #     input=test_input,
    #     # tokenizer_path=f'{PATH_PREFIX}/larger_balanced_set_dataset_random_tokenizer.pkl',
    #     tokenizer_path=f'larger_balanced_set_dataset_random_tokenizer.pkl',
    #     get_prediction=True,
    #     filename_prefix="input0_modified_shape"
    # )

    # # Step 1: Extract the first 40 elements and group into pairs
    # pairs = [(test_input[i], test_input[i + 1]) for i in range(0, 40, 2)]

    # # Step 2: Sort the pairs alphabetically by the first element of each pair
    # pairs = sorted(pairs, key=lambda x: x[0])
    # random.shuffle(pairs)

    # # Step 3: Flatten the sorted pairs back into a single list
    # sorted_flattened = [item for pair in pairs for item in pair]

    # # Step 4: Replace the first 40 elements with the sorted sequence
    # test_input[:40] = sorted_flattened

    # print("Sorted test input")

    # # test_input = [
    # #     "E", "striped", "B", "pink", "D", "one", "B", "squiggle", "C", "green", "D", "green", "D", "solid", "E", "two", "B", "one", "D", "oval", "E", "green", "C", "two", "A", "pink", "C", "solid", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
    # #     ">", "*", ".", "_", "_", "_", "_", "_", "_"
    # # ]

    # lineplot_specific(
    #     config=GPTConfig44_BalancedSets,
    #     input=test_input,
    #     # tokenizer_path=f'{PATH_PREFIX}/larger_balanced_set_dataset_random_tokenizer.pkl',
    #     tokenizer_path=f'larger_balanced_set_dataset_random_tokenizer.pkl',
    #     get_prediction=True,
    #     filename_prefix="input0_modified_shape"
    # )

    # # test_input = [
    #     "E", "striped", "B", "pink", "D", "two", "B", "oval", "C", "green", "D", "green", "D", "solid", "E", "two", "B", "one", "D", "oval", "E", "green", "C", "one", "A", "green", "C", "open", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
    #     ">", "*", ".", "_", "_", "_", "_", "_", "_"
    # ]

    # lineplot_specific(
    #     config=GPTConfig44_BalancedSets,
    #     input=test_input,
    #     tokenizer_path=f'{PATH_PREFIX}/larger_balanced_set_dataset_random_tokenizer.pkl',
    #     get_prediction=True,
    #     filename_prefix="input0_modified_color"
    # )

    # test_input = [
    #     "E", "striped", "B", "green", "D", "two", "B", "oval", "C", "green", "D", "green", "D", "solid", "E", "three", "B", "one", "D", "oval", "E", "green", "C", "one", "A", "green", "C", "open", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
    #     ">", "A", "B", "C", "/", "C", "D", "E", "."
    # ]

    # lineplot_specific(
    #     config=GPTConfig44_BalancedSets,
    #     input=test_input,
    #     tokenizer_path=f'{PATH_PREFIX}/larger_balanced_set_dataset_random_tokenizer.pkl',
    #     get_prediction=True,
    #     filename_prefix="input0_number"
    # )




    # OLD BELOW THIS

    # test_input = [
    #     "A", "diamond",      "B", "diamond",      "C", "diamond",
    #     "A", "striped",      "B", "striped",      "C", "striped",
    #     "A", "one",          "B", "one",          "C", "one",
    #     "A", "green",        "B", "green",        "C", "green",     "D", "green",   "E", "pink", "D", "oval",    "E", "oval", "D", "open",    "E", "open", "D", "three",   "E", "three",
    #     ">", "A", "B", "C", ".", ".", ".", ".", "."
    # ]

    # lineplot_specific(
    #     config=GPTConfig44,
    #     input=test_input,
    #     get_prediction=True,
    #     filename_prefix="synthetic1"
    # )

    # test_input = [
    #     "A", "diamond",      "B", "diamond",      "C", "diamond",   "D", "oval",    "E", "oval",
    #     "A", "striped",      "B", "striped",      "C", "striped",   "D", "open",    "E", "open",
    #     "A", "one",          "B", "one",          "C", "one",       "D", "three",   "E", "three",
    #     "A", "green",        "B", "green",        "C", "green",     "D", "green",   "E", "pink",
    #     ">", "A", "B", "C", ".", ".", ".", ".", "."
    # ]

    # lineplot_specific(
    #     config=GPTConfig44,
    #     input=test_input,
    #     get_prediction=True,
    #     filename_prefix="synthetic1"
    # )

    # test_input = [
    #     "A", "diamond",      "B", "diamond",      "C", "diamond",   "D", "oval",    "E", "oval",
    #     "A", "striped",      "B", "striped",      "C", "striped",   "D", "open",    "E", "open",
    #     "A", "one",          "B", "two",          "C", "three",       "D", "three",   "E", "three",
    #     "A", "green",        "B", "green",        "C", "green",     "D", "green",   "E", "pink",
    #     ">", "A", "B", "C", ".", ".", ".", ".", "."
    # ]

    # lineplot_specific(
    #     config=GPTConfig44,
    #     input=test_input,
    #     get_prediction=True,
    #     filename_prefix="synthetic1"
    # )

    # test_input = [
    #     "A", "squiggle",     "B", "squiggle",     "C", "squiggle",  "D", "oval",    "E", "oval",
    #     "A", "striped",      "B", "striped",      "C", "striped",   "D", "open",    "E", "open",
    #     "A", "one",          "B", "one",          "C", "one",       "D", "three",   "E", "three",
    #     "A", "green",        "B", "green",        "C", "green",     "D", "green",   "E", "pink",
    #     ">", "A", "B", "C", ".", ".", ".", ".", "."
    # ]

    # lineplot_specific(
    #     config=GPTConfig44,
    #     input=test_input,
    #     get_prediction=True,
    #     filename_prefix="synthetic1"
    # )

    # test_input = [
    #     "A", "striped",      "B", "striped",      "C", "striped",   "D", "open",    "E", "open",
    #     "A", "squiggle",     "B", "squiggle",     "C", "squiggle",  "D", "oval",    "E", "oval",
    #     "A", "one",          "B", "one",          "C", "one",       "D", "three",   "E", "three",
    #     "A", "green",        "B", "green",        "C", "green",     "D", "green",   "E", "pink",
    #     ">", "A", "B", "C", ".", ".", ".", ".", "."
    # ]

    # lineplot_specific(
    #     config=GPTConfig44,
    #     input=test_input,
    #     get_prediction=True,
    #     filename_prefix="synthetic1"
    # )

    # test_input = [
    #     "A", "one",          "B", "one",          "C", "one",       "D", "three",   "E", "three",
    #     "A", "striped",      "B", "striped",      "C", "striped",   "D", "open",    "E", "open",
    #     "A", "squiggle",     "B", "squiggle",     "C", "squiggle",  "D", "oval",    "E", "oval",
    #     "A", "green",        "B", "green",        "C", "green",     "D", "green",   "E", "pink",
    #     ">", "A", "B", "C", ".", ".", ".", ".", "."
    # ]

    # lineplot_specific(
    #     config=GPTConfig44,
    #     input=test_input,
    #     get_prediction=True,
    #     filename_prefix="synthetic1"
    # )

    # test_input = [
    #     "A", "green",        "B", "green",        "C", "green",     "D", "green",   "E", "pink",
    #     "A", "one",          "B", "one",          "C", "one",       "D", "three",   "E", "three",
    #     "A", "striped",      "B", "striped",      "C", "striped",   "D", "open",    "E", "open",
    #     "A", "squiggle",     "B", "squiggle",     "C", "squiggle",  "D", "oval",    "E", "oval",
    #     ">", "A", "B", "C", ".", ".", ".", ".", "."
    # ]

    # lineplot_specific(
    #     config=GPTConfig44,
    #     input=test_input,
    #     get_prediction=True,
    #     filename_prefix="synthetic1"
    # )

    # test_input = [
    #     "A", "squiggle",     "B", "squiggle",     "C", "squiggle",  "D", "oval",    "E", "oval",
    #     "A", "striped",      "B", "striped",      "C", "striped",   "D", "open",    "E", "open",
    #     "A", "one",          "B", "one",          "C", "one",       "D", "three",   "E", "three",
    #     "A", "green",        "B", "blue",         "C", "green",     "D", "green",   "E", "pink",
    #     ">", "*", ".", ".", ".", ".", ".", ".", "."
    # ]

    # lineplot_specific(
    #     config=GPTConfig44,
    #     input=test_input,
    #     get_prediction=True,
    #     filename_prefix="synthetic2"
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
