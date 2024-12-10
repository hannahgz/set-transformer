from graph import lineplot_specific
from model import GPTConfig44
from data_utils import pretty_print_input

if __name__ == "__main__":

    # test_input = [
    #     "A", "squiggle", "B", "squiggle", "C", "squiggle", "D", "squiggle", "E", "squiggle",
    #     "A", "striped", "B", "striped", "C", "striped", "D", "striped", "E", "striped", 
    #     "A", "one",  "B", "two",  "C", "one",  "D", "three",  "E", "one",
    #     "A", "green", "B", "green", "C", "blue", "D", "green", "E", "pink", 
    #     ">", "A", "B", "D", "/", "A", "C", "E", "."
    # ]
    
    # lineplot_specific(
    #     config=GPTConfig44,
    #     input=test_input,
    #     get_prediction=True,
    #     filename_prefix="test1"
    # )

    test_input = [
        "E", "striped", "B", "green", "D", "two", "B", "oval", "C", "green", "D", "green", "D", "solid", "E", "two", "B", "one", "D", "oval", "E", "green", "C", "one", "A", "green", "C", "open", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
        ">", "A", "B", "C", ".", ".", "_", "_", "_"
    ]

    lineplot_specific(
        config=GPTConfig44,
        input=test_input,
        get_prediction=True,
        filename_prefix="input0"
    )

    test_input = [
        "E", "striped", "B", "green", "D", "two", "B", "squiggle", "C", "green", "D", "green", "D", "solid", "E", "two", "B", "one", "D", "oval", "E", "green", "C", "one", "A", "green", "C", "open", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
        ">", "*", "_", "_", "_", "_", "_", "_", "_"
    ]

    lineplot_specific(
        config=GPTConfig44,
        input=test_input,
        get_prediction=True,
        filename_prefix="input0_modified_shape"
    )

    test_input = [
        "E", "striped", "B", "pink", "D", "two", "B", "oval", "C", "green", "D", "green", "D", "solid", "E", "two", "B", "one", "D", "oval", "E", "green", "C", "one", "A", "green", "C", "open", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
        ">", "*", "_", "_", "_", "_", "_", "_", "_"
    ]

    lineplot_specific(
        config=GPTConfig44,
        input=test_input,
        get_prediction=True,
        filename_prefix="input0_modified_color"
    )

    test_input = [
        "E", "striped", "B", "green", "D", "two", "B", "oval", "C", "green", "D", "green", "D", "solid", "E", "three", "B", "one", "D", "oval", "E", "green", "C", "one", "A", "green", "C", "open", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
        ">", "A", "B", "C", "/", "C", "D", "E", "."
    ]

    lineplot_specific(
        config=GPTConfig44,
        input=test_input,
        get_prediction=True,
        filename_prefix="input0_number"
    )


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
