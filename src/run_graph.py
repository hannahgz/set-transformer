from graph import lineplot_specific
from model import GPTConfig44
from data_utils import pretty_print_input

if __name__ == "__main__":

    test_input = [
        "A", "squiggle", "B", "squiggle", "C", "squiggle", "D", "squiggle", "E", "squiggle",
        "A", "striped", "B", "striped", "C", "striped", "D", "striped", "E", "striped", 
        "A", "one",  "B", "two",  "C", "one",  "D", "three",  "E", "one",
        "A", "green", "B", "green", "C", "blue", "D", "green", "E", "pink", 
        ">", "A", "B", "D", "/", "A", "C", "E", "."
    ]

    pretty_print_input(test_input)
    
    # lineplot_specific(
    #     config=GPTConfig44,
    #     input=test_input,
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
