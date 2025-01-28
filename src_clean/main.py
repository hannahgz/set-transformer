import torch
from model import GPTConfig44_Equal, GPT
from data_utils import initialize_triples_datasets, initialize_loaders, find_paired_sequence
import random
import numpy as np
from set_transformer_small import run, calculate_accuracy

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

if __name__ == "__main__":
    # small_combinations = run()
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Attempt to improve model accuracy
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # config = GPTConfig44_Equal()
    # dataset_path = f'{PATH_PREFIX}/equal_causal_balanced_dataset.pth'
    # tokenizer_path = f'{PATH_PREFIX}/equal_causal_balanced_tokenizer.pkl'
    # dataset_path = f'{PATH_PREFIX}/final_causal_balanced_dataset.pth'
    # tokenizer_path = f'{PATH_PREFIX}/final_causal_balanced_tokenizer.pkl'
    # dataset_path = f'{PATH_PREFIX}/balanced_set_dataset_random.pth'
    # tokenizer_path=f"{PATH_PREFIX}/balanced_set_dataset_random_tokenizer.pkl"
    # dataset_path=f"{PATH_PREFIX}/base_card_randomization_tuple_randomization_dataset.pth"
    # tokenizer_path=f"{PATH_PREFIX}/base_card_randomization_tuple_randomization_tokenizer.pkl"

    # dataset_path = f'{PATH_PREFIX}/base_dataset.pth'
    # tokenizer_path=f"{PATH_PREFIX}/base_tokenizer.pkl"
    # dataset = torch.load(dataset_path)
    # find_paired_sequence(
    #     dataset=dataset,
    #     tokenizer_path=tokenizer_path,
    #     target_sequence=["A", "oval", "A", "green", "A", "one", "A", "solid", "B", "oval", "B", "blue", "B", "one", "B", "solid", "C", "oval", "C", "pink", "C", "one", "C", "solid", "D", "oval", "D", "green", "D", "two", "D", "solid", "E", "oval", "E", "green", "E", "three", "E", "solid", ">", "A", "B", "C", "/", "A", "D", "E", "."]
    # )
    # print("Initializing dataset")
    # dataset = initialize_triples_datasets(
    #     config,
    #     save_dataset_path=dataset_path,
    #     save_tokenizer_path=tokenizer_path
    # )

    # print("Running model")
    # run(
    #     config,
    #     dataset_path=dataset_path
    # )


    # model = GPT(config).to(device)
    # checkpoint = torch.load(f"{PATH_PREFIX}/{config.filename}", weights_only=False)
    # model.load_state_dict(checkpoint["model"])

    # dataset = torch.load(dataset_path)

    # train_loader, val_loader = initialize_loaders(config, dataset)

    # equal_baserandom_val_accuracy = calculate_accuracy(
    #     model=model, 
    #     dataloader=val_loader,
    #     config=config, 
    #     tokenizer_path=tokenizer_path,
    #     save_incorrect_path=f'{PATH_PREFIX}/equal_baserandom_val_incorrect_predictions.txt',
    #     breakdown=True)
    # print("Val accuracy for equal model on base random dataset: ", equal_baserandom_val_accuracy)

    # equal_base_train_accuracy = calculate_accuracy(
    #     model=model, 
    #     dataloader=train_loader,
    #     config=config, 
    #     tokenizer_path=tokenizer_path,
    #     save_incorrect_path=f'{PATH_PREFIX}/equal_base_train_incorrect_predictions.txt',
    #     breakdown=True)
    # print("Train accuracy for equal model on base dataset: ", equal_base_train_accuracy)

    # equal_orig_train_accuracy = calculate_accuracy(
    #     model=model, 
    #     dataloader=train_loader,
    #     config=config, 
    #     tokenizer_path=tokenizer_path,
    #     save_incorrect_path=f'{PATH_PREFIX}/equal_orig_train_incorrect_predictions.txt',
    #     breakdown=True)
    # print("Train accuracy for equal model on orig dataset: ", equal_orig_train_accuracy)


    # equal_equal_val_accuracy = calculate_accuracy(
    #     model=model, 
    #     dataloader=val_loader,
    #     config=config, 
    #     tokenizer_path=tokenizer_path,
    #     save_incorrect_path=f'{PATH_PREFIX}/equal_equal_val_incorrect_predictions.txt',
    #     breakdown=True)
    # print("Val accuracy for final model on final dataset: ", equal_equal_val_accuracy)

    # equal_equal_train_accuracy = calculate_accuracy(
    #     model=model, 
    #     dataloader=train_loader,
    #     config=config, 
    #     tokenizer_path=tokenizer_path,
    #     breakdown=True)
    # print("Train accuracy for final model on final dataset: ", equal_equal_train_accuracy)

    # equal_final_val_accuracy = calculate_accuracy(
    #     model=model, 
    #     dataloader=val_loader,
    #     config=config, 
    #     tokenizer_path=tokenizer_path,
    #     save_incorrect_path=f'{PATH_PREFIX}/equal_final_incorrect_predictions.txt',
    #     breakdown=True)
    # print("Val accuracy for equal model on final dataset: ", equal_final_val_accuracy)

