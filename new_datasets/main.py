import torch
from model import GPT, GPTConfig44_SeededOrigDataset, GPTConfig44_Complete
from data_utils import initialize_triples_datasets, initialize_loaders
import random
import numpy as np
from set_transformer_small import run, calculate_accuracy
import os
import sys

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

if __name__ == "__main__":

    # for curr_seed in [100, 200, 300, 400, 500]:

    if len(sys.argv) > 1:
        curr_seed = int(sys.argv[1])
    else:
        raise ValueError("Seed value must be provided as a command line argument")

    seed_dir_path = f"{PATH_PREFIX}/seed{curr_seed}"
    if not os.path.exists(seed_dir_path):
        os.makedirs(seed_dir_path)
    
    torch.manual_seed(curr_seed)
    random.seed(curr_seed)
    np.random.seed(curr_seed)

    config = GPTConfig44_SeededOrigDataset(seed = curr_seed)
    run(
        config,
        dataset_path=config.dataset_path
    )
        

    # seed = 42
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)

    # small_combinations = run()

    # Attempt to improve model accuracy

    # config = GPTConfig44_Complete()
    # config = GPTConfig24_Complete()
    # config = GPTConfig34_Complete()
    
    # curr_seed = 4
    # config = GPTConfig44_Seeded(seed = curr_seed)

    # torch.manual_seed(config.seed)
    # random.seed(config.seed)
    # np.random.seed(config.seed)

    # run(
    #     config,
    #     dataset_path=config.dataset_path
    # )

    # run(
    #     config,
    #     dataset_path=config.dataset_path
    # )

    # # PIPELINE - Calculate accuracy for complete model on base random dataset, SEEDED DATASETS
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # dataset_path = f"{PATH_PREFIX}/base_card_randomization_tuple_randomization_dataset.pth"
    # dataset = torch.load(dataset_path)

    # config = GPTConfig44_Complete()
    # orig_model = GPT(config).to(device)
    # orig_checkpoint = torch.load(f"{config.filename}", weights_only=False)
    # orig_model.load_state_dict(orig_checkpoint["model"])
    
    # print("Validation accuracy on base random dataset with corresponding seeds")
    # for curr_seed in [1, 2, 3, 4]:
    #     config = GPTConfig44_Seeded(seed = curr_seed)

    #     torch.manual_seed(config.seed)
    #     random.seed(config.seed)
    #     np.random.seed(config.seed)

    #     train_loader, val_loader = initialize_loaders(dataset)

    #     model = GPT(config).to(device)
    #     checkpoint = torch.load(f"{config.filename}", weights_only=False)
    #     model.load_state_dict(checkpoint["model"])

    #     val_accuracy = calculate_accuracy(
    #         model=model, 
    #         dataloader=val_loader,
    #         config=config, 
    #         tokenizer_path=config.tokenizer_path,
    #         # save_incorrect_path=f'{PATH_PREFIX}/seed{curr_seed}/complete_baserandom_val_incorrect_predictions_orig_seed_42.txt',
    #         breakdown=True)
    #     print(f"Val accuracy for model{curr_seed} on dataset with seed {curr_seed}: ", val_accuracy)

    #     config = GPTConfig44_Complete()
    #     val_accuracy = calculate_accuracy(
    #         model=orig_model, 
    #         dataloader=val_loader,
    #         config=config, 
    #         tokenizer_path=config.tokenizer_path,
    #         breakdown=True)
    #     print(f"Val accuracy for orig model on dataset with seed {curr_seed}: ", val_accuracy)


    # # PIPELINE - Calculate accuracy for complete model on base random dataset, ORIGINAL DATASET
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # dataset_path = f"{PATH_PREFIX}/base_card_randomization_tuple_randomization_dataset.pth"
    # dataset = torch.load(dataset_path)
    # train_loader, val_loader = initialize_loaders(dataset)

    # print("Validation accuracy on base random dataset with seed 42")
    # for curr_seed in [1, 2, 3, 4]:
    #     config = GPTConfig44_Seeded(seed = curr_seed)

    #     # torch.manual_seed(config.seed)
    #     # random.seed(config.seed)
    #     # np.random.seed(config.seed)

    #     model = GPT(config).to(device)
    #     checkpoint = torch.load(f"{config.filename}", weights_only=False)
    #     model.load_state_dict(checkpoint["model"])

    #     val_accuracy = calculate_accuracy(
    #         model=model, 
    #         dataloader=val_loader,
    #         config=config, 
    #         tokenizer_path=config.tokenizer_path,
    #         save_incorrect_path=f'{PATH_PREFIX}/seed{curr_seed}/complete_baserandom_val_incorrect_predictions_orig_seed_42.txt',
    #         breakdown=True)
    #     print(f"Val accuracy for seed {curr_seed}: ", val_accuracy)




    # # PIPELINE - Calculate accuracy for complete model on base random dataset
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # dataset_path = f"{PATH_PREFIX}/base_card_randomization_tuple_randomization_dataset.pth"
    # model = GPT(config).to(device)
    # checkpoint = torch.load(f"{PATH_PREFIX}/{config.filename}", weights_only=False)
    # model.load_state_dict(checkpoint["model"])

    # dataset = torch.load(dataset_path)
    # train_loader, val_loader = initialize_loaders(config, dataset)

    # complete_baserandom_val_accuracy = calculate_accuracy(
    #     model=model, 
    #     dataloader=val_loader,
    #     config=config, 
    #     tokenizer_path=config.tokenizer_path,
    #     save_incorrect_path=f'{PATH_PREFIX}/complete_baserandom_val_incorrect_predictions_34.txt',
    #     breakdown=True)
    # print("Val accuracy for complete model with 3 layers 4 heads on base random dataset: ", complete_baserandom_val_accuracy)

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

