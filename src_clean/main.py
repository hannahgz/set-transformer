import torch
from model import GPTConfig44_Equal
from data_utils import initialize_triples_datasets
import random
import numpy as np
from set_transformer_small import run

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

if __name__ == "__main__":
    # small_combinations = run()
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Attempt to improve model accuracy
    print("Initializing dataset")
    config = GPTConfig44_Equal()
    dataset_path = f'{PATH_PREFIX}/equal_causal_balanced_dataset.pth'
    tokenizer_path = f'{PATH_PREFIX}/equal_causal_balanced_tokenizer.pkl'
    dataset = initialize_triples_datasets(
        config,
        save_dataset_path=dataset_path,
        save_tokenizer_path=tokenizer_path
    )

    print("Running model")
    run(
        config,
        dataset_path=dataset_path
    )

