import torch
from model import GPT, GPTConfig44_SeededOrigDataset, GPTConfig44_Complete
from data_utils import initialize_triples_datasets, initialize_loaders
import random
import numpy as np
from set_transformer_small import run, calculate_accuracy
import os
import sys

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

# Get seed from command line argument
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
