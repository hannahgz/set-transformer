import os
import torch
from torch import optim
import wandb
from model import GPT
# from model import GPTConfig24, GPTConfig42, GPTConfig44, GPTConfig, add_causal_masking, GPTConfig48, GPTConfig44_Patience20, GPTConfig44_AttrFirst
from model import GPTConfig44, GPTConfig44TriplesEmbdDrop, GPTConfig44_AttrFirst, GPTConfig44_BalancedSets, GPTConfig44_Final, GPTConfig44_FinalLR
from model import add_causal_masking
from data_utils import initialize_datasets, initialize_loaders, initialize_triples_datasets
import random
import numpy as np
from tokenizer import load_tokenizer
import pickle
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

if __name__ == "__main__":
    # small_combinations = run()
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Breakdown accuracy
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = f'{PATH_PREFIX}/triples_balanced_set_dataset_random.pth'
    dataset = torch.load(dataset_path)
    breakpoint()

