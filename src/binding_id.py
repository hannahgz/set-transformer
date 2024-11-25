import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_utils import initialize_loaders
from model import GPT
import os
from itertools import permutations

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'



def construct_binding_id_dataset(config, dataset_name, capture_layer):

    perms = list(permutations(range(20), 2))
    breakpoint()

    dataset_path = f"{PATH_PREFIX}/{dataset_name}.pth"
    dataset = torch.load(dataset_path)
    train_loader, val_loader = initialize_loaders(config, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)

    X = []
    y = []

    for index, batch in enumerate(val_loader):
        batch = batch.to(device)
        _, _, _, captured_embedding = model(batch, True, capture_layer)
        # torch.Size([64, 49, 64])
        # [batch_size, seq_len, embedding_dim]

        for index, indiv_embedding in enumerate(captured_embedding):
            curr_embedding = indiv_embedding[:40]
            curr_tokens = batch[index][:40]
            for (element1_index, element2_index) in perms:
                element1 = curr_embedding[element1_index]
                element2 = curr_embedding[element2_index]

                token1 = curr_tokens[element1_index - 1]
                token2 = curr_tokens[element2_index - 1]

                X.append(torch.cat((element1, element2)))
                if (token1 == token2):
                    y.append(1)
                else:
                    y.append(0)
                
                breakpoint()


    base_dir = f"{PATH_PREFIX}/binding_id/{dataset_name}/layer{capture_layer}"
    os.makedirs(base_dir, exist_ok=True)

    X_path = os.path.join(base_dir, "X.pt")
    y_path = os.path.join(base_dir, "y.pt")

    X_tensor = torch.stack(X)
    y_tensor = torch.tensor(y)

    breakpoint()
    torch.save(X_tensor, X_path)
    torch.save(y_tensor, y_path)

    return X_tensor, y_tensor
