import torch
from torch.utils.data import Dataset

class SetDataset(Dataset):
    def __init__(self, tokenized_combinations):
        self.combinations = tokenized_combinations

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        sequence = self.combinations[idx]
        return torch.tensor(sequence)


class BalancedSetDataset(Dataset):
    def __init__(self, set_sequences, non_set_sequences):
        self.set_sequences = set_sequences
        self.non_set_sequences = non_set_sequences
        self.length = (
            min(len(set_sequences), len(non_set_sequences)) * 2
        )  # 50/50 split in each batch

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Alternate between picking from sets and non-sets
        if idx % 2 == 0:
            return torch.tensor(self.set_sequences[idx // 2])  # Set
        return torch.tensor(self.non_set_sequences[idx // 2])  # Non-set
    

class BalancedTriplesSetDataset(Dataset):
    def __init__(self, no_set_sequences, one_set_sequences, two_set_sequences):
        self.no_set_sequences = no_set_sequences
        self.one_set_sequences = one_set_sequences
        self.two_set_sequences = two_set_sequences

        self.length = (
            min(len(no_set_sequences), min(len(one_set_sequences), len(two_set_sequences))) * 3
        )  

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx % 3 == 0:
            return torch.tensor(self.two_set_sequences[idx // 3])  # One set
        elif idx % 3 == 1:
            return torch.tensor(self.one_set_sequences[idx // 3])  # Two sets
        return torch.tensor(self.no_set_sequences[idx // 3])  # No sets
