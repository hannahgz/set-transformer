import pickle
from collections import defaultdict


class Tokenizer:
    def __init__(self):
        self.token_to_id = defaultdict(lambda: len(self.token_to_id))
        self.id_to_token = {}

    def encode(self, sequence):
        return [self.token_to_id[token] for token in sequence]

    def decode(self, ids):
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        return [self.id_to_token[id] for id in ids]

# Saving the tokenizer
def save_tokenizer(tokenizer, filename):
    # Convert defaultdict to a regular dict for serialization
    tokenizer_data = {
        "token_to_id": dict(tokenizer.token_to_id),
        "id_to_token": tokenizer.id_to_token
    }
    with open(filename, 'wb') as f:
        pickle.dump(tokenizer_data, f)

# Loading the tokenizer
def load_tokenizer(filename):
    with open(filename, 'rb') as f:
        tokenizer_data = pickle.load(f)

    # Restore data into a Tokenizer instance
    tokenizer = Tokenizer()
    tokenizer.token_to_id = defaultdict(lambda: len(
        tokenizer.token_to_id), tokenizer_data["token_to_id"])
    tokenizer.id_to_token = tokenizer_data["id_to_token"]
    return tokenizer
