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
