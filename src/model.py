import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

# @dataclass
# class GPTConfig:
#     block_size: int = 20
#     vocab_size: int = 19
#     n_layer: int = 4
#     n_head: int = 4
#     n_embd: int = 64
#     dropout: float = 0.0
#     bias: bool = False
#     input_size: int = 16
#     target_size: int = 4
#     pad_symbol: str = "_"

# @dataclass
# class GPTConfigIndiv3:
#     block_size: int = 29
#     vocab_size: int = 19
#     n_layer: int = n_layer
#     n_head: int = n_head
#     n_embd: int = n_embd
#     dropout: float = 0.0
#     bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
#     input_size: int = 25 # (3 cards, 4 attributes/card, 12 * 2 = 24, + 1 for predict = 25)
#     target_size: int = 4
#     pad_symbol: str = "_"

lr = 1e-3
epochs = 2
batch_size = 32
n_layer = 4
n_head = 4
n_embd = 128
patience = 3
# eval_freq = 26
eval_freq = 0

# @dataclass
# class GPTConfig:
#     lr: float = 1e-3
#     epochs: int = 50
#     batch_size: int = 32
#     n_layer: int = 2
#     n_head: int = 2
#     n_embd: int = 64
#     patience: int = 3
#     eval_freq: int = 0
#     dropout: float = 0.0
#     n_cards: int = 5
#     block_size: int = 49
#     vocab_size: int = 21
#     bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
#     input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
#     target_size: int = 8
#     pad_symbol: str = "_"
#     out_dir: str = ""
#     filename: str = "test.pt"

# @dataclass
# class GPTConfig:
#     lr: float = 1e-3
#     epochs: int = 100
#     batch_size: int = 64
#     n_layer: int = 2
#     n_head: int = 2
#     n_embd: int = 64
#     patience: int = 5
#     eval_freq: int = 0
#     dropout: float = 0.0
#     n_cards: int = 5
#     block_size: int = 49
#     vocab_size: int = 22
#     bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
#     input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
#     target_size: int = 8
#     pad_symbol: str = "_"
#     out_dir: str = ""
#     filename: str = "full_run_random.pt"
#     end_of_seq_token: int = 13
#     padding_token: int = 14

# @dataclass
# class GPTConfig24:
#     lr: float = 1e-3
#     epochs: int = 100
#     batch_size: int = 64
#     n_layer: int = 2
#     n_head: int = 4
#     n_embd: int = 64
#     patience: int = 5
#     eval_freq: int = 0
#     dropout: float = 0.0
#     n_cards: int = 5
#     block_size: int = 49
#     vocab_size: int = 22
#     bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
#     input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
#     target_size: int = 8
#     pad_symbol: str = "_"
#     out_dir: str = ""
#     filename: str = "full_run_random_layers_2_heads_4.pt"
#     end_of_seq_token: int = 13
#     padding_token: int = 14

# @dataclass
# class GPTConfig42:
#     lr: float = 1e-3
#     epochs: int = 100
#     batch_size: int = 64
#     n_layer: int = 4
#     n_head: int = 2
#     n_embd: int = 64
#     patience: int = 5
#     eval_freq: int = 0
#     dropout: float = 0.0
#     n_cards: int = 5
#     block_size: int = 49
#     vocab_size: int = 22
#     bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
#     input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
#     target_size: int = 8
#     pad_symbol: str = "_"
#     out_dir: str = ""
#     filename: str = "full_run_random_layers_4_heads_2.pt"
#     end_of_seq_token: int = 13
#     padding_token: int = 14


# @dataclass
# class GPTConfig44:
#     lr: float = 1e-3
#     epochs: int = 100
#     batch_size: int = 64
#     n_layer: int = 4
#     n_head: int = 4
#     n_embd: int = 64
#     patience: int = 5
#     eval_freq: int = 0
#     dropout: float = 0.0
#     n_cards: int = 5
#     block_size: int = 49
#     vocab_size: int = 22
#     bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
#     input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
#     target_size: int = 8
#     pad_symbol: str = "_"
#     out_dir: str = ""
#     filename: str = "full_run_random_layers_4_heads_4.pt"
#     end_of_seq_token: int = 13
#     padding_token: int = 14

@dataclass
class GPTConfig:
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    patience: int = 5
    eval_freq: int = 0
    dropout: float = 0.0
    n_cards: int = 5
    block_size: int = 49
    vocab_size: int = 22
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
    target_size: int = 8
    pad_symbol: str = "_"
    out_dir: str = ""
    filename: str = "causal_full_run_random.pt"
    end_of_seq_token: int = 13
    padding_token: int = 14

@dataclass
class GPTConfig24:
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    patience: int = 5
    eval_freq: int = 0
    dropout: float = 0.0
    n_cards: int = 5
    block_size: int = 49
    vocab_size: int = 22
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
    target_size: int = 8
    pad_symbol: str = "_"
    out_dir: str = ""
    filename: str = "causal_full_run_random_layers_2_heads_4.pt"
    end_of_seq_token: int = 13
    padding_token: int = 14

@dataclass
class GPTConfig42:
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    n_layer: int = 4
    n_head: int = 2
    n_embd: int = 64
    patience: int = 5
    eval_freq: int = 0
    dropout: float = 0.0
    n_cards: int = 5
    block_size: int = 49
    vocab_size: int = 22
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
    target_size: int = 8
    pad_symbol: str = "_"
    out_dir: str = ""
    filename: str = "causal_full_run_random_layers_4_heads_2.pt"
    end_of_seq_token: int = 13
    padding_token: int = 14


@dataclass
class GPTConfig44:
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    patience: int = 5
    eval_freq: int = 0
    dropout: float = 0.0
    n_cards: int = 5
    block_size: int = 49
    vocab_size: int = 22
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
    target_size: int = 8
    pad_symbol: str = "_"
    out_dir: str = ""
    filename: str = "causal_full_run_random_layers_4_heads_4.pt"
    end_of_seq_token: int = 13
    padding_token: int = 14

@dataclass
class GPTConfig44_AttrFirst:
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    patience: int = 5
    eval_freq: int = 0
    dropout: float = 0.0
    n_cards: int = 5
    block_size: int = 49
    vocab_size: int = 22
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
    target_size: int = 8
    pad_symbol: str = "_"
    out_dir: str = ""
    filename: str = "attr_first_causal_full_run_random_layers_4_heads_4.pt"
    end_of_seq_token: int = 13
    padding_token: int = 14

@dataclass
class GPTConfig48:
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 64
    patience: int = 5
    eval_freq: int = 0
    dropout: float = 0.0
    n_cards: int = 5
    block_size: int = 49
    vocab_size: int = 22
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
    target_size: int = 8
    pad_symbol: str = "_"
    out_dir: str = ""
    filename: str = "causal_full_run_random_layers_4_heads_8.pt"
    end_of_seq_token: int = 13
    padding_token: int = 14


@dataclass
class GPTConfig44_Patience20:
    lr: float = 1e-3
    epochs: int = 400
    batch_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    patience: int = 20
    eval_freq: int = 0
    dropout: float = 0.0
    n_cards: int = 5
    block_size: int = 49
    vocab_size: int = 22
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
    target_size: int = 8
    pad_symbol: str = "_"
    out_dir: str = ""
    filename: str = "causal_full_run_random_layers_4_heads_4_patience_20.pt"
    end_of_seq_token: int = 13
    padding_token: int = 14


@dataclass
class GPTConfig44TriplesEmbd:
    lr: float = 1e-3
    epochs: int = 500
    batch_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    patience: int = 100
    dropout: float = 0.0
    n_cards: int = 5
    block_size: int = 49
    vocab_size: int = 22
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
    target_size: int = 8
    pad_symbol: str = "_"
    out_dir: str = ""
    filename: str = "triples_layers_4_heads_4_embd.pt"
    end_of_seq_token: int = 13
    padding_token: int = 14


@dataclass
class GPTConfig44TriplesEmbdDrop:
    lr: float = 1e-3
    epochs: int = 500
    batch_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    patience: int = 100
    dropout: float = 0.1
    n_cards: int = 5
    block_size: int = 49
    vocab_size: int = 22
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
    target_size: int = 8
    pad_symbol: str = "_"
    out_dir: str = ""
    filename: str = "triples_layers_4_heads_4_embd_dropout_0.1.pt"
    end_of_seq_token: int = 13
    padding_token: int = 14


@dataclass
class GPTConfig44TriplesLR:
    lr: float = 1e-2
    epochs: int = 500
    batch_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    patience: int = 100
    dropout: float = 0.0
    n_cards: int = 5
    block_size: int = 49
    vocab_size: int = 22
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
    target_size: int = 8
    pad_symbol: str = "_"
    out_dir: str = ""
    filename: str = "triples_layers_4_heads_4_lr.pt"
    end_of_seq_token: int = 13
    padding_token: int = 14

@dataclass
class GPTConfig44Triples:
    lr: float = 1e-3
    epochs: int = 500
    batch_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    patience: int = 100
    dropout: float = 0.0
    n_cards: int = 5
    block_size: int = 49
    vocab_size: int = 22
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
    target_size: int = 8
    pad_symbol: str = "_"
    out_dir: str = ""
    filename: str = "triples_layers_4_heads_4.pt"
    end_of_seq_token: int = 13
    padding_token: int = 14

@dataclass
class GPTConfig48Triples:
    lr: float = 1e-3
    epochs: int = 500
    batch_size: int = 64
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 64
    patience: int = 100
    dropout: float = 0.0
    n_cards: int = 5
    block_size: int = 49
    vocab_size: int = 22
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
    target_size: int = 8
    pad_symbol: str = "_"
    out_dir: str = ""
    filename: str = "triples_layers_4_heads_8.pt"
    end_of_seq_token: int = 13
    padding_token: int = 14

@dataclass
class GPTConfig88Triples:
    lr: float = 1e-3
    epochs: int = 500
    batch_size: int = 64
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 64
    patience: int = 100
    dropout: float = 0.0
    n_cards: int = 5
    block_size: int = 49
    vocab_size: int = 22
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
    target_size: int = 8
    pad_symbol: str = "_"
    out_dir: str = ""
    filename: str = "triples_layers_8_heads_8.pt"
    end_of_seq_token: int = 13
    padding_token: int = 14


@dataclass
class GPTConfig44_BalancedSets:
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    patience: int = 5
    eval_freq: int = 0
    dropout: float = 0.0
    n_cards: int = 5
    block_size: int = 49
    vocab_size: int = 22
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_size: int = 41 # (5 cards, 4 attributes/card, 20 * 2 = 40, + 1 for predict = 41)
    target_size: int = 8
    pad_symbol: str = "_"
    out_dir: str = ""
    filename: str = "larger_causal_balanced_random_layers_4_heads_4.pt"
    end_of_seq_token: int = 13
    padding_token: int = 14


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        # if not self.flash:
        #     print(
        #         "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
        #     )
        
        # UNCOMMENT
        # print("Using causal masking")
            # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # Calculate raw attention scores
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # UNCOMMENT
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att_weights = F.softmax(att, dim=-1)

            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # TODO: might need to comment this out if not using bias
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att_weights = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att_weights


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        attn_output, att_weights = self.attn(self.ln_1(x))
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, att_weights


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # TODO: probably manually put this in for now so don't have to reload the dataset everytime do the run
        end_of_seq_token = self.config.end_of_seq_token

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd, end_of_seq_token),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout), 
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # def forward(self, idx, targets=None):
    def forward(self, idx, get_loss=False, capture_layer=None):
        # device = idx.device
        # print("forward device: ", device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)

        # x = self.transformer.drop(tok_emb + pos_emb)
        x = tok_emb + pos_emb
        attention_weights = []
        capture_embedding = None

        for layer_idx, block in enumerate(self.transformer.h):
            x, att_weights = block(x)
            attention_weights.append(att_weights)

            # with torch.no_grad():
            #     if capture_layer is not None and capture_head is not None:
            #         if layer_idx == capture_layer:
            #             head_dim = x.size(-1) // self.config.n_head
            #             capture_embedding = x.reshape(b, t, self.config.n_head, head_dim)[:, :, capture_head, :].detach()

            with torch.no_grad():
                if capture_layer is not None:
                    if layer_idx == capture_layer:
                        capture_embedding = x.detach()

            torch.cuda.empty_cache()
        
        x = self.transformer.ln_f(x)

        if get_loss:
            logits = self.lm_head(x) # (batch_size, seq_length, vocab_size)
            logits_for_loss = logits[
                :, -(GPTConfig().target_size+1):-1, :
            ]  
            targets = idx[:, -GPTConfig().target_size :]
            
            loss = F.cross_entropy(
                logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                targets.reshape(-1),
                ignore_index=self.config.padding_token,
            )

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, attention_weights, capture_embedding

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # leaving this in, won't ever be triggered in our case
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # Greedy sampling
            idx_next = torch.argmax(probs, dim=-1).unsqueeze(-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    

def add_causal_masking(config, modified_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)
    # checkpoint = torch.load(os.path.join(
    #     config.out_dir, config.filename), weights_only=False)
    checkpoint = torch.load(os.path.join(
        PATH_PREFIX, "full_run_random_layers_4_heads_4.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)

    for block in model.transformer.h:
        if not hasattr(block.attn, 'bias'):
            block_size = model.config.block_size
            block.attn.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size))
                    .view(1, 1, block_size, block_size)
            )

    torch.save({
        'model': model.state_dict(),
        'config': model.config,
    }, modified_path)
