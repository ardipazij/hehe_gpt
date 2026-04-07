import torch
from torch import nn


import torch
from torch import nn


class HeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int):
        super().__init__()
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.W_k = nn.Linear(emb_size, head_size)
        self.W_q = nn.Linear(emb_size, head_size)
        self.W_v = nn.Linear(emb_size, head_size)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))

        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, use_cache=True, cache=None):
        k = self.W_k(x)
        q = self.W_q(x)
        v = self.W_v(x)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=1)
            v = torch.cat([cache[1], v], dim=1)
        score = (q @ k.transpose(-2, -1)) / self.head_size**0.5
        if cache is None:
            score = score.masked_fill(
                self.mask[: x.size(1), : x.size(1)] == 0, float("-inf")
            )
        w = torch.softmax(score, dim=-1)
        return (w @ v, (k, v)) if use_cache is True else (w @ v, None)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        dropout=0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len

        self.heads = nn.ModuleList(
            [
                HeadAttention(self.emb_size, self.head_size, self.max_seq_len)
                for _ in range(self.num_heads)
            ]
        )

        self.W_o = nn.Linear(self.head_size * self.num_heads, self.emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, use_cache=True, cache=None):
        current_caches = cache if cache is not None else [None] * self.num_heads

        all_head_outs = []
        new_cache_list = []

        for i, head in enumerate(self.heads):
            head_out, head_new_cache = head(
                x, use_cache=use_cache, cache=current_caches[i]
            )

            all_head_outs.append(head_out)
            new_cache_list.append(head_new_cache)

        combined_heads = torch.cat(all_head_outs, dim=-1)
        x_out = self.W_o(combined_heads)
        x_out = self.dropout(x_out)
        return (x_out, new_cache_list) if use_cache else (x_out, None)
