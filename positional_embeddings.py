import torch
from torch import nn


class PositionalEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)

    def forward(self, seq_len: int):
        return self.embedding(torch.Tensor([i for i in range(seq_len)]).long())
