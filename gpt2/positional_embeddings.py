import torch
from torch import nn


class PositionalEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)

    def forward(self, seq_len: int, start_pos=0):
        return self.embedding(
            torch.Tensor([i for i in range(start_pos, start_pos + seq_len)]).long()
        )
