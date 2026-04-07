import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, emb_size: int, dropout=0.1):
        super().__init__()

        self.emb_size = emb_size

        self.first_linear = nn.Linear(self.emb_size, self.emb_size * 4)
        self.relu = nn.ReLU(inplace=True)
        self.second_linear = nn.Linear(self.emb_size * 4, self.emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        first_iter = self.first_linear(x)
        func_act = self.relu(first_iter)
        second_iter = self.second_linear(func_act)

        return self.dropout(second_iter)
