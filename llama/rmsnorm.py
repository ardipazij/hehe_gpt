import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim : int, eps = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x : torch.Tensor):
        rms = torch.sqrt(
          torch.mean(
            x**2, dim=-1, keepdim=True
          )
          + self.eps)
        return self.weight * (x / rms)
