import torch
from torch import nn
from attention import MultiHeadAttention
from ffn import FeedForward

class Decoder(nn.Module):

    def __init__(
        self,
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        dropout=0.1,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(
            num_heads, emb_size, head_size, max_seq_len, dropout
        )
        self.ffn = FeedForward(emb_size, dropout)
        self.first_norm = nn.LayerNorm(emb_size)
        self.second_norm = nn.LayerNorm(emb_size)
        
    def forward(self, x : torch.Tensor, use_cache=True, cache=None):
        first_norm = self.first_norm(x)
        first_iter, cache_result = self.mha(first_norm, use_cache=use_cache, cache=cache)
        result_first_iter = first_iter + x
        
        second_norm = self.second_norm(result_first_iter)
        
        second_iter = self.ffn(second_norm)
        result_second_iter = result_first_iter + second_iter
        
        return (result_second_iter, cache_result) if use_cache else (result_second_iter, cache_result)
