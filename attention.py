from torch import nn

class HeadAttention(nn.Module):
    def __init__(self, emb_size : int, head_size : int, max_seq_len : int):
        super().__init__()
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.W_k = nn.Linear(emb_size, head_size)
        self.W_q = nn.Linear(emb_size, head_size)
        self.W_v = nn.Linear(emb_size, head_size)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        
        self.register_buffer('mask', mask)
        
    def forward(self, x : torch.Tensor):
        k = self.W_k(x)
        q = self.W_q(x)
        v = self.W_v(x)
        
        score = (q @ k.transpose(-2, -1)) / self.head_size ** 0.5
        
        score = score.masked_fill(self.mask[:x.size(1), :x.size(1)] == 0, float('-inf'))
        w = torch.softmax(score, dim=-1)
        
        return w @ v





class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads : int, emb_size : int, head_size : int, max_seq_len : int, dropout = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        
        self.heads = nn.ModuleList([
            HeadAttention(self.emb_size, self.head_size, self.max_seq_len)
            for _ in range(self.num_heads)
        ])
        
        self.W_o = nn.Linear(self.head_size * self.num_heads, self.emb_size)
        self.dropout = nn.Dropout(dropout)
      
    def forward(self, x : torch.Tensor):
        head_result = torch.cat([current_head.forward(x) for current_head in self.heads],dim=-1)
        head_lin_result = self.W_o(head_result)
        return self.dropout(head_lin_result)
        
        
            
        
        




