from torch import nn
class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size : int, emb_size : int):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        
    def forward(self, x : torch.Tensor):
        return self.embedding(x)
