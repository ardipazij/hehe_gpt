import torch
import math
from torch import nn

from positional embeddings import PositionalEmbeddings
from ffn import FeedForward
from decoder import Decoder

class GetData(Dataset):
    def __init__(self, data: list, seq_len: int, device: str):
        self.data = torch.tensor(data, dtype=torch.long, device=device)
        self.seq_len = seq_len
        self.device = device

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + 1 + self.seq_len]

        return x, y


class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)

    def forward(self, x: torch.Tensor):
        return self.embedding(x)

class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        emb_size: int,
        num_heads: int,
        head_size: int,
        num_layers: int,
        dropout=0.1,
        device="cpu",
    ):
        super().__init__()
        self.te = TokenEmbeddings(vocab_size, emb_size)
        self.pe = PositionalEmbeddings(max_seq_len, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers
        self.drop = dropout
        self.device = device
        self.the_last_layer = nn.LayerNorm(self.emb_size)

        self.decoders = nn.ModuleList(
            [
                Decoder(num_heads, emb_size, head_size, max_seq_len, dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.linear = nn.Linear(emb_size, vocab_size)

    def forward(self, x, use_cache=True, cache=None):
        if cache is None:
            res_te = self.te(x)
            res_pe = self.pe(x.size(1)) 
        else:
            start_pos = cache[0][0][0].size(1) 
            res_te = self.te(x)
            res_pe = self.pe(1, start_pos=start_pos)

        x_embs = self.dropout(res_te + res_pe)

        new_cache = [] if use_cache else None
        
        for i, decoder_layer in enumerate(self.decoders):
            layer_cache = cache[i] if cache is not None else None
            x_embs, layer_new_cache = decoder_layer(x_embs, use_cache=use_cache, cache=layer_cache)
            
            if use_cache:
                new_cache.append(layer_new_cache)

        x_embs = self.the_last_layer(x_embs)
        logits = self.linear(x_embs)

        return (logits, new_cache) if use_cache else (logits, None)

    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        use_cache=True,
        top_k=None,
        top_p=None,
        temperature=1.0,
    ):
        current_cache = None
        
        for i in range(max_new_tokens):
            if use_cache and current_cache is not None:
                x_input = x[:, -1:]
            else:
                x_input = x

            logits, current_cache = self.forward(x_input, use_cache=use_cache, cache=current_cache)
            
            logits = logits[:, -1, :] / temperature

            if do_sample:
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                if top_p is not None:
                    probs = torch.softmax(logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    keep_sorted = cumulative_probs < top_p
                    keep_sorted[:, 0] = 1 

                    top_p_mask = torch.zeros_like(keep_sorted, dtype=torch.bool)
                    top_p_mask.scatter_(dim=-1, index=sorted_indices, src=keep_sorted)
                    logits[~top_p_mask] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            x = torch.cat((x, next_token), dim=1)

        return x

    def fit(self, train_loader, valid_loader, num_epoch: int, learning_rate: float):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epoch):
            self.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                logits = self.forward(inputs)

                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)

                loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
                self.train_loss = loss

                optimizer.zero_grad()
                self.train_loss.backward()
                optimizer.step()

            self.eval()
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    logits = self.forward(inputs)
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = targets.view(-1)

                    v_loss = torch.nn.functional.cross_entropy(
                        logits_flat, targets_flat
                    )
                    self.valid_loss = v_loss






