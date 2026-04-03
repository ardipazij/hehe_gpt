from torch import nn
from typing import List
class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size : int, emb_size : int):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        
    def forward(self, x : torch.Tensor):
        return self.embedding(x)

class PositionalEmbeddings(nn.Module):
    def __init__(self, vocab_size : int, emb_size : int):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        
    def forward(self, seq_len : int):
         return self.embedding(torch.Tensor([i for i in range(seq_len)]).long())
        
class FeedForward(nn.Module):
    def __init__(self, emb_size : int, dropout = 0.1):
        super().__init__()
        
        self.emb_size = emb_size
        
        self.first_linear = nn.Linear(self.emb_size, self.emb_size * 4)
        self.relu = nn.ReLU(inplace=True)
        self.second_linear = nn.Linear(self.emb_size * 4, self.emb_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x : torch.Tensor):
        first_iter = self.first_linear(x)
        func_act = self.relu(first_iter)
        second_iter = self.second_linear(func_act)
        
        return self.dropout(second_iter)
class HeadAttention(nn.Module):
    def __init__(self, emb_size : int, head_size : int, max_seq_len : int):
        super().__init__()
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.W_k = nn.Linear(emb_size, head_size)
        self.W_q = nn.Linear(emb_size, head_size)
        self.W_v = nn.Linear(emb_size, head_size)
        mask = torch.tril(torch.ones(int(max_seq_len), int(max_seq_len)))
        
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
    
class Decoder(nn.Module):
    '''
    1.На вход блока декодера поступает тензор размером batch_size x seq_len x emb_size.
    2.Пропускаем его через многоголовый механизм внимания. На выходе из него получаем тензор такого же размера.
    3.Складываем его с тензором поступившим на вход механизма Внимания (они одинакового размера).
    4.Пропускаем получившийся тензор через первый слой нормализации.
    5.После чего подаем его в блок полносвязной сети. На выходе получаем тензор такого же размера.
    6.Складываем его с тензором, который поступил на вход в FNN.
    7.Затем пропускается получившийся тензор через второй слой нормализации.
    8.Наш итоговый тензор будет иметь вид: batch_size x seq_len x emb_size (точно такой же как поступил на вход блока Декодера).
    '''
    def __init__(self, num_heads : int, emb_size : int, head_size : int, max_seq_len : int, dropout = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len, dropout)
        self.ffn = FeedForward(emb_size, dropout)
        self.first_norm = nn.LayerNorm(emb_size)
        self.second_norm = nn.LayerNorm(emb_size)
        
    def forward(self, x : torch.Tensor):
        first_iter = self.mha.forward(x)
        result_first_iter = first_iter + x
        
        first_norm = self.first_norm(result_first_iter)
        
        second_iter = self.ffn(first_norm)
        result_second_iter = first_norm + second_iter
        
        return self.second_norm(result_second_iter)
        





class GPT(nn.Module):
    def __init__(self, vocab_size : int, max_seq_len : int, emb_size : int, num_heads : int, head_size : int,  num_layers : int, dropout = 0.1, device = 'cpu'):
        '''
        Экземпляр класса TokenEmbeddings, передав ему параметры vocab_size и emb_size.
Экземпляр класса PositionalEmbeddings, передав ему max_seq_len и emb_size.
Слой nn.Dropout с вероятностью dropout.
Создайте блоки Декодера (из класса Decoder) в количестве num_layers.
Передайте им всем одни и те же параметры: num_heads, emb_size, head_size и max_seq_len.
Создайте линейный слой (nn.Linear) размером emb_size х vocab_size
'''
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
        
        self.decoders = nn.ModuleList([
            Decoder(num_heads, emb_size, head_size, max_seq_len, dropout)
            for _ in range(self.num_layers)
        ])
        self.linear = nn.Linear(emb_size, vocab_size)
        
    def forward(self, x):
            res_te = self.te.forward(x)
            res_pe = self.pe.forward(x.size(1))
            
            input_embs = res_te + res_pe
            
            res_drop = self.dropout(input_embs)
            
            for decoder_layer in self.decoders:
                res_drop = decoder_layer.forward(res_drop)
            
            logits = self.linear(res_drop)
            
            return logits
    def generate(self, x : torch.Tensor, max_new_tokens : int):
        '''
        Начните цикл до max_new_tokens:
Обрежьте поступившую последовательность до максимальной возможной длины. Для этого возьмите последние max_seq_len токенов в последовательности.
Передайте последовательность в метод forward класса GPT и получите логиты.
Возьмите последний вектор логитов и прогоните его через функцию Softmax.
Токен с максимальной вероятностью добавьте в конец последовательности.
Верните единую последовательность: все поступившие токены + все сгенерированные токены. Размер выходной последовательности должен быть batch_size x seq_len + max_new_tokens.
'''
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.max_seq_len:]
            logits = self.forward(x_cond)
            
            #Берем логиты только для последнего временного шага 
            last_logits = logits[:, -1, :]
            
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True) # (batch_size, 1)
            
            x = torch.cat((x, next_token), dim=1)
            
        return x
    def save(self, path):
      torch.save({
          'model_state_dict': self.state_dict(),
          'vocab_size': self.vocab_size,
          'max_seq_len': self.max_seq_len,
          'emb_size': self.emb_size,
          'num_heads': self.num_heads,
          'head_size': self.head_size,
          'num_layers': self.num_layers
      }, path)
      
  @classmethod
  def load(cls, path, device):
      checkpoint = torch.load(path, map_location=device)
      model = cls(
          vocab_size=checkpoint['vocab_size'],
          max_seq_len=checkpoint['max_seq_len'],
          emb_size=checkpoint['emb_size'],
          num_heads=checkpoint['num_heads'],
          head_size=checkpoint['head_size'],
          num_layers=checkpoint['num_layers']
      )
      model.load_state_dict(checkpoint['model_state_dict'])
      model.to(device)
      return model
        




