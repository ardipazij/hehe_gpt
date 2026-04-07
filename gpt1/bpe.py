from collections import Counter
from typing import List


class BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.uniq_tokens = set()
        self.tokens = []
        self.id2token = {}
        self.token2if = {}

    def fit(self, text: str):
        # 1.уникальные токены
        self.uniq_tokens = sorted(set(text))
        # 2.обучение
        # 2.1 разбить весь текст на отдельные токены
        self.tokens = list(text)

        while len(self.uniq_tokens) < self.vocab_size:
            pairs = Counter()
            for i in range(len(self.tokens) - 1):
                pair = self.tokens[i] + self.tokens[i + 1]
                pairs[pair] += 1
            best_pair, _ = pairs.most_common(1)[0]
            new_token = "".join(best_pair)
            if new_token not in self.uniq_tokens:
                self.uniq_tokens.append(new_token)
            i = 0
            new_tokens = []
            while i < len(self.tokens):
                if (
                    i < len(self.tokens) - 1
                    and ("".join(self.tokens[i] + self.tokens[i + 1])) == best_pair
                ):
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(self.tokens[i])
                    i += 1

            # Добавляем оставшиеся хвосты, если окно вышло за пределы
            while i < len(self.tokens):
                new_tokens.append(self.tokens[i])
                i += 1
            self.tokens = new_tokens
        self.token2id = {token: i for i, token in enumerate(self.uniq_tokens)}
        self.id2token = {i: token for i, token in enumerate(self.uniq_tokens)}
        total_tokens_count = len(self.tokens)

        # 2. Формируем строку из уникальных токенов через "_"
        vocab_string = "_" + "_".join(self.uniq_tokens)

        return (total_tokens_count, vocab_string)

    def encode(self, text):
        encoded_ids = []
        self.tokens = list(text)
        n = len(self.tokens)
        i = 0
        while i < n:
            matched = False
            maybe_token = sorted(
                [s for s in list(self.token2id.keys()) if s.startswith(self.tokens[i])],
                key=len,
                reverse=True,
            )
            for current_token in maybe_token:
                if i + len(current_token) <= len(self.tokens):
                    text_slice = "".join(self.tokens[i : i + len(current_token)])
                    if text_slice == current_token:
                        encoded_ids.append(self.token2id[current_token])
                        i += len(current_token)
                        matched = True
                        break
            if not matched:
                i += 1
        return encoded_ids

    def decode(self, token_ids: List[int]):
        decoded_list = []
        for current_ids in token_ids:
            decoded_list.append(self.id2token[current_ids])
        return "".join(decoded_list)
