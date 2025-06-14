import numpy as np
from typing import List

def auto_select_maxlen(text: str, maxlen_candidates: List[int] = [32, 64, 128, 256, 512, 768, 1024], min_fill_ratio: float = 0.8) -> int:
    """
    Chọn maxlen tối ưu để chia văn bản sao cho mỗi sample có ít padding nhất.
    - maxlen_candidates: danh sách các độ dài maxlen để thử.
    - min_fill_ratio: % tối thiểu token thật trên tổng maxlen (vd: 0.8 nghĩa là ít nhất 80% là token thật).
    """
    tokens = text.strip().split()
    total_tokens = len(tokens)

    best_maxlen = maxlen_candidates[0]
    best_score = 0

    for maxlen in maxlen_candidates:
        num_chunks = (total_tokens + maxlen - 1) // maxlen
        padding_total = num_chunks * maxlen - total_tokens
        fill_ratio = 1 - (padding_total / (num_chunks * maxlen))

        if fill_ratio >= min_fill_ratio and fill_ratio > best_score:
            best_score = fill_ratio
            best_maxlen = maxlen

    return best_maxlen


class ManualTokenizer:
    def __init__(self, maxlen: int = 512, scale: str = "minmax"):
        self.maxlen = maxlen
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self._is_vocab_built = False
        self.scale_method = scale

    def tokenize(self, text: str) -> List[str]:
        # Tách token theo khoảng trắng, giữ nguyên dấu câu và ký tự đặc biệt
        return text.strip().split()

    def build_vocab(self, text: str):
        if not isinstance(text, str):
            raise TypeError("text must be a single string.")

        tokens = self.tokenize(text)
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        self._is_vocab_built = True

    def _scale_indices(self, encoded: List[int]) -> np.ndarray:
        arr = np.array(encoded, dtype=np.float32)

        if self.scale_method == "minmax":
            max_val = max(1, max(self.vocab.values()))
            return arr / max_val  # scale về [0, 1]
        elif self.scale_method == "standard":
            mean = np.mean(list(self.vocab.values()))
            std = np.std(list(self.vocab.values()))
            std = std if std != 0 else 1.0
            return (arr - mean) / std  # chuẩn hóa z-score
        else:
            raise ValueError("Unsupported scale method: choose 'minmax' or 'standard'")

    def transform(self, text: str) -> np.ndarray:
        if not self._is_vocab_built:
            raise RuntimeError("You must build the vocab first using build_vocab().")

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        tokens = self.tokenize(text)
        num_chunks = (len(tokens) + self.maxlen - 1) // self.maxlen
        results = []

        for i in range(num_chunks):
            chunk = tokens[i * self.maxlen: (i + 1) * self.maxlen]
            encoded = [self.vocab.get(token, self.vocab["<UNK>"]) for token in chunk]

            # Padding nếu thiếu
            if len(encoded) < self.maxlen:
                encoded += [self.vocab["<PAD>"]] * (self.maxlen - len(encoded))

            scaled = self._scale_indices(encoded)
            results.append(scaled)

        return np.array(results, dtype=np.float32)

    def get_vocab(self) -> dict:
        return self.vocab
