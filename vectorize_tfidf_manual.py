import math
import numpy as np
from collections import defaultdict, Counter

class ManualTFIDF:
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.vocab = {}
        self.idf = {}
        self.fitted = False

    def _tokenize(self, text):
        return text.lower().split()

    def _build_vocab(self, documents):
        df = defaultdict(int)
        vocab_counter = Counter()

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] += 1
            vocab_counter.update(self._tokenize(doc))

        # Sort theo tần suất để chọn top N từ
        vocab_list = sorted(vocab_counter.items(), key=lambda x: -x[1])
        if self.max_features:
            vocab_list = vocab_list[:self.max_features]

        self.vocab = {word: idx for idx, (word, _) in enumerate(vocab_list)}

        N = len(documents)
        self.idf = {
            word: math.log(N / (1 + df[word])) for word in self.vocab
        }

    def fit(self, documents):
        self._build_vocab(documents)
        self.fitted = True

    def transform(self, documents):
        """
        Trả về ma trận (num_docs, vocab_size)
        """
        if not self.fitted:
            raise ValueError("Chưa gọi .fit()")

        matrix = np.zeros((len(documents), len(self.vocab)))

        for i, doc in enumerate(documents):
            tf = Counter(self._tokenize(doc))
            total_terms = sum(tf.values())

            for word, count in tf.items():
                if word in self.vocab:
                    tf_val = count / total_terms
                    idf_val = self.idf[word]
                    tfidf = tf_val * idf_val
                    matrix[i, self.vocab[word]] = tfidf

        return matrix

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)
