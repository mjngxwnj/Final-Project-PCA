from vectorize_tfidf_manual import ManualTFIDF
import numpy as np

class VectorizeChunkedTFIDF:
    """
    
    """
    def __init__(self, chunk_size=50, max_features=1000, pooling="mean"):
        self.chunk_size = chunk_size
        self.max_features = max_features
        self.pooling = pooling
        self.tfidf = ManualTFIDF(max_features=max_features)
        self.vocab_fitted = False

    def chunk_text(self, text):
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunks.append(" ".join(words[i:i + self.chunk_size]))
        return chunks

    def fit(self, documents):
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_text(doc))
        self.tfidf.fit(all_chunks)
        self.vocab_fitted = True
        return self

    def transform(self, text):
        if not self.vocab_fitted:
            raise ValueError("Cần gọi fit() trước")
        chunks = self.chunk_text(text)
        tfidf_matrix = self.tfidf.transform(chunks)  # shape (num_chunks, vocab_size)

        if self.pooling == "mean":
            return np.mean(tfidf_matrix, axis=0, keepdims=True)
        elif self.pooling == "max":
            return np.max(tfidf_matrix, axis=0, keepdims=True)
        else:
            return tfidf_matrix
