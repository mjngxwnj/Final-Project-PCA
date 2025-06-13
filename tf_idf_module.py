import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import Tuple
import numpy as np

class TfidfVectorizer:
    """
    A simple implementation of TF-IDF vectorizer.
    """

    def __init__(self):
        self.word_set = set()
        self.word_count = {}
        self.word_dict = {}
        self.total_documents = 0
        self.is_fitted = False

    def _split_sentences(self, text: str) -> list:
        """
        Split the text into sentences.
        """
        return sent_tokenize(text)

    def _preprocess_text(self, sentences_list: list) -> Tuple[list, set, dict, dict, int]:
        """
        Preprocess the text: tokenize, lowercase, remove punctuation.
        """
        sentences = []
        word_set = []

        for sent in sentences_list:
            words = [word.lower() for word in word_tokenize(sent) if word.isalpha()]
            sentences.append(words)
            for word in words:
                if word not in word_set:
                    word_set.append(word)

        word_set = set(word_set)
        total_documents = len(sentences)
        word_dict = {word: i for i, word in enumerate(word_set)}

        word_count = {}
        for word in word_set:
            word_count[word] = sum(1 for sent in sentences if word in sent)

        return sentences, word_set, word_count, word_dict, total_documents

    def _term_freq(self, words: list, word: str) -> float:
        """
        Calculate TF (Term Frequency) of a word in a sentence.
        """
        N = len(words)
        if N == 0:
            return 0.0
        return words.count(word) / N

    def _inverse_doc_freq(self, word: str) -> float:
        """
        Calculate IDF (Inverse Document Frequency) of a word.
        """
        word_occurrence = self.word_count.get(word, 1)
        return np.log(self.total_documents / word_occurrence)

    def _tf_idf(self, sentence: list) -> np.ndarray:
        """
        Calculate TF-IDF vector for a sentence.
        """
        tf_idf_vec = np.zeros((len(self.word_set),))

        for word in sentence:
            if word in self.word_dict:
                tf = self._term_freq(sentence, word)
                idf = self._inverse_doc_freq(word)
                tf_idf_vec[self.word_dict[word]] = tf * idf

        return tf_idf_vec

    def fit(self, texts: list[str]) -> None:
        """
        Fit the vectorizer on a list of documents.
        """
        all_sentences = []
        for text in texts:
            all_sentences.extend(self._split_sentences(text))

        sentences, word_set, word_count, word_dict, total_documents = self._preprocess_text(all_sentences)

        self.word_set = word_set
        self.word_count = word_count
        self.word_dict = word_dict
        self.total_documents = total_documents
        self.is_fitted = True

    def transform(self, text: str) -> np.ndarray:
        """
        Transform a single document into a TF-IDF vector matrix (one row per sentence).
        """
        if not self.is_fitted:
            raise RuntimeError("TfidfVectorizer must be fitted before calling transform().")

        sentences_list = self._split_sentences(text)
        vectors = []

        for sent in sentences_list:
            words = [word.lower() for word in word_tokenize(sent) if word.isalpha()]
            tf_idf_vec = self._tf_idf(words)
            vectors.append(tf_idf_vec)

        return np.array(vectors)
