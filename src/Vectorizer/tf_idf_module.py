import nltk
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from typing import Tuple
import numpy as np

class TfidfVectorizer:
    """
    A class to represent a TF-IDF vectorizer.
    """

    def __init__(self):
        pass


    def _split_sentences(self, text: str) -> None:
        """
        Split the text into sentences.

        Args:
            text (str): The input text to be split into sentences.
        
        Returns:
            list: A list of sentences.
        """

        # Cut text into sentences
        return sent_tokenize(text)


    def _preprocess_text(self, sentences_list: list) -> Tuple[list, set, dict, dict, int]:
        """
        Preprocess the text:
            - Convert to lowercase.
            - Remove punctuation.

        Returns:
            sentences (list): List of tokenized sentences.
            word_set (set): Set of unique words.
            word_count (dict): Dictionary with words as keys and their document frequency as values.
            word_dict (dict): Dictinary with words as keys and their indices as values.
            total_documents (int): Number of sentences.
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
        
        word_dict = {}
        word_dict = {word: i for i, word in enumerate(word_set)}


        word_count = {}
        for word in word_set:
            word_count[word] = 0
            for sent in sentences:
                if word in sent:
                    word_count[word] += 1
                    
        return sentences, word_set, word_count, word_dict, total_documents
    

    
    def _term_freq(self, words: list, word: str) -> float:
        """
        Calculate TF (Term Frequency) for the preprocessed text.

        Args:
            words (list): List of words in the sentence.s
            word (str): The word to calculate TF.

        Returns:
            float: the TF value.
        """

        N = len(words)
        occurance = len([token for token in words if token == word])

        return occurance / N
    
    
    def _inverse_doc_freq(self, word: str, word_count: dict, total_documents: int) -> float:
        """
        Calculate IDF (Inverse Document Frequency) for the given word.

        Args:
            word (str): The word to calculate IFD.
            word_count (dict): Dictionary with words as keys and their doc frequency as values.
            total_documents (int): Total number of sentences.

        Returns:
            float: The IDF value for the word.
        """
        word_occurrence = word_count.get(word) 

        idf = np.log(total_documents / word_occurrence)

        return idf
    

    def _tf_idf(self, sentence: list, word_set: set, word_count: dict, 
                     word_dict: dict, total_documents: int) -> np.ndarray:
        """
        Calculate the TF-IDF vector.

        Args:
            sentence (list): List of words in the sentence.
            word_set (set): Set of unique words.
            word_count (dict): Dictionary with words as keys and their document frequency as values.
            word_dict (dict): Dictionary with words as keys and their indices as values.
            total_documents (int): Total number of sentences.
        
        Returns:
            np.ndarray: The TF-IDF vector.
        """
        
        tf_idf_vec = np.zeros((len(word_set),))

        for word in sentence:
            tf = self._term_freq(sentence, word)
            idf = self._inverse_doc_freq(word, word_count, total_documents)
            
            value = tf * idf
            tf_idf_vec[word_dict[word]] = value 

        return tf_idf_vec
    

    def transform(self, text: str) -> np.ndarray:
        """
        Implement the TF-IDF vectorization process.

        Returns:
            np.array: The TF-IDF vector for the text.
        """

        #Split text into sentences
        sentences_list = self._split_sentences(text)

        # Preprocess text and calculate TF-IDF vectors
        vectors = []
        sentences, word_set, word_count, word_dict, total_documents = self._preprocess_text(sentences_list)

        for sentence in sentences:
            tf_idf_vec = self._tf_idf(sentence, word_set, word_count, word_dict, total_documents)
            vectors.append(tf_idf_vec)
        
        return np.array(vectors)
    
