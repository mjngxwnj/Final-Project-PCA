import nltk
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np

class TfidfVectorizer:
    """
    A class to represent a TF-IDF vectorizer.
    """

    def __init__(self, text: str):
        """
        Initialize the TfidfVectorizer with text data.

        Params:
            text (str): The text to be vectorized.
        """

        # Cut text into sentences
        self._sentences = sent_tokenize(text)


    def preprocess_text(self) -> str:
        """
        Preprocess the text:
            - Convert to lowercase.
            - Remove punctuation.

        Returns:
            str: The preprocessed text.
        """

        sentences = []
        word_set = []
        
        for sent in self._sentences:
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
    

    
    def term_freq(self, words: list, word: str):
        """
        Calculate TF (Term Frequency) for the preprocessed text.

        Returns:
            dict: A dictionary with words as keys and their term frequencies as values.
        """
        N = len(words)
        occurance = len([token for token in words if token == word])
        return occurance/N
    
    
    def inverse_doc_freq(self, word: str, word_count: dict, total_documents: int):
        """
        Calculate IDF (Inverse Document Frequency) for the given word.

        Returns:
            float: The IDF value for the word.
        """
        word_occurrence = word_count.get(word) 

        idf = np.log(total_documents / word_occurrence)

        return idf
    

    def tf_idf(self, sentence: list, word_set: set, word_count: dict, 
                     word_dict: dict, total_documents: int) -> np.array:
        
        tf_idf_vec = np.zeros((len(word_set),))

        for word in sentence:
            tf = self.term_freq(sentence, word)
            idf = self.inverse_doc_freq(word, word_count, total_documents)
            
            value = tf*idf
            tf_idf_vec[word_dict[word]] = value 

        return tf_idf_vec
    

    def implement(self) -> np.array:
        """
        Implement the TF-IDF vectorization process.

        Returns:
            np.array: The TF-IDF vector for the text.
        """

        vectors = []
        sentences, word_set, word_count, word_dict, total_documents = self.preprocess_text()
        for sentence in sentences:
            tf_idf_vec = self.tf_idf(sentence, word_set, word_count, word_dict, total_documents)
            vectors.append(tf_idf_vec)
        
        return np.array(vectors)