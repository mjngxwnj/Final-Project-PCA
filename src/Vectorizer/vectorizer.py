import numpy as np
from .tf_idf_module import TfidfVectorizer


class FeatureVectorizer:
    """
    A class to vectorize text, arrays, dataframes to a feature vector.
    """

    def __init__(self, data_info: dict):
        """
        Initialize the FeatureVectorizer with data information.

        Params:
            data_info (dict): A dict containing information about the data (type, content, metadata).
        """

        # Check params
        if not isinstance(data_info, dict):
            raise TypeError("data_info must be a dict.")
        
        if not all(key in data_info for key in ['type', 'content', 'metadata']):
            raise ValueError("data_info must contain 'type', 'content', and 'metadata' keys.")
        
        # add attributes
        self._data_info = data_info
        self._data_type = self._data_info['type']
        self._content   = self._data_info['content']


    def text_vectorizer(self) -> np.array:
        """
        Vectorize text data into a feature vector.

        Returns:
            np.array: An array representing the feature vector of the text data.
        """
        return TfidfVectorizer(self._content).implement()
