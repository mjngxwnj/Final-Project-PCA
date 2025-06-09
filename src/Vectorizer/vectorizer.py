import numpy as np
import pandas as pd
from .tf_idf_module import TfidfVectorizer


class FeatureVectorizer:
    """
    A class to vectorize text, arrays, dataframes to a feature vector.
    """

    def __init__(self):
        pass


    def _text_vectorizer(self, text: str) -> np.ndarray:
        """
        Vectorize text data into a feature vector.

        Returns:
            np.ndarray: An array representing the feature vector of the text data.
        """
        return TfidfVectorizer().transform(text)


    def _image_vectorizer(self, image_matrix: np.ndarray) -> np.ndarray:
        """
        Vectorize image data into a feature vector.
        
        Returns:
            np.ndarray: An array representing the feature vector of the image data.
        """

        # Check params
        if not isinstance(image_matrix, np.ndarray):
            raise TypeError("image_matrix must be a numpy array.")
        
        if image_matrix.shape != (64, 64):
            raise ValueError("image_matrix must be of shape (64, 64).")
        
        flat_array = np.zeros((4096,))

        idx = 0
        
        for i in range(64):
            for j in range(64):
                flat_array[idx] = image_matrix[i][j]
                idx += 1
            
        return flat_array      # shape (4096,)
    

    def _table_vectorizer(self, table_data: pd.DataFrame) -> np.ndarray:
        """
        Normalize and vectorize table data into a feature vector.

        Returns:
            np.ndarray: An array representing the feature vector of the table data.
        """

        # Check params
        if not isinstance(table_data, pd.DataFrame):
            raise TypeError("table_data must be a pandas DataFrame.")
        
        array_data = table_data.to_numpy()

        # Normalize the data using min-max scaling
        min_values = np.min(array_data, axis = 0)
        max_values = np.max(array_data, axis = 0)

        normalized_data = (array_data - min_values) / (max_values - min_values)

        return normalized_data
    
    
    def vectorize(self, data_info: dict) -> np.ndarray:
        """
        Vectorize the data based on its type.
        
        Returns:
            np.ndarray: An array representing the feature vector of the data.
        """

        # Check params
        if not isinstance(data_info, dict):
            raise TypeError("data_info must be a dict.")
        
        if not all(key in data_info for key in ['type', 'content', 'meta']):
            raise ValueError("data_info must contain 'type', 'content', and 'meta' keys.")

        # Check params
        if data_info['type'] not in ['text', 'image', 'table']:
            raise ValueError("data_type must be one of 'text', 'image', 'table'.")
        
        data_type = data_info['type']
        content   = data_info['content']
        metadata  = data_info['meta']

        if data_type == 'text':
            return self._text_vectorizer(content)
        
        elif data_type == 'image':
            return self._image_vectorizer(content)
        
        else: return self._table_vectorizer(content)