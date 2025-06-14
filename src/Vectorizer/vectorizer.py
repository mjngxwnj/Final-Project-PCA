import numpy as np
import pandas as pd
import cv2
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
        
        # if image_matrix.shape != (64, 64):
        #     raise ValueError("image_matrix must be of shape (64, 64).")

        # Convert image to grayscale matrix
        gray_image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
    
        row, col = gray_image_matrix.shape

        flat_array = np.zeros((row * col,))

        idx = 0

        for i in range(row):
            for j in range(col):
                flat_array[idx] = gray_image_matrix[i][j]
                idx += 1
            
        return flat_array      # shape (row * col,)
    

    def _table_vectorizer(self, table_data: pd.DataFrame, length_threshold: int = 50) -> np.ndarray:
        """
        Normalize and vectorize table data into a feature vector.

        Returns:
            np.ndarray: An array representing the feature vector of the table data.
        """

        # Check params
        if not isinstance(table_data, pd.DataFrame):
            raise TypeError("table_data must be a pandas DataFrame.")
        
        # Normalize numeric columns
        for numeric_col in table_data.select_dtypes(include = ['number']).columns:
            mean = table_data[numeric_col].mean()
            std  = table_data[numeric_col].std()
            table_data[numeric_col] = (table_data[numeric_col] - mean) / std

        # Columns to drop (comment, long text, ...)
        cols_to_drop = []

        # Label Encoding for text columns
        for text_col in table_data.select_dtypes(include = ['object']).columns:
            if isinstance(table_data[text_col], pd.DataFrame):
                print(table_data)
            avg_length = table_data[text_col].fillna('').apply(lambda x: len(str(x))).mean()
            unique_values = list(table_data[text_col].unique())

            # Check:
            # 1. If the average length of the text is below the threshold
            # (Avoid comment columns, long text columns, ...)
            # 2. If the number of unique values is less than half of the total number of rows
            # (Avoid columns with too many unique values, like names, ...)
            if avg_length < length_threshold and len(unique_values) < len(table_data) / 2: 
                table_data[text_col] = table_data[text_col].\
                                       apply(lambda x: unique_values.index(x)).fillna(-1).astype('int')
            
            # If the average length is above the threshold or the number of unique values is too high, remove the column
            else:
                cols_to_drop.append(text_col)
        
        # Drop long text columns
        normalized_data = table_data.drop(columns = cols_to_drop)

        # Convert to numpy array
        normalized_data = normalized_data.to_numpy()

        return normalized_data
    
    
    def vectorize(self, list_data: list) -> np.ndarray:
        """
        Vectorize the data based on its type.
        
        Returns:
            np.ndarray: An array representing the feature vector of the data.
        """

        # Check params
        if not isinstance(list_data, list):
            raise TypeError("list_data must be a list.")
  
        # create vector to store all vectorized data
        vectorized_vector = []
            
        for data in list_data:    
            if not all(key in data for key in ['type', 'content', 'meta']):
                raise ValueError("list_data must contain 'type', 'content', and 'meta' keys.")

            if data['type'] not in ['text', 'image', 'table']:
                raise ValueError("data_type must be one of 'text', 'image', 'table'.")
       
            # If condition to vectorize appropriate data type
            if data['type'] == 'text':
                vectorized_vector.append(self._text_vectorizer(data['content']))
            
            elif data['type'] == 'image':
                vectorized_vector.append(self._image_vectorizer(data['content']))

            else: 
                vectorized_vector.append(self._table_vectorizer(data['content']))

        
        return vectorized_vector








