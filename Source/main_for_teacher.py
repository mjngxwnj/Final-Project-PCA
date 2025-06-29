# main.py
from Reader.reader_agent import DataExpander
from Vectorizer.vectorizer_module import FeatureVectorizer
from Evaluate.evaluate_error_module import mean_squared_error_manual, explained_variance
from PCA.pca_module import PCA
import numpy as np
import pandas as pd
import os
import time

import numpy as np
import pandas as pd

def main():
    # 1. read data
    path = "./Source/Test_Data/csv/file1.csv" # Adjust the path as needed
    data = DataExpander().expand(path)

    # 2. vectorize data
    fv = FeatureVectorizer()
    vectorized_data = fv.vectorize([data[0]])
    print(vectorized_data[0].shape)
    print(vectorized_data)

    # 3: apply PCA for 1 file
    import time 
    start_time = time.time()
    my_pca = PCA().fit(vectorized_data[0])
    X_reduced = my_pca.transform(vectorized_data[0])
    X_reconstructed = my_pca.inverse_transform(X_reduced)
    end_time = time.time()
    print(f"Time taken for PCA: {end_time - start_time} seconds")
    # 4. evaluate error
    error = mean_squared_error_manual(vectorized_data[0], X_reconstructed)
    print(f" MSE: {error}") 
    explain_var = explained_variance(vectorized_data[0], X_reconstructed)
    print(f" Explained Variance: {explain_var}")
    
if __name__ == "__main__":
    main()
