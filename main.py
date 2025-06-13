# main.py
from reader_module import DataExpander
from vector_module import FeatureVectorizer
from pca_module import PCA
from evaluate_error_module import mean_squared_error_manual
from evaluate_error_module import explained_variance

import numpy as np

def main():
    path = ""
    
    # 1. read data
    data = DataExpander().expand(path)

    # 2. Vectorize

    fv = FeatureVectorizer()
    fv.fit_texts(data['content'])    
    vectorized_data = fv.vectorize(data)

    # 3.1 apply PCA for many files
    # vectors = []
    # for file in list_of_paths:
    #     data_info = DataExpander().expand(file)
    #     vector = fv.vectorize(data_info)
    #     vectors.append(vector)

    # X = np.vstack(vectors)  # shape: (num_samples, num_features)
    # my_pca = PCA().fit(X)

    # 3.2: apply PCA for 1 file
    my_pca = PCA().fit(vectorized_data)
    X_reduced = my_pca.transform(vectorized_data)
    X_reconstructed = my_pca.inverse_transform(X_reduced)

    # 4. evaluate error
    error = mean_squared_error_manual(vectorized_data, X_reconstructed)
    print(error) 
    explain_var = explained_variance(vectorized_data, X_reconstructed)
    print(explain_var)
    # test with sample test
    
if __name__ == "__main__":
    main()
