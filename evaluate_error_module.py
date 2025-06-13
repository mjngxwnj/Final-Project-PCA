import numpy as np

def mean_squared_error_manual(X, X_reconstructed):
    """
    Tinsh mean square error của original data và reconstructed data 
    
    """
    total_error = 0
    n_samples = len(X)
    n_features = len(X[0])

    for i in range(n_samples):
        for j in range(n_features):
            diff = X[i][j] - X_reconstructed[i][j]
            total_error += diff ** 2

    mse = total_error / (n_samples * n_features)
    return mse

def explained_variance(original, reconstructed):
    """
    Tính explained variance (%) theo công thức toán học cơ bản, không dùng np.var.
    
    Parameters:
        original: np.ndarray, shape (n_samples, n_features)
        reconstructed: np.ndarray, shape (n_samples, n_features)
    Returns:
        explained_variance_percent: float
    """
    assert original.shape == reconstructed.shape, "Shape mismatch"
    
    n_samples = original.shape[0]
    
    # Mean của original
    mean_original = np.mean(original, axis=0)
    
    # Tính tổng phương sai thật (từng feature): Var = (1/n) * sum((x - mean)^2)
    total_variance = 0.0
    for i in range(original.shape[1]):  # theo từng feature
        var_i = np.sum((original[:, i] - mean_original[i])**2) / n_samples
        total_variance += var_i
    
    # Sai số khôi phục
    error = original - reconstructed
    mean_error = np.mean(error, axis=0)

    # residual variance
    residual_variance = 0.0
    for i in range(error.shape[1]):
        var_i = np.sum((error[:, i] - mean_error[i])**2) / n_samples
        residual_variance += var_i

    # Explained variance
    explained = total_variance - residual_variance
    explained_percent = (explained / total_variance) * 100

    return float(explained_percent.real)
