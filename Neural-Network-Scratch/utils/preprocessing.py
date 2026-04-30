import numpy as np

def standardize(X):
    """
    Standardize the input features.
    Formula: (X - mean) / std
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    # Add a small epsilon to avoid division by zero
    X_std = (X - mean) / (std + 1e-8)
    return X_std, mean, std

def flatten_images(X):
    """
    Reshapes a (m, 28, 28) image array into a (784, m) array.
    """
    m = X.shape[0]
    return X.reshape(m, -1).T