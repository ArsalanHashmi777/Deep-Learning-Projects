import numpy as np

def normalize(X):
    """
    Scales pixel values from [0, 255] to [0, 1].
    """
    return X / 255.0

def one_hot_encode(Y, num_classes=10):
    """
    Converts a label like 3 into [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
    """
    m = Y.shape[0]
    one_hot = np.zeros((num_classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot

def prepare_data(X, Y):
    """
    The final 'wrapper' to get data ready for the model.
    """
    # Transpose X so its shape is (784, m) instead of (m, 784)
    X_flattened = X.T
    
    # Normalize
    X_norm = normalize(X_flattened)
    
    # One-hot encode labels
    Y_encoded = one_hot_encode(Y)
    
    return X_norm, Y_encoded