import numpy as np
import pandas as pd

def load_mnist_data(filepath):
    """
    Loads MNIST data from a CSV file.
    Expected format: first column is the label (0-9), 
    the rest are 784 pixel values.
    """
    data = pd.read_csv(filepath)
    data = np.array(data)
    
    # Shuffle the data to ensure the model doesn't learn the order
    np.random.shuffle(data)
    
    # Separate Labels and Features
    # Y_raw shape: (m,) | X_raw shape: (m, 784)
    Y_raw = data[:, 0]
    X_raw = data[:, 1:]
    
    # 1. Transpose X to get (784, m)
    X = X_raw.T
    
    # 2. Standardize/Normalize the data (Scale pixels to 0-1)
    X = X / 255.0
    
    # 3. One-Hot Encode the labels
    Y = one_hot_encode(Y_raw, 10)
    
    return X, Y

def one_hot_encode(Y, num_classes):
    """
    Converts integer labels into one-hot vectors.
    Example: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    """
    m = Y.shape[0]
    one_hot_Y = np.zeros((num_classes, m))
    one_hot_Y[Y, np.arange(m)] = 1
    return one_hot_Y