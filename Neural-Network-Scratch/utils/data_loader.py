import numpy as np
import pandas as pd

def load_mnist_data(file_path):
    """
    Loads MNIST data from a CSV file using pandas for speed.
    """
    # Using pandas is significantly faster than raw numpy for large CSVs
    data = pd.read_csv(file_path)
    
    # Convert to numpy array
    data = np.array(data)
    
    # Shuffle the data to ensure the model doesn't learn the order of the rows
    np.random.shuffle(data)
    
    # Separate Labels (Y) and Features (X)
    # MNIST CSVs usually have the label in the first column
    Y = data[:, 0]
    X = data[:, 1:]
    
    return X, Y