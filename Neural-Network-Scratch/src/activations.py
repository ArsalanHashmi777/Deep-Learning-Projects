import numpy as np

def relu(Z):
    """
    Computes the Rectified Linear Unit activation.
    Formula: f(z) = max(0, z)
    """
    # We use np.maximum to ensure element-wise comparison against 0
    return np.maximum(0, Z)

def relu_derivative(Z):
    """
    Computes the gradient of ReLU for backpropagation.
    Returns 1 for z > 0, and 0 otherwise.
    """
    # This creates a boolean array and converts it to integers (1s and 0s)
    return (Z > 0).astype(int)

def softmax(Z):
    """
    Computes the Softmax activation for the output layer.
    Formula: exp(zi) / sum(exp(zj))
    """
    # Substracting np.max(Z) is a "stability trick" to prevent 
    # the exp() function from exploding into Infinity
    exp_z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)