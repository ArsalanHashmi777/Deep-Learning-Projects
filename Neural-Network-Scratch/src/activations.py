import numpy as np

def relu(Z):
    """
    Computes the Rectified Linear Unit activation.
    Formula: A = max(0, Z)
    """
    return np.maximum(0, Z)

def softmax(Z):
    """
    Computes the Softmax activation for the output layer.
    Formula: exp(Zi) / sum(exp(Z))
    """
    # Subtracting np.max(Z) is a professional trick to prevent 
    # 'Exploding Gradients' or overflow during exponentiation.
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def relu_backward(dA, Z):
    """
    The derivative of ReLU needed for backpropagation.
    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ