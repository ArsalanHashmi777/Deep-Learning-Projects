import numpy as np

def init_params(layer_dims):
    """
    Initializes weights and biases for each layer.
    Input: layer_dims -- list containing the dimensions of each layer [784, 64, 32, 10]
    """
    params = {}
    L = len(layer_dims) # Number of layers in the network

    for l in range(1, L):
        # He Initialization: Good for ReLU activations to keep gradients stable
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return params

def linear_forward(A_prev, W, b):
    """
    Computes the linear part of a layer's forward propagation.
    Z = W.A_prev + b
    """
    Z = np.dot(W, A_prev) + b
    # We save A_prev, W, and b in a cache because we need them for backprop later[cite: 3]
    cache = (A_prev, W, b)
    return Z, cache

def linear_backward(dZ, cache):
    """
    Computes the gradients dW, db, and dA_prev using the chain rule.[cite: 3]
    """
    A_prev, W, b = cache
    m = A_prev.shape[1] # Number of examples

    # Math: The gradients of our weights and biases[cite: 3]
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ) # This "passes" the error back to the previous layer[cite: 3]

    return dA_prev, dW, db