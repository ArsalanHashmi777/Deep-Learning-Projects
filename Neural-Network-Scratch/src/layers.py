import numpy as np

def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(1) # For consistency
    parameters = {}
    L = len(layer_dims)            

    for l in range(1, L):
        # He Initialization (Professional standard for ReLU layers)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

def linear_forward(A_prev, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    Formula: Z = W * A_prev + b
    """
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b) # Stored for backpropagation
    
    return Z, cache