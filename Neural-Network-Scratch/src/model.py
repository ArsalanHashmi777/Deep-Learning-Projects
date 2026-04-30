import numpy as np
from src.layers import linear_forward
from src.activations import relu, softmax

def model_forward(X, parameters):
    """
    Implements the full forward propagation.
    
    Arguments:
    X -- input data (e.g., flattened MNIST images)
    parameters -- output of initialize_parameters()
    
    Returns:
    AL -- last post-activation value (probabilities)
    caches -- list of caches containing every cache of linear_activation_forward()
    """
    caches = []
    A = X
    # Calculate number of layers (divided by 2 because of W and b pairs)
    L = len(parameters) // 2                  

    # [LINEAR -> RELU] * (L-1)
    for l in range(1, L):
        A_prev = A 
        Z, linear_cache = linear_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)])
        A = relu(Z)
        # We store (A_prev, W, b, Z) to make backprop easier later
        caches.append((linear_cache, Z))

    # [LINEAR -> SOFTMAX] for the output layer
    ZL, linear_cache = linear_forward(A, parameters['W' + str(L)], parameters['b' + str(L)])
    AL = softmax(ZL)
    caches.append((linear_cache, ZL))
            
    return AL, caches