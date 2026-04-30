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

def compute_cost(AL, Y):
    """
    Computes the cross-entropy cost.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (10, number of examples)
    Y -- true "label" vector (one-hot encoded), shape (10, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]

    # Compute loss from aL and y.
    # We add a tiny epsilon (1e-8) to avoid log(0) errors.
    cost = - (1 / m) * np.sum(Y * np.log(AL + 1e-8))
    
    # Ensure the cost is a scalar value
    cost = np.squeeze(cost)      
    
    return cost

def linear_backward(dZ, cache):
    """
    Implements the linear portion of backward propagation for a single layer.
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def model_backward(AL, Y, caches):
    """
    Implements the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR->SOFTMAX network.
    """
    grads = {}
    L = len(caches) # number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # ensure Y is the same shape as AL
    
    # 1. Initializing the backpropagation (Derivative of cost w.r.t Z for Softmax/Cross-Entropy)
    # This is a specific simplification for Softmax + Cross-Entropy: dZ = AL - Y
    dZL = AL - Y
    
    # 2. Final Layer (SOFTMAX -> LINEAR) gradients
    current_cache = caches[L-1] # The (linear_cache, ZL) we stored in model_forward
    linear_cache_L, ZL = current_cache
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZL, linear_cache_L)
    
    # 3. Loop from L-2 to 0 for [RELU -> LINEAR] layers
    for l in reversed(range(L-1)):
        # current_cache is (linear_cache, Z)
        linear_cache, Z = caches[l]
        
        # Calculate dZ for the ReLU layer
        from src.activations import relu_backward
        dZ = relu_backward(grads["dA" + str(l + 1)], Z)
        
        # Calculate gradients for this layer
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads