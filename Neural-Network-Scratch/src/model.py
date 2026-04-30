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

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  W = W - learning_rate * dW
                  b = b - learning_rate * db
    """
    L = len(parameters) // 2 # number of layers

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        
    return parameters

def train(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements an L-layer neural network: [LINEAR->RELU]*(L-1) -> LINEAR->SOFTMAX.
    
    Arguments:
    X -- data, numpy array of shape (num_features, number of examples)
    Y -- true "label" vector, shape (10, number of examples)
    layer_dims -- list containing the input size and each layer size
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    costs -- list of costs (useful for plotting the learning curve)
    """
    np.random.seed(1)
    costs = []                         

    # 1. Initialize parameters
    from src.layers import initialize_parameters
    parameters = initialize_parameters(layer_dims)
    
    # 2. Optimization Loop
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SOFTMAX
        AL, caches = model_forward(X, parameters)
        
        # Compute cost
        cost = compute_cost(AL, Y)
        
        # Backward propagation
        grads = model_backward(AL, Y, caches)
        
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Record and print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
            costs.append(cost)
            
    return parameters, costs

"""
It is an excellent observation—the "Expert" way is always to prioritize vectorization wherever possible. However, in deep learning, we actually use a combination of both to get the job done. 

Here is the breakdown of why we used a `for` loop for the layers while still keeping the project "vectorized."

### 1. Layers are "Serial," Not "Parallel"
While we want to process all **images** at the same time (which we did with NumPy), we cannot process the **layers** at the same time.
*   **The Dependency:** Layer 2 cannot start its calculation until Layer 1 has finished. Layer 3 has to wait for Layer 2, and so on.
*   **The Loop:** Because each layer depends on the output of the one before it, we use a `for` loop to move the data through the "stack" of layers sequentially.

### 2. We ARE Using Vectorization (Inside the Loop)
It is important to notice that inside each iteration of that `for` loop, we are using pure NumPy vectorization:
*   **The Math:** When we write `np.dot(W, A_prev) + b`, we are multiplying thousands of weights by thousands of pixels across the entire batch of MNIST images simultaneously.
*   **The Speed:** Because we use NumPy arrays for the data ($X$) and the parameters ($W, b$), we are already leveraging the high-performance C-code that runs under the hood of NumPy.

### 3. Flexibility for "Deep" Networks
By using a loop to iterate through the `parameters` dictionary, your code is now **dynamic**. 
*   If you want to change your model from a 3-layer network to a 100-layer "Deep" network, you don't have to rewrite your math. 
*   You simply change the `layer_dims` list, and the `for` loop handles the rest automatically.

---

### Summary: Loop vs. Vectorization
| Component | Method | Why? |
| :--- | :--- | :--- |
| **Across Examples** | **Vectorized** | To process the whole MNIST batch at once for speed. |
| **Across Layers** | **For Loop** | Because Layer $L$ requires the output of Layer $L-1$. |
| **Inside a Layer** | **Vectorized** | To perform matrix multiplication ($W \cdot A + b$) efficiently. |
"""