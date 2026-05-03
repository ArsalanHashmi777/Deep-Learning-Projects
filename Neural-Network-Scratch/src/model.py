import numpy as np
from src.layers import linear_forward, linear_backward
from src.activations import relu, relu_derivative, softmax

def forward_propagation(X, parameters):
    """
    Coordinates the full forward pass: [Linear -> ReLU] * (L-1) -> [Linear -> Softmax]
    """
    caches = []
    A = X
    L = len(parameters) // 2 # Number of layers

    # 1. Hidden Layers: [Linear -> ReLU]
    for l in range(1, L):
        A_prev = A
        Z, linear_cache = linear_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)])
        A = relu(Z)
        # We store (linear_cache, Z) to use in backprop
        caches.append((linear_cache, Z))

    # 2. Output Layer: [Linear -> Softmax]
    ZL, linear_cache = linear_forward(A, parameters['W'+str(L)], parameters['b'+str(L)])
    AL = softmax(ZL)
    caches.append((linear_cache, ZL))
    
    return AL, caches

def compute_cost(AL, Y):
    """
    Calculates Categorical Cross-Entropy Loss.
    """
    m = Y.shape[1]
    # Small epsilon prevents log(0) which would crash the model
    cost = -1/m * np.sum(Y * np.log(AL + 1e-8))
    return np.squeeze(cost)

def backward_propagation(AL, Y, caches):
    """
    Coordinates the full backward pass through all layers.
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    
    # 1. Output Layer Gradient (Softmax + Cross-Entropy)
    # This simplification dZ = AL - Y is a math "magic trick" for this specific combo
    dZL = AL - Y
    current_cache = caches[L-1]
    linear_cache, _ = current_cache
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZL, linear_cache)
    
    # 2. Hidden Layers Gradients: [ReLU -> Linear]
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        linear_cache, Z = current_cache
        
        # dZ = dA * relu_derivative(Z)
        dZ = grads["dA" + str(l+1)] * relu_derivative(Z)
        grads["dA" + str(l)], grads["dW" + str(l+1)], grads["db" + str(l+1)] = linear_backward(dZ, linear_cache)
        
    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Standard Gradient Descent update step.
    """
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return parameters

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