# Neural-Network-Scratch: MNIST from the Ground Up

A modular, pure NumPy implementation of a Deep Neural Network designed to classify handwritten digits from the MNIST dataset. This project was built to master the underlying mathematics of backpropagation and gradient descent without the use of high-level frameworks like PyTorch or TensorFlow.

## 🚀 Project Highlights
*   **Built from Scratch**: Manual implementation of Linear layers, ReLU activations, and Softmax.
*   **Vectorized Math**: Optimized using NumPy matrix operations for efficient CPU training.
*   **Modular Architecture**: Clean separation of concerns with dedicated modules for activations, layers, and preprocessing.
*   **91.19% Test Accuracy**: Achieved high generalization performance on unseen data.

## 📂 Project Structure
```text
Neural-Network-Scratch/
├── data/               # MNIST CSV files (train & test)
├── src/                # Core Math Engine
│   ├── activations.py  # ReLU, Softmax, and their derivatives
│   ├── layers.py       # Linear forward/backward & He Initialization
│   └── model.py        # Forward/Backward propagation orchestration
├── utils/              # Data Pipeline
│   ├── data_loader.py  # Efficient CSV loading
│   └── preprocessing.py# Normalization & One-Hot Encoding
├── notebooks/          # Exploration & Training
│   ├── 01_mnist_exploration.ipynb
│   └── 02_model_training_visuals.ipynb
└── Optimization_Experiments/
    └── Learning_Rate_Search.ipynb
```

## 📊 Results & Visualization
The model was trained using a learning rate of `0.1` over 200 epochs.

### Cost Function
The cost curve shows a smooth convergence, validating the mathematical correctness of the backpropagation logic.

### Confusion Matrix
The model demonstrates strong performance across all digits, with minor common confusions (e.g., 4 vs 9).

## 🛠️ How to Run
1. Clone the repository.
2. Install dependencies: `pip install numpy pandas matplotlib seaborn scikit-learn`.
3. Run `notebooks/02_model_training_visuals.ipynb` to train the final model.

## 🧠 What I Learned
*   **He Initialization**: Essential for preventing vanishing/exploding gradients in ReLU networks.
*   **Categorical Cross-Entropy**: How to derive the gradient of Softmax combined with Cross-Entropy.
*   **Hyperparameter Tuning**: Comparing learning rates ($0.5$, $0.1$, $0.01$) to find the optimal "step size" for convergence.
```
