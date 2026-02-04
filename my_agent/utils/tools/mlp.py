import numpy as np
from typing import List, Literal
from pydantic import Field
from langchain_core.tools import tool


class MLP:
    def __init__(self, hidden_layers=[64, 32], activation="relu", learning_rate=0.001, 
                 max_epochs=500, batch_size=32, optimizer="adam"):
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer_name = optimizer
        self.params = {}
        self.cache = {}
        self.grads = {}
        self.opt_state = {}  # For Adam/RMSProp state

    def _initialize_params(self, input_dim, output_dim):
        layer_dims = [input_dim] + self.hidden_layers + [output_dim]
        
        for i in range(1, len(layer_dims)):
            # He initialization
            self.params[f'W{i}'] = np.random.randn(layer_dims[i-1], layer_dims[i]) * np.sqrt(2 / layer_dims[i-1])
            self.params[f'b{i}'] = np.zeros((1, layer_dims[i]))
            
            # Initialize optimizer state
            if self.optimizer_name in ['adam', 'rmsprop']:
                self.opt_state[f'v_W{i}'] = np.zeros_like(self.params[f'W{i}'])
                self.opt_state[f'v_b{i}'] = np.zeros_like(self.params[f'b{i}'])
                if self.optimizer_name == 'adam':
                    self.opt_state[f'm_W{i}'] = np.zeros_like(self.params[f'W{i}'])
                    self.opt_state[f'm_b{i}'] = np.zeros_like(self.params[f'b{i}'])

    def _activation(self, Z, deriv=False):
        if self.activation_name == "relu":
            if deriv: return (Z > 0).astype(float)
            return np.maximum(0, Z)
        elif self.activation_name == "sigmoid":
            s = 1 / (1 + np.exp(-Z))
            if deriv: return s * (1 - s)
            return s
        elif self.activation_name == "tanh":
            t = np.tanh(Z)
            if deriv: return 1 - t**2
            return t
        return Z

    def _softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        self.cache['A0'] = X
        L = len(self.hidden_layers)
        
        # Hidden layers
        for i in range(1, L + 1):
            Z = np.dot(self.cache[f'A{i-1}'], self.params[f'W{i}']) + self.params[f'b{i}']
            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i}'] = self._activation(Z)
            
        # Output layer (assume Softmax/Classification for generality)
        Z_out = np.dot(self.cache[f'A{L}'], self.params[f'W{L+1}']) + self.params[f'b{L+1}']
        self.cache[f'Z{L+1}'] = Z_out
        self.cache[f'A{L+1}'] = self._softmax(Z_out)
        return self.cache[f'A{L+1}']

    def backward(self, Y, n_samples):
        L = len(self.hidden_layers)
        # Output layer gradient (Softmax + Cross Entropy derivative is P - Y)
        dZ = self.cache[f'A{L+1}'] - Y
        
        self.grads[f'dW{L+1}'] = np.dot(self.cache[f'A{L}'].T, dZ) / n_samples
        self.grads[f'db{L+1}'] = np.sum(dZ, axis=0, keepdims=True) / n_samples
        
        # Backprop through hidden layers
        for i in range(L, 0, -1):
            dA = np.dot(dZ, self.params[f'W{i+1}'].T)
            dZ = dA * self._activation(self.cache[f'Z{i}'], deriv=True)
            self.grads[f'dW{i}'] = np.dot(self.cache[f'A{i-1}'].T, dZ) / n_samples
            self.grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / n_samples

    def _update_params(self, t):
        L = len(self.hidden_layers) + 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        
        for i in range(1, L + 1):
            grad_w = self.grads[f'dW{i}']
            grad_b = self.grads[f'db{i}']
            
            if self.optimizer_name == "sgd":
                self.params[f'W{i}'] -= self.learning_rate * grad_w
                self.params[f'b{i}'] -= self.learning_rate * grad_b
                
            elif self.optimizer_name == "rmsprop":
                self.opt_state[f'v_W{i}'] = beta1 * self.opt_state[f'v_W{i}'] + (1 - beta1) * (grad_w**2)
                self.opt_state[f'v_b{i}'] = beta1 * self.opt_state[f'v_b{i}'] + (1 - beta1) * (grad_b**2)
                self.params[f'W{i}'] -= self.learning_rate * grad_w / (np.sqrt(self.opt_state[f'v_W{i}']) + epsilon)
                self.params[f'b{i}'] -= self.learning_rate * grad_b / (np.sqrt(self.opt_state[f'v_b{i}']) + epsilon)

            elif self.optimizer_name == "adam":
                # Momentum
                self.opt_state[f'm_W{i}'] = beta1 * self.opt_state[f'm_W{i}'] + (1 - beta1) * grad_w
                self.opt_state[f'm_b{i}'] = beta1 * self.opt_state[f'm_b{i}'] + (1 - beta1) * grad_b
                # RMSProp-like
                self.opt_state[f'v_W{i}'] = beta2 * self.opt_state[f'v_W{i}'] + (1 - beta2) * (grad_w**2)
                self.opt_state[f'v_b{i}'] = beta2 * self.opt_state[f'v_b{i}'] + (1 - beta2) * (grad_b**2)
                
                # Bias correction
                m_w_hat = self.opt_state[f'm_W{i}'] / (1 - beta1**t)
                v_w_hat = self.opt_state[f'v_W{i}'] / (1 - beta2**t)
                m_b_hat = self.opt_state[f'm_b{i}'] / (1 - beta1**t)
                v_b_hat = self.opt_state[f'v_b{i}'] / (1 - beta2**t)
                
                self.params[f'W{i}'] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                self.params[f'b{i}'] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    def fit(self, X, y):
        # y should be one-hot encoded
        n_samples, n_features = X.shape
        n_classes = y.shape[1]
        self._initialize_params(n_features, n_classes)
        
        iter_count = 1
        for epoch in range(self.max_epochs):
            # Shuffle
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]
            
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_X = X[start:end]
                batch_y = y[start:end]
                
                self.forward(batch_X)
                self.backward(batch_y, end-start)
                self._update_params(iter_count)
                iter_count += 1


@tool
def mlp_tool(
    hidden_layers: List[int] = Field(default=[64, 32]),
    activation: Literal["relu", "sigmoid", "tanh"] = Field(default="relu"),
    learning_rate: float = Field(default=0.01, ge=0.001, le=0.1),
    max_epochs: int = Field(default=100, ge=50, le=1000),
    batch_size: int = Field(default=32, ge=16, le=128),
    optimizer: Literal["adam", "sgd", "rmsprop"] = Field(default="adam"),
):
    """Creates an MLP tool with specified hyperparameters."""
    pass
