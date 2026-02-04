import numpy as np
from pydantic import Field
from langchain_core.tools import tool


class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=100, bias=True):
        """
        Args:
            learning_rate (float): Step size for weight updates (0.001-0.1).
            max_epochs (int): Maximum iterations over the dataset (50-1000).
            bias (bool): Whether to include a bias term.
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.use_bias = bias
        self.weights = None
        self.bias_weight = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias_weight = 0.0 if self.use_bias else 0.0
        
        for _ in range(self.max_epochs):
            errors = 0
            for i in range(n_samples):
                # Calculate linear output
                linear_output = np.dot(X[i], self.weights)
                if self.use_bias:
                    linear_output += self.bias_weight
                
                # Heaviside step function
                y_pred = 1 if linear_output >= 0 else 0
                
                # Update rule: w = w + lr * (target - predicted) * x
                update = self.learning_rate * (y[i] - y_pred)
                
                self.weights += update * X[i]
                if self.use_bias:
                    self.bias_weight += update
                
                if update != 0:
                    errors += 1
            
            # Early stopping if converged
            if errors == 0:
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights)
        if self.use_bias:
            linear_output += self.bias_weight
        return np.where(linear_output >= 0, 1, 0)


@tool
def perceptron_tool(
    learning_rate: float = Field(default=0.01, ge=0.001, le=0.1),
    max_epochs: int = Field(default=100, ge=50, le=1000),
    bias: bool = Field(default=True),
):
    """
    Create and return a Perceptron instance with specified hyperparameters.

    Args:
        learning_rate (float): Step size for weight updates (0.001-0.1).
        max_epochs (int): Maximum iterations over the dataset (50-1000).
        bias (bool): Whether to include a bias term.

    Returns:
        Perceptron: Configured Perceptron instance.
    """
    pass
