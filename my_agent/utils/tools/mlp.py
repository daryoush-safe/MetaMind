import numpy as np
import uuid
from typing import List, Optional, Any, Dict, Literal
from pydantic import Field
from langchain_core.tools import tool

# In-memory model storage
MODEL_STORE: Dict[str, Any] = {}


class MLP:
    """
    Multi-Layer Perceptron (MLP) neural network for classification.
    
    A fully-connected feedforward neural network with configurable architecture.
    Supports multiple hidden layers, various activation functions (ReLU, Sigmoid, Tanh),
    and modern optimizers (Adam, SGD, RMSProp).
    
    The network uses:
    - He initialization for weights
    - Softmax output layer for multi-class classification
    - Cross-entropy loss function
    - Mini-batch gradient descent
    
    Attributes:
        hidden_layers (List[int]): Number of neurons in each hidden layer.
        activation_name (str): Activation function for hidden layers.
        learning_rate (float): Learning rate for optimization.
        max_epochs (int): Maximum training epochs.
        batch_size (int): Mini-batch size for training.
        optimizer_name (str): Optimization algorithm.
        params (Dict): Network weights and biases.
    
    Example:
        >>> mlp = MLP(hidden_layers=[64, 32], activation="relu", optimizer="adam")
        >>> mlp.fit(X_train, y_train_onehot)
        >>> predictions = mlp.predict(X_test)
    """
    
    def __init__(
        self, 
        hidden_layers: List[int] = [64, 32], 
        activation: str = "relu", 
        learning_rate: float = 0.001,
        max_epochs: int = 500, 
        batch_size: int = 32, 
        optimizer: str = "adam"
    ):
        """
        Initialize the MLP neural network.
        
        Args:
            hidden_layers: List of integers specifying neurons per hidden layer.
                Example: [128, 64, 32] creates 3 hidden layers. Default: [64, 32].
            activation: Activation function for hidden layers. Options:
                - "relu": Rectified Linear Unit (recommended for deep networks)
                - "sigmoid": Logistic sigmoid (good for binary outputs)
                - "tanh": Hyperbolic tangent (zero-centered outputs)
                Default: "relu".
            learning_rate: Step size for optimizer. Range: 0.0001-0.01.
                Default: 0.001.
            max_epochs: Maximum training epochs. Range: 100-2000. Default: 500.
            batch_size: Mini-batch size. Larger batches are more stable but
                use more memory. Range: 16-128. Default: 32.
            optimizer: Optimization algorithm. Options:
                - "adam": Adaptive moment estimation (recommended)
                - "sgd": Stochastic gradient descent
                - "rmsprop": RMSProp adaptive learning rate
                Default: "adam".
        """
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer_name = optimizer
        self.params: Dict[str, np.ndarray] = {}
        self.cache: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.opt_state: Dict[str, np.ndarray] = {}
        self._is_fitted: bool = False
        self._n_classes: int = 0
        self._n_features: int = 0
        self._training_history: List[float] = []

    def _initialize_params(self, input_dim: int, output_dim: int) -> None:
        """Initialize network parameters using He initialization."""
        layer_dims = [input_dim] + self.hidden_layers + [output_dim]
        
        for i in range(1, len(layer_dims)):
            # He initialization: sqrt(2 / n_in)
            self.params[f'W{i}'] = np.random.randn(
                layer_dims[i-1], layer_dims[i]
            ) * np.sqrt(2 / layer_dims[i-1])
            self.params[f'b{i}'] = np.zeros((1, layer_dims[i]))
            
            # Initialize optimizer state
            if self.optimizer_name in ['adam', 'rmsprop']:
                self.opt_state[f'v_W{i}'] = np.zeros_like(self.params[f'W{i}'])
                self.opt_state[f'v_b{i}'] = np.zeros_like(self.params[f'b{i}'])
                if self.optimizer_name == 'adam':
                    self.opt_state[f'm_W{i}'] = np.zeros_like(self.params[f'W{i}'])
                    self.opt_state[f'm_b{i}'] = np.zeros_like(self.params[f'b{i}'])

    def _activation(self, Z: np.ndarray, deriv: bool = False) -> np.ndarray:
        """Apply activation function or its derivative."""
        if self.activation_name == "relu":
            if deriv:
                return (Z > 0).astype(float)
            return np.maximum(0, Z)
        elif self.activation_name == "sigmoid":
            s = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
            if deriv:
                return s * (1 - s)
            return s
        elif self.activation_name == "tanh":
            t = np.tanh(Z)
            if deriv:
                return 1 - t**2
            return t
        return Z

    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities with numerical stability."""
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        self.cache['A0'] = X
        L = len(self.hidden_layers)
        
        # Hidden layers
        for i in range(1, L + 1):
            Z = np.dot(self.cache[f'A{i-1}'], self.params[f'W{i}']) + self.params[f'b{i}']
            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i}'] = self._activation(Z)
        
        # Output layer with softmax
        Z_out = np.dot(self.cache[f'A{L}'], self.params[f'W{L+1}']) + self.params[f'b{L+1}']
        self.cache[f'Z{L+1}'] = Z_out
        self.cache[f'A{L+1}'] = self._softmax(Z_out)
        return self.cache[f'A{L+1}']

    def backward(self, Y: np.ndarray, n_samples: int) -> None:
        """Backward pass to compute gradients."""
        L = len(self.hidden_layers)
        
        # Output layer gradient (softmax + cross-entropy)
        dZ = self.cache[f'A{L+1}'] - Y
        
        self.grads[f'dW{L+1}'] = np.dot(self.cache[f'A{L}'].T, dZ) / n_samples
        self.grads[f'db{L+1}'] = np.sum(dZ, axis=0, keepdims=True) / n_samples
        
        # Backprop through hidden layers
        for i in range(L, 0, -1):
            dA = np.dot(dZ, self.params[f'W{i+1}'].T)
            dZ = dA * self._activation(self.cache[f'Z{i}'], deriv=True)
            self.grads[f'dW{i}'] = np.dot(self.cache[f'A{i-1}'].T, dZ) / n_samples
            self.grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / n_samples

    def _update_params(self, t: int) -> None:
        """Update parameters using the selected optimizer."""
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

    def _compute_loss(self, Y: np.ndarray, Y_pred: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        epsilon = 1e-15
        Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(Y * np.log(Y_pred), axis=1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLP':
        """
        Train the MLP on the provided data.
        
        Args:
            X: Training features of shape (n_samples, n_features).
            y: One-hot encoded targets of shape (n_samples, n_classes).
        
        Returns:
            MLP: The fitted model instance (self).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        n_classes = y.shape[1]
        
        self._n_features = n_features
        self._n_classes = n_classes
        self._initialize_params(n_features, n_classes)
        
        iter_count = 1
        self._training_history = []
        
        for epoch in range(self.max_epochs):
            # Shuffle data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_X = X_shuffled[start:end]
                batch_y = y_shuffled[start:end]
                
                # Forward pass
                Y_pred = self.forward(batch_X)
                
                # Compute loss
                epoch_loss += self._compute_loss(batch_y, Y_pred)
                n_batches += 1
                
                # Backward pass
                self.backward(batch_y, end - start)
                
                # Update parameters
                self._update_params(iter_count)
                iter_count += 1
            
            self._training_history.append(epoch_loss / n_batches)
        
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Predicted class indices of shape (n_samples,).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions.")
        
        X = np.asarray(X)
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions.")
        
        return self.forward(np.asarray(X))


@tool
def train_mlp_tool(
    X_train: List[List[float]] = Field(description="Training feature matrix as a 2D list of shape (n_samples, n_features)"),
    y_train: List[List[float]] = Field(description="One-hot encoded labels as a 2D list of shape (n_samples, n_classes). Example: [[1,0,0], [0,1,0], [0,0,1]] for 3 classes"),
    hidden_layers: List[int] = Field(default=[64, 32], description="Number of neurons in each hidden layer. Example: [128, 64, 32] creates 3 hidden layers"),
    activation: Literal["relu", "sigmoid", "tanh"] = Field(default="relu", description="Activation function: 'relu' (recommended), 'sigmoid', or 'tanh'"),
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.01, description="Learning rate for optimizer. Lower values give more stable training"),
    max_epochs: int = Field(default=500, ge=100, le=2000, description="Maximum number of training epochs"),
    batch_size: int = Field(default=32, ge=16, le=128, description="Mini-batch size. Larger batches are more stable but slower per epoch"),
    optimizer: Literal["adam", "sgd", "rmsprop"] = Field(default="adam", description="Optimization algorithm: 'adam' (recommended), 'sgd', or 'rmsprop'")
) -> Dict[str, Any]:
    """
    Train a Multi-Layer Perceptron (MLP) neural network for classification.
    
    The MLP is a powerful feedforward neural network capable of learning complex
    non-linear patterns. It consists of multiple fully-connected layers with
    configurable activation functions and uses backpropagation for training.
    
    **When to use:**
    - Multi-class or binary classification tasks
    - When data has complex non-linear relationships
    - Medium to large datasets (hundreds to millions of samples)
    - When accuracy is more important than interpretability
    
    **Architecture recommendations:**
    - Start with 2 hidden layers: [64, 32] or [128, 64]
    - For complex problems, try deeper: [256, 128, 64, 32]
    - Use 'relu' activation for hidden layers (faster training)
    - Use 'adam' optimizer (adaptive learning rate)
    
    **Parameter tuning guide:**
    - learning_rate: Start with 0.001. If loss doesn't decrease, try 0.0001.
      If training is slow, try 0.01.
    - batch_size: 32 is a good default. Use 64 or 128 for large datasets.
    - max_epochs: Monitor training loss. Increase if loss is still decreasing.
    
    **Important:** Labels must be one-hot encoded. For example, with 3 classes:
    - Class 0: [1, 0, 0]
    - Class 1: [0, 1, 0]
    - Class 2: [0, 0, 1]
    
    Args:
        X_train: Training features as a 2D list (n_samples, n_features).
        y_train: One-hot encoded labels as a 2D list (n_samples, n_classes).
        hidden_layers: Neurons per hidden layer. Default: [64, 32].
        activation: Activation function. Default: "relu".
        learning_rate: Optimizer learning rate (0.0001-0.01). Default: 0.001.
        max_epochs: Maximum training epochs (100-2000). Default: 500.
        batch_size: Mini-batch size (16-128). Default: 32.
        optimizer: Optimization algorithm. Default: "adam".
    
    Returns:
        Dict containing:
            - model_id (str): Unique identifier for the trained model
            - status (str): "success" or "error"
            - message (str): Status message
            - n_features (int): Number of input features
            - n_classes (int): Number of output classes
            - n_samples (int): Number of training samples
            - architecture (List[int]): Network architecture [input, hidden..., output]
            - final_loss (float): Training loss at the end
            - training_history (List[float]): Loss values per epoch
    
    Example:
        >>> # Train on Iris-like data (4 features, 3 classes)
        >>> result = train_mlp_tool(
        ...     X_train=[[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], ...],
        ...     y_train=[[1, 0, 0], [0, 1, 0], ...],
        ...     hidden_layers=[64, 32],
        ...     activation="relu",
        ...     optimizer="adam",
        ...     max_epochs=500
        ... )
        >>> print(result['model_id'])  # Use this ID for inference
    """
    try:
        X = np.array(X_train)
        y = np.array(y_train)
        
        # Validate inputs
        if len(X.shape) != 2:
            return {"status": "error", "message": "X_train must be a 2D array"}
        if len(y.shape) != 2:
            return {"status": "error", "message": "y_train must be a 2D one-hot encoded array"}
        if X.shape[0] != y.shape[0]:
            return {"status": "error", "message": "X_train and y_train must have same number of samples"}
        if not hidden_layers:
            return {"status": "error", "message": "hidden_layers cannot be empty"}
        
        # Create and train model
        model = MLP(
            hidden_layers=hidden_layers,
            activation=activation,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            optimizer=optimizer
        )
        model.fit(X, y)
        
        # Store model
        model_id = f"mlp_{uuid.uuid4().hex[:8]}"
        MODEL_STORE[model_id] = model
        
        architecture = [X.shape[1]] + hidden_layers + [y.shape[1]]
        
        return {
            "status": "success",
            "message": f"MLP trained successfully. Architecture: {architecture}",
            "model_id": model_id,
            "n_features": X.shape[1],
            "n_classes": y.shape[1],
            "n_samples": X.shape[0],
            "architecture": architecture,
            "final_loss": model._training_history[-1] if model._training_history else None,
            "training_history": model._training_history[-10:]  # Last 10 epochs
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool
def inference_mlp_tool(
    model_id: str = Field(description="The unique model ID returned from train_mlp_tool"),
    X_test: List[List[float]] = Field(description="Test feature matrix as a 2D list of shape (n_samples, n_features)"),
    return_probabilities: bool = Field(default=False, description="If True, return class probabilities instead of class indices")
) -> Dict[str, Any]:
    """
    Make predictions using a trained MLP model.
    
    Uses a previously trained MLP model to predict class labels or probabilities
    for new samples. The model is identified by the model_id returned from
    train_mlp_tool.
    
    **Usage:**
    1. Train a model using train_mlp_tool to get a model_id
    2. Use this tool with that model_id to make predictions
    
    **Output modes:**
    - return_probabilities=False: Returns predicted class indices (0, 1, 2, ...)
    - return_probabilities=True: Returns probability distribution over classes
    
    Args:
        model_id: Unique identifier from train_mlp_tool.
        X_test: Test features as a 2D list (n_samples, n_features).
        return_probabilities: Whether to return probabilities. Default: False.
    
    Returns:
        Dict containing:
            - status (str): "success" or "error"
            - message (str): Status message
            - predictions (List[int] or List[List[float]]): Predicted classes
              or probabilities
            - n_samples (int): Number of samples predicted
    
    Example:
        >>> result = inference_mlp_tool(
        ...     model_id="mlp_abc12345",
        ...     X_test=[[5.0, 3.4, 1.5, 0.2]],
        ...     return_probabilities=True
        ... )
        >>> print(result['predictions'])  # [[0.95, 0.03, 0.02]]
    """
    try:
        if model_id not in MODEL_STORE:
            return {"status": "error", "message": f"Model '{model_id}' not found. Train a model first."}
        
        model = MODEL_STORE[model_id]
        X = np.array(X_test)
        
        if len(X.shape) != 2:
            return {"status": "error", "message": "X_test must be a 2D array"}
        if X.shape[1] != model._n_features:
            return {
                "status": "error",
                "message": f"X_test has {X.shape[1]} features but model expects {model._n_features}"
            }
        
        if return_probabilities:
            predictions = model.predict_proba(X).tolist()
        else:
            predictions = model.predict(X).tolist()
        
        return {
            "status": "success",
            "message": f"Successfully predicted {len(predictions)} samples",
            "predictions": predictions,
            "n_samples": len(predictions)
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}