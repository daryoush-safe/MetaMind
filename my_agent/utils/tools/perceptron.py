import numpy as np
import uuid
from typing import List, Optional, Any, Dict
from pydantic import Field, BaseModel
from langchain_core.tools import tool

# In-memory model storage (in production, use Redis or a database)
MODEL_STORE: Dict[str, Any] = {}


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculate classification metrics for binary classification.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
    
    Returns:
        Dictionary with accuracy, precision, recall, f1_score, confusion_matrix
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Basic counts
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "confusion_matrix": {
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        },
        "total_samples": int(len(y_true)),
        "correct_predictions": int(tp + tn),
        "incorrect_predictions": int(fp + fn)
    }


class Perceptron:
    """
    A single-layer Perceptron classifier for binary classification tasks.
    
    The Perceptron is a fundamental neural network unit that learns a linear
    decision boundary to separate two classes. It uses the Heaviside step
    function for activation and updates weights using the Perceptron Learning Rule.
    
    Attributes:
        learning_rate (float): Step size for weight updates during training.
        max_epochs (int): Maximum number of passes over the training data.
        use_bias (bool): Whether to include a bias term in the model.
        weights (np.ndarray): Learned weight vector for input features.
        bias_weight (float): Learned bias term (if use_bias is True).
    
    Example:
        >>> perceptron = Perceptron(learning_rate=0.01, max_epochs=100)
        >>> perceptron.fit(X_train, y_train)
        >>> predictions = perceptron.predict(X_test)
    """
    
    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 100, bias: bool = True):
        """
        Initialize the Perceptron classifier.
        
        Args:
            learning_rate (float): Step size for weight updates. Higher values
                lead to faster but potentially unstable learning. Range: 0.001-0.1.
                Default: 0.01.
            max_epochs (int): Maximum number of iterations over the entire dataset.
                Training stops early if the model converges. Range: 50-1000.
                Default: 100.
            bias (bool): Whether to include a bias term. The bias allows the
                decision boundary to be offset from the origin. Default: True.
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.use_bias = bias
        self.weights: Optional[np.ndarray] = None
        self.bias_weight: float = 0.0
        self._is_fitted: bool = False
        self._training_history: List[Dict] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Perceptron':
        """
        Train the Perceptron on the provided data.
        
        Uses the Perceptron Learning Rule: w = w + lr * (target - predicted) * x
        Training continues until convergence (no errors) or max_epochs is reached.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features).
            y (np.ndarray): Binary target labels of shape (n_samples,).
                Values should be 0 or 1.
        
        Returns:
            Perceptron: The fitted model instance (self).
        
        Raises:
            ValueError: If X and y have incompatible shapes.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples. Got {X.shape[0]} and {y.shape[0]}")
        
        n_samples, n_features = X.shape
        
        # Initialize weights to zeros
        self.weights = np.zeros(n_features)
        self.bias_weight = 0.0
        self._training_history = []
        
        for epoch in range(self.max_epochs):
            errors = 0
            for i in range(n_samples):
                # Calculate linear output
                linear_output = np.dot(X[i], self.weights)
                if self.use_bias:
                    linear_output += self.bias_weight
                
                # Heaviside step function activation
                y_pred = 1 if linear_output >= 0 else 0
                
                # Perceptron update rule
                update = self.learning_rate * (y[i] - y_pred)
                
                self.weights += update * X[i]
                if self.use_bias:
                    self.bias_weight += update
                
                if update != 0:
                    errors += 1
            
            # Track training progress
            self._training_history.append({
                "epoch": epoch + 1,
                "errors": errors,
                "error_rate": errors / n_samples
            })
            
            # Early stopping if converged (no misclassifications)
            if errors == 0:
                break
        
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Predicted class labels (0 or 1) of shape (n_samples,).
        
        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions. Call fit() first.")
        
        X = np.asarray(X)
        linear_output = np.dot(X, self.weights)
        if self.use_bias:
            linear_output += self.bias_weight
        return np.where(linear_output >= 0, 1, 0)
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters as a dictionary."""
        return {
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs,
            'use_bias': self.use_bias,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias_weight': self.bias_weight,
            'is_fitted': self._is_fitted
        }


class TrainPerceptronInput(BaseModel):
    X_train: List[List[float]] = Field(description="Training feature matrix as a 2D list of shape (n_samples, n_features)")
    y_train: List[int] = Field(description="Training labels as a list of binary values (0 or 1)")
    learning_rate: float = Field(default=0.01, ge=0.001, le=0.1, description="Learning rate for weight updates. Higher values mean faster but potentially unstable learning.")
    max_epochs: int = Field(default=100, ge=50, le=1000, description="Maximum number of training iterations over the dataset.")
    bias: bool = Field(default=True, description="Whether to include a bias term in the model.")


class InferencePerceptronInput(BaseModel):
    model_id: str = Field(description="The unique model ID returned from train_perceptron_tool")
    X_test: List[List[float]] = Field(description="Test feature matrix as a 2D list of shape (n_samples, n_features)")
    y_true: Optional[List[int]] = Field(default=None, description="Optional ground truth labels for computing metrics (list of 0s and 1s)")


@tool(args_schema=TrainPerceptronInput)
def train_perceptron_tool(
    X_train: List[List[float]],
    y_train: List[int],
    learning_rate: float = 0.01,
    max_epochs: int = 100,
    bias: bool = True,
) -> Dict[str, Any]:
    """
    Train a Perceptron classifier on the provided dataset.
    
    The Perceptron is a fundamental neural network unit suitable for linearly
    separable binary classification problems. It learns a linear decision boundary
    using the Perceptron Learning Rule.
    
    **When to use:**
    - Binary classification tasks
    - When data is linearly separable or nearly so
    - When interpretability is important (weights directly show feature importance)
    - As a baseline before trying more complex models
    
    **Limitations:**
    - Cannot solve non-linearly separable problems (e.g., XOR)
    - Only supports binary classification
    - May not converge if data is not linearly separable
    
    **Parameters Guide:**
    - learning_rate: Start with 0.01. Increase to 0.1 for faster training or
      decrease to 0.001 for more stable convergence.
    - max_epochs: 100 is usually sufficient for small datasets. Increase for
      larger or more complex datasets.
    - bias: Keep True unless you specifically want the decision boundary to
      pass through the origin.

    Returns:
        Dict containing:
            - model_id: Unique identifier for the trained model
            - status: "success" or "error"
            - message: Status message
            - weights: Learned feature weights
            - bias_weight: Learned bias term
            - n_features: Number of input features
            - n_samples: Number of training samples
            - converged: Whether training converged
            - epochs_run: Actual number of epochs run
    """
    try:
        X = np.array(X_train)
        y = np.array(y_train)
        
        # Validate inputs
        if len(X.shape) != 2:
            return {"status": "error", "message": "X_train must be a 2D array"}
        if len(y.shape) != 1:
            return {"status": "error", "message": "y_train must be a 1D array"}
        if X.shape[0] != y.shape[0]:
            return {"status": "error", "message": f"X_train and y_train must have same number of samples"}
        
        # Create and train model
        model = Perceptron(
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            bias=bias
        )
        model.fit(X, y)
        
        # Store model with unique ID
        model_id = f"perceptron_{uuid.uuid4().hex[:8]}"
        MODEL_STORE[model_id] = model
        
        # Check convergence
        converged = len(model._training_history) < max_epochs or model._training_history[-1]["errors"] == 0
        
        return {
            "status": "success",
            "message": f"Perceptron trained successfully on {X.shape[0]} samples with {X.shape[1]} features",
            "model_id": model_id,
            "weights": model.weights.tolist(),
            "bias_weight": model.bias_weight,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "converged": converged,
            "epochs_run": len(model._training_history),
            "final_error_rate": model._training_history[-1]["error_rate"] if model._training_history else None
        }
        
    except Exception as e:
        print("error during perceptron:", e)
        return {"status": "error", "message": str(e)}


@tool(args_schema=InferencePerceptronInput)
def inference_perceptron_tool(
    model_id: str,
    X_test: List[List[float]],
    y_true: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Make predictions using a trained Perceptron model.
    
    Uses a previously trained Perceptron model (identified by model_id) to
    predict binary class labels for new samples.
    
    **Usage:**
    1. First train a model using train_perceptron_tool to get a model_id
    2. Use this tool with that model_id to make predictions
    
    Returns:
        Dict containing:
            - status: "success" or "error"
            - message: Status message
            - predictions: Predicted class labels (0 or 1)
            - n_samples: Number of samples predicted
            - metrics: Classification metrics (only if y_true provided)
    """
    try:
        # Retrieve model
        if model_id not in MODEL_STORE:
            return {"status": "error", "message": f"Model '{model_id}' not found. Train a model first."}
        
        model = MODEL_STORE[model_id]
        X = np.array(X_test)
        
        # Validate input dimensions
        if len(X.shape) != 2:
            return {"status": "error", "message": "X_test must be a 2D array"}
        if X.shape[1] != len(model.weights):
            return {
                "status": "error", 
                "message": f"X_test has {X.shape[1]} features but model expects {len(model.weights)}"
            }
        
        # Make predictions
        predictions = model.predict(X)
        
        result = {
            "status": "success",
            "message": f"Successfully predicted {len(predictions)} samples",
            "predictions": predictions.tolist(),
            "n_samples": len(predictions)
        }
        
        # Calculate metrics if ground truth is provided
        if y_true is not None:
            y_true_arr = np.array(y_true)
            if len(y_true_arr) != len(predictions):
                return {"status": "error", "message": f"y_true length ({len(y_true_arr)}) must match X_test samples ({len(predictions)})"}
            
            metrics = calculate_classification_metrics(y_true_arr, predictions)
            result["metrics"] = metrics
            result["message"] = f"Successfully predicted {len(predictions)} samples with {metrics['accuracy']*100:.1f}% accuracy"
        
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}