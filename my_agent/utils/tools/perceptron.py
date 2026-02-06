import numpy as np
import uuid
from typing import List, Optional, Any, Dict
from pydantic import Field, BaseModel
from langchain_core.tools import tool

# In-memory model storage (in production, use Redis or a database)
MODEL_STORE: Dict[str, Any] = {}


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                      y_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics for binary classification.
    
    Metrics computed:
        - Accuracy: (TP + TN) / Total
        - Precision: TP / (TP + FP)
        - Recall: TP / (TP + FN)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        - AUC-ROC: Area under ROC curve (when y_scores provided)
        - Confusion Matrix: [[TN, FP], [FN, TP]]
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_scores: Predicted scores/probabilities for AUC-ROC computation
    
    Returns:
        Dictionary with comprehensive classification metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Basic counts
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # --- Accuracy ---
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # --- Precision ---
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # --- Recall ---
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # --- F1 Score ---
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # --- Confusion Matrix: [[TN, FP], [FN, TP]] ---
    confusion_matrix = [[int(tn), int(fp)], [int(fn), int(tp)]]
    
    result = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "confusion_matrix": confusion_matrix,
        "confusion_matrix_detail": {
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        },
        "total_samples": int(len(y_true)),
        "correct_predictions": int(tp + tn),
        "incorrect_predictions": int(fp + fn)
    }
    
    # --- AUC-ROC (when continuous scores are available) ---
    if y_scores is not None:
        y_scores = np.asarray(y_scores).flatten()
        auc_roc = _compute_auc_roc(y_true, y_scores)
        result["auc_roc"] = float(auc_roc)
    
    return result


def _compute_auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute AUC-ROC using the trapezoidal rule.
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_scores: Predicted scores/probabilities (continuous)
    
    Returns:
        float: Area under the ROC curve (0.0 to 1.0)
    """
    # Sort by score descending
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.0
    
    # Compute TPR and FPR at each threshold
    tpr_list = [0.0]
    fpr_list = [0.0]
    tp_count = 0
    fp_count = 0
    
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp_count += 1
        else:
            fp_count += 1
        tpr_list.append(tp_count / n_pos)
        fpr_list.append(fp_count / n_neg)
    
    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2.0
    
    return auc


def cross_validate_perceptron(X: np.ndarray, y: np.ndarray, k: int = 5,
                                learning_rate: float = 0.01, max_epochs: int = 100,
                                bias: bool = True) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation for Perceptron.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        k: Number of folds
        learning_rate: Perceptron learning rate
        max_epochs: Max training epochs
        bias: Whether to use bias
    
    Returns:
        Dictionary with cross-validation results
    """
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // k
    
    fold_accuracies = []
    fold_f1s = []
    fold_precisions = []
    fold_recalls = []
    
    for fold in range(k):
        # Split into train and validation
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < k - 1 else n_samples
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X[val_indices]
        y_val_fold = y[val_indices]
        
        # Train and predict
        model = Perceptron(learning_rate=learning_rate, max_epochs=max_epochs, bias=bias)
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_val_fold)
        
        # Compute metrics for this fold
        fold_metrics = calculate_classification_metrics(y_val_fold, y_pred_fold)
        fold_accuracies.append(fold_metrics["accuracy"])
        fold_f1s.append(fold_metrics["f1_score"])
        fold_precisions.append(fold_metrics["precision"])
        fold_recalls.append(fold_metrics["recall"])
    
    return {
        "k_folds": k,
        "cv_accuracy_mean": float(np.mean(fold_accuracies)),
        "cv_accuracy_std": float(np.std(fold_accuracies)),
        "cv_f1_mean": float(np.mean(fold_f1s)),
        "cv_f1_std": float(np.std(fold_f1s)),
        "cv_precision_mean": float(np.mean(fold_precisions)),
        "cv_precision_std": float(np.std(fold_precisions)),
        "cv_recall_mean": float(np.mean(fold_recalls)),
        "cv_recall_std": float(np.std(fold_recalls)),
        "fold_accuracies": [float(a) for a in fold_accuracies],
        "fold_f1_scores": [float(f) for f in fold_f1s]
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
        
        Returns:
            Perceptron: The fitted model instance (self).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples. Got {X.shape[0]} and {y.shape[0]}")
        
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias_weight = 0.0
        self._training_history = []
        
        for epoch in range(self.max_epochs):
            errors = 0
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights)
                if self.use_bias:
                    linear_output += self.bias_weight
                
                y_pred = 1 if linear_output >= 0 else 0
                
                update = self.learning_rate * (y[i] - y_pred)
                
                self.weights += update * X[i]
                if self.use_bias:
                    self.bias_weight += update
                
                if update != 0:
                    errors += 1
            
            self._training_history.append({
                "epoch": epoch + 1,
                "errors": errors,
                "error_rate": errors / n_samples
            })
            
            if errors == 0:
                break
        
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions. Call fit() first.")
        
        X = np.asarray(X)
        linear_output = np.dot(X, self.weights)
        if self.use_bias:
            linear_output += self.bias_weight
        return np.where(linear_output >= 0, 1, 0)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute raw decision scores (linear output before thresholding).
        Used for AUC-ROC computation.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Raw decision scores of shape (n_samples,).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before computing decision function.")
        
        X = np.asarray(X)
        linear_output = np.dot(X, self.weights)
        if self.use_bias:
            linear_output += self.bias_weight
        return linear_output
    
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
    learning_rate: float = Field(default=0.01, ge=0.001, le=0.1, description="Learning rate for weight updates.")
    max_epochs: int = Field(default=100, ge=50, le=1000, description="Maximum number of training iterations over the dataset.")
    bias: bool = Field(default=True, description="Whether to include a bias term in the model.")
    cross_validate: bool = Field(default=False, description="Whether to perform k-fold cross-validation")
    cv_folds: int = Field(default=5, ge=2, le=10, description="Number of folds for cross-validation")


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
    cross_validate: bool = False,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """
    Train a Perceptron classifier on the provided dataset.
    
    The Perceptron is a fundamental neural network unit suitable for linearly
    separable binary classification problems.
    
    **When to use:**
    - Binary classification tasks
    - When data is linearly separable or nearly so
    - When interpretability is important
    - As a baseline before trying more complex models
    
    **Metrics computed (on inference with y_true):**
    - Accuracy: (TP + TN) / Total
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    - AUC-ROC: Area under ROC curve
    - Confusion Matrix: [[TN, FP], [FN, TP]]
    - Cross-Validation Score: Mean accuracy across k folds (when enabled)

    Returns:
        Dict containing:
            - model_id: Unique identifier for the trained model
            - status: "success" or "error"
            - weights: Learned feature weights
            - converged: Whether training converged
            - cross_validation: CV results (when cross_validate=True)
    """
    try:
        X = np.array(X_train)
        y = np.array(y_train)
        
        if len(X.shape) != 2:
            return {"status": "error", "message": "X_train must be a 2D array"}
        if len(y.shape) != 1:
            return {"status": "error", "message": "y_train must be a 1D array"}
        if X.shape[0] != y.shape[0]:
            return {"status": "error", "message": f"X_train and y_train must have same number of samples"}
        
        model = Perceptron(
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            bias=bias
        )
        model.fit(X, y)
        
        model_id = f"perceptron_{uuid.uuid4().hex[:8]}"
        MODEL_STORE[model_id] = model
        
        converged = len(model._training_history) < max_epochs or model._training_history[-1]["errors"] == 0
        
        result = {
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
        
        # Cross-validation
        if cross_validate:
            cv_results = cross_validate_perceptron(
                X, y, k=cv_folds,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                bias=bias
            )
            result["cross_validation"] = cv_results
            result["message"] += f". CV accuracy: {cv_results['cv_accuracy_mean']:.3f} +/- {cv_results['cv_accuracy_std']:.3f}"
        
        return result
        
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
    
    **Metrics (when y_true is provided):**
    - Accuracy: (TP + TN) / Total
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    - AUC-ROC: Area under ROC curve
    - Confusion Matrix: [[TN, FP], [FN, TP]]
    
    Returns:
        Dict containing:
            - predictions: Predicted class labels (0 or 1)
            - metrics: Comprehensive classification metrics (when y_true provided)
    """
    try:
        if model_id not in MODEL_STORE:
            return {"status": "error", "message": f"Model '{model_id}' not found. Train a model first."}
        
        model = MODEL_STORE[model_id]
        X = np.array(X_test)
        
        if len(X.shape) != 2:
            return {"status": "error", "message": "X_test must be a 2D array"}
        if X.shape[1] != len(model.weights):
            return {
                "status": "error", 
                "message": f"X_test has {X.shape[1]} features but model expects {len(model.weights)}"
            }
        
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
            
            # Get decision scores for AUC-ROC
            y_scores = model.decision_function(X)
            
            metrics = calculate_classification_metrics(y_true_arr, predictions, y_scores=y_scores)
            result["metrics"] = metrics
            result["message"] = f"Successfully predicted {len(predictions)} samples with {metrics['accuracy']*100:.1f}% accuracy"
            if "auc_roc" in metrics:
                result["message"] += f", AUC-ROC: {metrics['auc_roc']:.3f}"
        
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}