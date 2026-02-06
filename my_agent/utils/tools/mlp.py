import numpy as np
import uuid
from typing import List, Optional, Any, Dict, Literal
from pydantic import Field, BaseModel
from langchain_core.tools import tool

# In-memory model storage
MODEL_STORE: Dict[str, Any] = {}


def _compute_auc_roc_ovr(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> Dict[str, Any]:
    """Compute AUC-ROC using One-vs-Rest for multi-class classification."""
    per_class_auc = {}
    auc_values = []
    
    for c in range(n_classes):
        y_binary = (y_true == c).astype(int)
        scores = y_proba[:, c]
        
        n_pos = np.sum(y_binary == 1)
        n_neg = np.sum(y_binary == 0)
        
        if n_pos == 0 or n_neg == 0:
            per_class_auc[f"class_{c}"] = 0.0
            continue
        
        sorted_indices = np.argsort(-scores)
        y_sorted = y_binary[sorted_indices]
        
        tpr_list = [0.0]
        fpr_list = [0.0]
        tp_count = 0
        fp_count = 0
        
        for i in range(len(y_sorted)):
            if y_sorted[i] == 1:
                tp_count += 1
            else:
                fp_count += 1
            tpr_list.append(tp_count / n_pos)
            fpr_list.append(fp_count / n_neg)
        
        auc = 0.0
        for i in range(1, len(fpr_list)):
            auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2.0
        
        per_class_auc[f"class_{c}"] = float(auc)
        auc_values.append(auc)
    
    macro_auc = float(np.mean(auc_values)) if auc_values else 0.0
    return {"macro_auc_roc": macro_auc, "per_class_auc_roc": per_class_auc}


def calculate_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
                                  y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics for multi-class classification.
    
    Metrics computed:
        - Accuracy: (TP + TN) / Total
        - Precision: TP / (TP + FP) (macro-averaged)
        - Recall: TP / (TP + FN) (macro-averaged)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall) (macro-averaged)
        - AUC-ROC: Area under ROC curve (One-vs-Rest, when probabilities available)
        - Confusion Matrix: NxN matrix
        - Per-class metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    accuracy = np.mean(y_true == y_pred)
    
    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[f"class_{c}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "support": int(np.sum(y_true == c))
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[true_label, pred_label] += 1
    
    result = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class_metrics": per_class,
        "confusion_matrix": confusion_matrix.tolist(),
        "total_samples": int(len(y_true)),
        "correct_predictions": int(np.sum(y_true == y_pred)),
        "incorrect_predictions": int(np.sum(y_true != y_pred))
    }
    
    if y_proba is not None:
        auc_results = _compute_auc_roc_ovr(y_true, y_proba, n_classes)
        result["auc_roc"] = auc_results["macro_auc_roc"]
        result["per_class_auc_roc"] = auc_results["per_class_auc_roc"]
    
    return result


class MLP:
    """
    Multi-Layer Perceptron (MLP) neural network for classification.
    
    A fully-connected feedforward neural network with configurable architecture.
    Supports multiple hidden layers, various activation functions (ReLU, Sigmoid, Tanh),
    and modern optimizers (Adam, SGD, RMSProp).
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
            self.params[f'W{i}'] = np.random.randn(
                layer_dims[i-1], layer_dims[i]
            ) * np.sqrt(2 / layer_dims[i-1])
            self.params[f'b{i}'] = np.zeros((1, layer_dims[i]))
            
            if self.optimizer_name in ['adam', 'rmsprop']:
                self.opt_state[f'v_W{i}'] = np.zeros_like(self.params[f'W{i}'])
                self.opt_state[f'v_b{i}'] = np.zeros_like(self.params[f'b{i}'])
                if self.optimizer_name == 'adam':
                    self.opt_state[f'm_W{i}'] = np.zeros_like(self.params[f'W{i}'])
                    self.opt_state[f'm_b{i}'] = np.zeros_like(self.params[f'b{i}'])

    def _activation(self, Z: np.ndarray, deriv: bool = False) -> np.ndarray:
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
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.cache['A0'] = X
        L = len(self.hidden_layers)
        
        for i in range(1, L + 1):
            Z = np.dot(self.cache[f'A{i-1}'], self.params[f'W{i}']) + self.params[f'b{i}']
            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i}'] = self._activation(Z)
        
        Z_out = np.dot(self.cache[f'A{L}'], self.params[f'W{L+1}']) + self.params[f'b{L+1}']
        self.cache[f'Z{L+1}'] = Z_out
        self.cache[f'A{L+1}'] = self._softmax(Z_out)
        return self.cache[f'A{L+1}']

    def backward(self, Y: np.ndarray, n_samples: int) -> None:
        L = len(self.hidden_layers)
        dZ = self.cache[f'A{L+1}'] - Y
        
        self.grads[f'dW{L+1}'] = np.dot(self.cache[f'A{L}'].T, dZ) / n_samples
        self.grads[f'db{L+1}'] = np.sum(dZ, axis=0, keepdims=True) / n_samples
        
        for i in range(L, 0, -1):
            dA = np.dot(dZ, self.params[f'W{i+1}'].T)
            dZ = dA * self._activation(self.cache[f'Z{i}'], deriv=True)
            self.grads[f'dW{i}'] = np.dot(self.cache[f'A{i-1}'].T, dZ) / n_samples
            self.grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / n_samples

    def _update_params(self, t: int) -> None:
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
                self.opt_state[f'm_W{i}'] = beta1 * self.opt_state[f'm_W{i}'] + (1 - beta1) * grad_w
                self.opt_state[f'm_b{i}'] = beta1 * self.opt_state[f'm_b{i}'] + (1 - beta1) * grad_b
                self.opt_state[f'v_W{i}'] = beta2 * self.opt_state[f'v_W{i}'] + (1 - beta2) * (grad_w**2)
                self.opt_state[f'v_b{i}'] = beta2 * self.opt_state[f'v_b{i}'] + (1 - beta2) * (grad_b**2)
                
                m_w_hat = self.opt_state[f'm_W{i}'] / (1 - beta1**t)
                v_w_hat = self.opt_state[f'v_W{i}'] / (1 - beta2**t)
                m_b_hat = self.opt_state[f'm_b{i}'] / (1 - beta1**t)
                v_b_hat = self.opt_state[f'v_b{i}'] / (1 - beta2**t)
                
                self.params[f'W{i}'] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                self.params[f'b{i}'] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    def _compute_loss(self, Y: np.ndarray, Y_pred: np.ndarray) -> float:
        epsilon = 1e-15
        Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(Y * np.log(Y_pred), axis=1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLP':
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
                
                Y_pred = self.forward(batch_X)
                epoch_loss += self._compute_loss(batch_y, Y_pred)
                n_batches += 1
                
                self.backward(batch_y, end - start)
                self._update_params(iter_count)
                iter_count += 1
            
            self._training_history.append(epoch_loss / n_batches)
        
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions.")
        X = np.asarray(X)
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions.")
        return self.forward(np.asarray(X))


def cross_validate_mlp(X: np.ndarray, y_onehot: np.ndarray, k: int = 5,
                        hidden_layers: List[int] = [64, 32],
                        activation: str = "relu",
                        learning_rate: float = 0.001,
                        max_epochs: int = 500,
                        batch_size: int = 32,
                        optimizer: str = "adam") -> Dict[str, Any]:
    """Perform k-fold cross-validation for MLP."""
    n_samples = len(X)
    n_classes = y_onehot.shape[1]
    y_indices = np.argmax(y_onehot, axis=1)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // k
    
    fold_accuracies = []
    fold_f1s = []
    
    for fold in range(k):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < k - 1 else n_samples
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        model = MLP(hidden_layers=hidden_layers, activation=activation,
                     learning_rate=learning_rate, max_epochs=max_epochs,
                     batch_size=batch_size, optimizer=optimizer)
        model.fit(X[train_idx], y_onehot[train_idx])
        y_pred_fold = model.predict(X[val_idx])
        
        fold_metrics = calculate_multiclass_metrics(y_indices[val_idx], y_pred_fold, n_classes)
        fold_accuracies.append(fold_metrics["accuracy"])
        fold_f1s.append(fold_metrics["macro_f1"])
    
    return {
        "k_folds": k,
        "cv_accuracy_mean": float(np.mean(fold_accuracies)),
        "cv_accuracy_std": float(np.std(fold_accuracies)),
        "cv_f1_mean": float(np.mean(fold_f1s)),
        "cv_f1_std": float(np.std(fold_f1s)),
        "fold_accuracies": [float(a) for a in fold_accuracies],
        "fold_f1_scores": [float(f) for f in fold_f1s]
    }


class TrainMLPInput(BaseModel):
    X_train: List[List[float]] = Field(description="Training feature matrix as a 2D list of shape (n_samples, n_features)")
    y_train: List[List[float]] = Field(description="One-hot encoded labels as a 2D list of shape (n_samples, n_classes)")
    hidden_layers: List[int] = Field(default=[64, 32], description="Number of neurons in each hidden layer")
    activation: Literal["relu", "sigmoid", "tanh"] = Field(default="relu", description="Activation function")
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.01, description="Learning rate for optimizer")
    max_epochs: int = Field(default=500, ge=100, le=2000, description="Maximum number of training epochs")
    batch_size: int = Field(default=32, ge=16, le=128, description="Mini-batch size")
    optimizer: Literal["adam", "sgd", "rmsprop"] = Field(default="adam", description="Optimization algorithm")
    cross_validate: bool = Field(default=False, description="Whether to perform k-fold cross-validation")
    cv_folds: int = Field(default=5, ge=2, le=10, description="Number of folds for cross-validation")


class InferenceMLPInput(BaseModel):
    model_id: str = Field(description="The unique model ID returned from train_mlp_tool")
    X_test: List[List[float]] = Field(description="Test feature matrix as a 2D list of shape (n_samples, n_features)")
    return_probabilities: bool = Field(default=False, description="If True, return class probabilities instead of class indices")
    y_true: Optional[List[int]] = Field(default=None, description="Optional ground truth class indices for computing metrics")


@tool(args_schema=TrainMLPInput)
def train_mlp_tool(
    X_train: List[List[float]],
    y_train: List[List[float]],
    hidden_layers: List[int] = [64, 32],
    activation: Literal["relu", "sigmoid", "tanh"] = "relu",
    learning_rate: float = 0.001,
    max_epochs: int = 500,
    batch_size: int = 32,
    optimizer: Literal["adam", "sgd", "rmsprop"] = "adam",
    cross_validate: bool = False,
    cv_folds: int = 5,
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
            - model_id: Unique identifier for the trained model
            - status: "success" or "error"
            - message: Status message
            - n_features: Number of input features
            - n_classes: Number of output classes
            - n_samples: Number of training samples
            - architecture: Network architecture [input, hidden..., output]
            - final_loss: Training loss at the end
            - training_history: Loss values per epoch
    """
    try:
        X = np.array(X_train)
        y = np.array(y_train)
        
        if len(X.shape) != 2:
            return {"status": "error", "message": "X_train must be a 2D array"}
        if len(y.shape) != 2:
            return {"status": "error", "message": "y_train must be a 2D one-hot encoded array"}
        if X.shape[0] != y.shape[0]:
            return {"status": "error", "message": "X_train and y_train must have same number of samples"}
        if not hidden_layers:
            return {"status": "error", "message": "hidden_layers cannot be empty"}
        
        model = MLP(
            hidden_layers=hidden_layers,
            activation=activation,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            optimizer=optimizer
        )
        model.fit(X, y)
        
        model_id = f"mlp_{uuid.uuid4().hex[:8]}"
        MODEL_STORE[model_id] = model
        
        architecture = [X.shape[1]] + hidden_layers + [y.shape[1]]
        
        result = {
            "status": "success",
            "message": f"MLP trained successfully. Architecture: {architecture}",
            "model_id": model_id,
            "n_features": X.shape[1],
            "n_classes": y.shape[1],
            "n_samples": X.shape[0],
            "architecture": architecture,
            "final_loss": model._training_history[-1] if model._training_history else None,
            "training_history": model._training_history[-10:]
        }
        
        # Cross-validation
        if cross_validate:
            cv_results = cross_validate_mlp(
                X, y, k=cv_folds,
                hidden_layers=hidden_layers,
                activation=activation,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                batch_size=batch_size,
                optimizer=optimizer
            )
            result["cross_validation"] = cv_results
            result["message"] += f". CV accuracy: {cv_results['cv_accuracy_mean']:.3f} +/- {cv_results['cv_accuracy_std']:.3f}"
        
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool(args_schema=InferenceMLPInput)
def inference_mlp_tool(
    model_id: str,
    X_test: List[List[float]],
    return_probabilities: bool = False,
    y_true: Optional[List[int]] = None
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
    
    **Metrics (when y_true is provided):**
    If ground truth labels are provided, computes:
    - Accuracy: Overall correctness
    - Macro Precision/Recall/F1: Averaged across classes
    - Per-class metrics: Precision, recall, F1 for each class
    - Confusion Matrix: NxN matrix of true vs predicted
    
    Args:
        model_id: Unique identifier from train_mlp_tool.
        X_test: Test features as a 2D list (n_samples, n_features).
        return_probabilities: Whether to return probabilities. Default: False.
        y_true: Optional ground truth class indices for computing metrics.
    
    Returns:
        Dict containing:
            - status: "success" or "error"
            - message: Status message
            - predictions: Predicted classes
              or probabilities
            - n_samples: Number of samples predicted
            - metrics: Classification metrics (only if y_true provided)
    """
    try:
        if model_id not in MODEL_STORE:
            return {"status": "error", "message": f"Model '{model_id}' not found."}
        
        model = MODEL_STORE[model_id]
        X = np.array(X_test)
        
        if len(X.shape) != 2:
            return {"status": "error", "message": "X_test must be a 2D array"}
        if X.shape[1] != model._n_features:
            return {"status": "error", "message": f"X_test has {X.shape[1]} features but model expects {model._n_features}"}
        
        # Always compute probabilities (needed for AUC-ROC)
        proba = model.predict_proba(X)
        
        if return_probabilities:
            predictions = proba.tolist()
        else:
            predictions = model.predict(X).tolist()
        
        result = {
            "status": "success",
            "message": f"Successfully predicted {len(predictions)} samples",
            "predictions": predictions,
            "n_samples": len(predictions)
        }
        
        if y_true is not None and not return_probabilities:
            y_true_arr = np.array(y_true)
            y_pred_arr = np.array(predictions)
            
            if len(y_true_arr) != len(y_pred_arr):
                return {"status": "error", "message": f"y_true length ({len(y_true_arr)}) must match X_test samples ({len(y_pred_arr)})"}
            
            metrics = calculate_multiclass_metrics(y_true_arr, y_pred_arr, model._n_classes, y_proba=proba)
            result["metrics"] = metrics
            msg = f"Successfully predicted {len(predictions)} samples with {metrics['accuracy']*100:.1f}% accuracy"
            if "auc_roc" in metrics:
                msg += f", AUC-ROC: {metrics['auc_roc']:.3f}"
            result["message"] = msg
        
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}