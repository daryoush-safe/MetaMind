import numpy as np
import uuid
from typing import List, Optional, Any, Dict, Literal, Tuple
from pydantic import Field, BaseModel
from langchain_core.tools import tool

# In-memory model storage
MODEL_STORE: Dict[str, Any] = {}

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dictionary with MSE, MAE, RMSE, R², MAPE
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    n = len(y_true)
    
    # Mean Squared Error
    mse = float(np.mean((y_true - y_pred) ** 2))
    
    # Root Mean Squared Error
    rmse = float(np.sqrt(mse))
    
    # Mean Absolute Error
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # R² Score (coefficient of determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    # Mean Absolute Percentage Error (avoid division by zero)
    mask = y_true != 0
    if np.any(mask):
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float('inf')
    
    # Max Error
    max_error = float(np.max(np.abs(y_true - y_pred)))
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "mape": mape,
        "max_error": max_error,
        "n_samples": n,
        "explained_variance_ratio": r2
    }


class FuzzyController:
    """
    Fuzzy Logic Controller with automatic rule generation.
    
    A fuzzy inference system that learns fuzzy rules from data using the
    Wang-Mendel method. The system fuzzifies inputs, evaluates learned rules,
    and defuzzifies to produce crisp outputs.
    """
    
    def __init__(
        self,
        n_membership_functions: int = 3,
        membership_type: str = "triangular",
        defuzzification: str = "centroid",
        rule_generation: str = "wang_mendel"
    ):
        self.n_mf = n_membership_functions
        self.mf_type = membership_type
        self.defuz_method = defuzzification
        self.rule_gen_method = rule_generation
        self.rules: Dict = {}
        self.var_ranges: List[Tuple[float, float]] = []
        self.mf_params: List[List[Tuple[float, float]]] = []
        self._is_fitted: bool = False
        self._n_features: int = 0

    def _membership(self, x: float, center: float, width: float) -> float:
        """Calculate membership degree for a value in a fuzzy set."""
        if self.mf_type == "triangular":
            a, b, c = center - width, center, center + width
            if width == 0:
                return 1.0 if x == center else 0.0
            return max(min((x - a) / (b - a + 1e-10), (c - x) / (c - b + 1e-10)), 0)
        elif self.mf_type == "gaussian":
            sigma = width / 2.0
            return np.exp(-((x - center) ** 2) / (2 * sigma ** 2 + 1e-10))
        elif self.mf_type == "trapezoidal":
            a = center - width
            b = center - width / 2
            c = center + width / 2
            d = center + width
            if width == 0:
                return 1.0 if x == center else 0.0
            return max(min((x - a) / (b - a + 1e-10), 1, (d - x) / (d - c + 1e-10)), 0)
        return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FuzzyController':
        """Learn fuzzy rules from data using Wang-Mendel method."""
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        data = np.hstack((X, y))
        n_features = data.shape[1]
        self._n_features = X.shape[1]
        
        self.var_ranges = []
        self.mf_params = []
        
        for i in range(n_features):
            d_min, d_max = np.min(data[:, i]), np.max(data[:, i])
            self.var_ranges.append((d_min, d_max))
            step = (d_max - d_min) / (self.n_mf - 1) if self.n_mf > 1 else 0
            params = []
            for m in range(self.n_mf):
                center = d_min + m * step
                width = step if step > 0 else 1.0
                params.append((center, width))
            self.mf_params.append(params)

        if self.rule_gen_method == "wang_mendel":
            raw_rules = []
            for row in data:
                rule_indices = []
                degree = 1.0
                for i, val in enumerate(row):
                    mu_vals = [self._membership(val, p[0], p[1]) for p in self.mf_params[i]]
                    idx = np.argmax(mu_vals)
                    rule_indices.append(idx)
                    degree *= mu_vals[idx]
                raw_rules.append((tuple(rule_indices[:-1]), rule_indices[-1], degree))

            rule_dict = {}
            for inputs, output, deg in raw_rules:
                if inputs in rule_dict:
                    if deg > rule_dict[inputs]['degree']:
                        rule_dict[inputs] = {'out': output, 'degree': deg}
                else:
                    rule_dict[inputs] = {'out': output, 'degree': deg}
            self.rules = rule_dict
        
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output values using fuzzy inference."""
        if not self._is_fitted:
            raise RuntimeError("Controller must be fitted before prediction.")
        
        X = np.asarray(X)
        predictions = []
        
        for row in X:
            out_min, out_max = self.var_ranges[-1]
            y_points = np.linspace(out_min, out_max, 100)
            agg_output = np.zeros_like(y_points)
            
            for rule_in, rule_meta in self.rules.items():
                out_idx = rule_meta['out']
                mu_vals = []
                for i, r_idx in enumerate(rule_in):
                    c, w = self.mf_params[i][r_idx]
                    mu_vals.append(self._membership(row[i], c, w))
                firing_strength = np.min(mu_vals) if mu_vals else 0
                
                if firing_strength > 0:
                    c_out, w_out = self.mf_params[-1][out_idx]
                    rule_output = np.array([
                        min(firing_strength, self._membership(y, c_out, w_out))
                        for y in y_points
                    ])
                    agg_output = np.maximum(agg_output, rule_output)

            if np.sum(agg_output) == 0:
                predictions.append((out_min + out_max) / 2)
                continue

            if self.defuz_method == "centroid":
                predictions.append(np.sum(y_points * agg_output) / np.sum(agg_output))
            elif self.defuz_method == "mom":
                max_val = np.max(agg_output)
                indices = np.where(agg_output == max_val)[0]
                predictions.append(np.mean(y_points[indices]))
            else:
                predictions.append(np.sum(y_points * agg_output) / np.sum(agg_output))
        
        return np.array(predictions)
    

class TrainFuzzyInput(BaseModel):
    X_train: List[List[float]] = Field(description="Training input features as a 2D list (n_samples, n_features)")
    y_train: List[float] = Field(description="Training output values as a 1D list (n_samples,)")
    n_membership_functions: Literal[3, 5, 7] = Field(default=3, description="Number of fuzzy sets per variable")
    membership_type: Literal["triangular", "gaussian", "trapezoidal"] = Field(default="triangular")
    defuzzification: Literal["centroid", "bisector", "mom", "som", "lom"] = Field(default="centroid")
    rule_generation: Literal["wang_mendel", "manual"] = Field(default="wang_mendel")


class InferenceFuzzyInput(BaseModel):
    model_id: str = Field(description="Model ID from train_fuzzy_tool")
    X_test: List[List[float]] = Field(description="Test input features")
    y_true: Optional[List[float]] = Field(default=None)


@tool(args_schema=TrainFuzzyInput)
def train_fuzzy_tool(
    X_train: List[List[float]],
    y_train: List[float],
    n_membership_functions: Literal[3, 5, 7] = 3,
    membership_type: Literal["triangular", "gaussian", "trapezoidal"] = "triangular",
    defuzzification: Literal["centroid", "bisector", "mom", "som", "lom"] = "centroid",
    rule_generation: Literal["wang_mendel", "manual"] = "wang_mendel",
) -> Dict[str, Any]:
    """
    Train a Fuzzy Logic Controller for regression/control tasks.
    
    Fuzzy controllers use linguistic variables and IF-THEN rules to model
    complex relationships. They're particularly useful when interpretability
    is important and the system has inherent vagueness or imprecision.
    
    Use Fuzzy System for: Control systems, Decision support systems requiring interpretable rules, Modeling expert knowledge, Regression with non-linear relationships
    
    **Membership functions:**
    - n_mf=3: "Low", "Medium", "High" - coarse granularity
    - n_mf=5: Adds "Very Low" and "Very High"
    - n_mf=7: Even finer granularity for precise control
    
    **Membership function types:**
    - triangular: Simple, efficient, good for most cases
    - gaussian: Smooth transitions, natural for sensor data
    - trapezoidal: Robust to noise, has "plateau" region
    
    Returns:
        Dict containing:
            - model_id: Unique identifier for the trained controller
            - status: "success" or "error"
            - n_rules: Number of learned rules
            - output_range: Min/max output values
    """
    try:
        X = np.array(X_train)
        y = np.array(y_train)
        
        if len(X.shape) != 2:
            return {"status": "error", "message": "X_train must be a 2D array"}
        if len(y.shape) != 1:
            return {"status": "error", "message": "y_train must be a 1D array"}
        if X.shape[0] != y.shape[0]:
            return {"status": "error", "message": "X_train and y_train must have same number of samples"}
        
        model = FuzzyController(
            n_membership_functions=n_membership_functions,
            membership_type=membership_type,
            defuzzification=defuzzification,
            rule_generation=rule_generation
        )
        model.fit(X, y)
        
        model_id = f"fuzzy_{uuid.uuid4().hex[:8]}"
        MODEL_STORE[model_id] = model
        
        # Calculate training metrics
        train_pred = model.predict(X)
        train_metrics = calculate_regression_metrics(y, train_pred)
        
        return {
            "status": "success",
            "message": f"Fuzzy controller trained with {len(model.rules)} rules",
            "model_id": model_id,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "n_rules": len(model.rules),
            "n_membership_functions": n_membership_functions,
            "output_range": model.var_ranges[-1] if model.var_ranges else None,
            "training_mse": train_metrics["mse"],
            "training_r2": train_metrics["r2_score"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool(args_schema=InferenceFuzzyInput)
def inference_fuzzy_tool(
    model_id: str,
    X_test: List[List[float]],
    y_true: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Make predictions using a trained Fuzzy Controller.
    
    Applies fuzzy inference to predict output values:
    1. Fuzzify inputs (calculate membership degrees)
    2. Evaluate rules (determine firing strengths)
    3. Aggregate rule outputs
    4. Defuzzify to get crisp predictions
    
    Returns:
        Dict containing:
            - status: "success" or "error"
            - predictions: Predicted output values
            - n_samples: Number of samples predicted
            - metrics: Metrics when y_true is provided:
    """
    try:
        if model_id not in MODEL_STORE:
            return {"status": "error", "message": f"Model '{model_id}' not found."}
        
        model = MODEL_STORE[model_id]
        X = np.array(X_test)
        
        if len(X.shape) != 2:
            return {"status": "error", "message": "X_test must be a 2D array"}
        if X.shape[1] != model._n_features:
            return {"status": "error", "message": f"Expected {model._n_features} features, got {X.shape[1]}"}
        
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
                return {"status": "error", "message": f"y_true length must match X_test samples"}
            
            metrics = calculate_regression_metrics(y_true_arr, predictions)
            result["metrics"] = metrics
            result["message"] = f"Successfully predicted {len(predictions)} samples with R²={metrics['r2_score']:.3f}"
        
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}