import numpy as np
import copy
import random
import uuid
from typing import List, Optional, Any, Dict, Callable
from pydantic import Field
from langchain_core.tools import tool

# In-memory model storage
MODEL_STORE: Dict[str, Any] = {}

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate regression metrics."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    n = len(y_true)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if np.any(mask) else float('inf')
    max_error = float(np.max(np.abs(y_true - y_pred)))
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "mape": mape,
        "max_error": max_error,
        "n_samples": n
    }


class GPNode:
    """
    Node in a GP expression tree.
    
    Each node is either a terminal (variable or constant) or a function
    (operator with children).
    """
    
    def __init__(self, val: Any, children: Optional[List['GPNode']] = None):
        self.val = val
        self.children = children if children else []
        self.is_terminal = len(self.children) == 0

    def evaluate(self, x: float) -> float:
        """Evaluate the expression tree for input x."""
        if self.is_terminal:
            if self.val == 'x':
                return x
            return float(self.val)
        
        args = [c.evaluate(x) for c in self.children]
        
        if self.val == '+':
            return args[0] + args[1]
        if self.val == '-':
            return args[0] - args[1]
        if self.val == '*':
            return args[0] * args[1]
        if self.val == '/':
            # Protected division
            return args[0] / (args[1] if abs(args[1]) > 1e-6 else 1)
        if self.val == 'sin':
            return np.sin(args[0])
        if self.val == 'cos':
            return np.cos(args[0])
        if self.val == 'exp':
            return np.exp(np.clip(args[0], -10, 10))
        return 0

    def size(self) -> int:
        """Count total nodes in the tree."""
        return 1 + sum(c.size() for c in self.children)
    
    def depth(self) -> int:
        """Calculate tree depth."""
        if self.is_terminal:
            return 1
        return 1 + max(c.depth() for c in self.children)
    
    def to_string(self) -> str:
        """Convert tree to human-readable expression."""
        if self.is_terminal:
            if self.val == 'x':
                return 'x'
            return f"{self.val:.3f}" if isinstance(self.val, float) else str(self.val)
        
        if self.val in ['sin', 'cos', 'exp']:
            return f"{self.val}({self.children[0].to_string()})"
        
        left = self.children[0].to_string()
        right = self.children[1].to_string()
        return f"({left} {self.val} {right})"


class GeneticProgramming:
    """
    Genetic Programming for symbolic regression.
    
    Evolves mathematical expressions (represented as trees) to fit data.
    Uses genetic operators (crossover, mutation) to explore the space of
    possible expressions.
    
    Attributes:
        pop_size (int): Population size.
        generations (int): Number of generations.
        max_depth (int): Maximum tree depth.
        cx_rate (float): Crossover probability.
        mut_rate (float): Mutation probability.
        funcs (List[str]): Available function operators.
        terms (List[str]): Terminal symbols.
        parsimony (float): Penalty for tree size (bloat control).
    """
    
    def __init__(
        self,
        population_size: int = 200,
        generations: int = 50,
        max_depth: int = 6,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        function_set: List[str] = ["+", "-", "*", "/"],
        terminal_set: List[str] = ["x", "constants"],
        parsimony_coefficient: float = 0.001
    ):
        self.pop_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.funcs = function_set
        self.terms = terminal_set
        self.parsimony = parsimony_coefficient
        self.population: List[GPNode] = []
        self._best_program: Optional[GPNode] = None
        self._best_error: float = float('inf')
        self._history: List[float] = []

    def _random_tree(self, depth: int, method: str = "full") -> GPNode:
        """Generate a random expression tree."""
        if depth == 0 or (method == "grow" and random.random() < 0.5):
            # Create terminal
            term = random.choice(self.terms)
            if term == "constants":
                val = random.uniform(-5, 5)
            else:
                val = 'x'
            return GPNode(val)
        else:
            # Create function node
            func = random.choice(self.funcs)
            arity = 1 if func in ['sin', 'cos', 'exp'] else 2
            children = [self._random_tree(depth - 1, method) for _ in range(arity)]
            return GPNode(func, children)

    def fit(self, X: np.ndarray, y: np.ndarray) -> GPNode:
        """
        Evolve a mathematical expression to fit the data.
        
        Args:
            X: Input values of shape (n_samples,).
            y: Target values of shape (n_samples,).
        
        Returns:
            GPNode: The best evolved expression tree.
        """
        X = np.asarray(X).flatten()
        y = np.asarray(y).flatten()
        
        # Initialize population using ramped half-and-half
        self.population = []
        for _ in range(self.pop_size):
            depth = random.randint(2, self.max_depth)
            method = "full" if random.random() < 0.5 else "grow"
            self.population.append(self._random_tree(depth, method))

        self._best_program = None
        self._best_error = float('inf')
        self._history = []

        for gen in range(self.generations):
            scores = []
            
            for individual in self.population:
                # Calculate MSE
                try:
                    preds = np.array([individual.evaluate(xi) for xi in X])
                    mse = np.mean((y - preds) ** 2)
                    if np.isnan(mse) or np.isinf(mse):
                        mse = float('inf')
                except:
                    mse = float('inf')
                
                # Add parsimony pressure (penalize large trees)
                fitness = mse + self.parsimony * individual.size()
                scores.append(fitness)

                if mse < self._best_error:
                    self._best_error = mse
                    self._best_program = copy.deepcopy(individual)
            
            self._history.append(self._best_error)

            # Create new population
            new_pop = [copy.deepcopy(self._best_program)]  # Elitism
            
            while len(new_pop) < self.pop_size:
                parent1 = self._tournament(scores)
                parent2 = self._tournament(scores)
                
                if random.random() < self.cx_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)
                
                if random.random() < self.mut_rate:
                    child = self._mutate(child)
                
                # Depth limit
                if child.depth() <= self.max_depth:
                    new_pop.append(child)
                else:
                    new_pop.append(copy.deepcopy(parent1))
            
            self.population = new_pop

        return self._best_program

    def _tournament(self, scores: List[float], size: int = 3) -> GPNode:
        """Tournament selection."""
        indices = np.random.choice(len(self.population), size, replace=False)
        best_idx = indices[np.argmin([scores[i] for i in indices])]
        return self.population[best_idx]

    def _crossover(self, p1: GPNode, p2: GPNode) -> GPNode:
        """Subtree crossover."""
        child = copy.deepcopy(p1)
        nodes = self._get_nodes(child)
        target = random.choice(nodes)
        source = random.choice(self._get_nodes(p2))
        target.val = source.val
        target.children = copy.deepcopy(source.children)
        target.is_terminal = source.is_terminal
        return child

    def _mutate(self, p: GPNode) -> GPNode:
        """Subtree mutation."""
        child = copy.deepcopy(p)
        nodes = self._get_nodes(child)
        target = random.choice(nodes)
        new_subtree = self._random_tree(2, "grow")
        target.val = new_subtree.val
        target.children = new_subtree.children
        target.is_terminal = new_subtree.is_terminal
        return child

    def _get_nodes(self, node: GPNode) -> List[GPNode]:
        """Get all nodes in tree."""
        nodes = [node]
        for c in node.children:
            nodes.extend(self._get_nodes(c))
        return nodes
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best evolved expression."""
        if self._best_program is None:
            raise RuntimeError("Model must be fitted first.")
        X = np.asarray(X).flatten()
        return np.array([self._best_program.evaluate(xi) for xi in X])


@tool
def train_gp_tool(
    X_train: List[float] = Field(description="Training input values as a 1D list (for single-variable symbolic regression)"),
    y_train: List[float] = Field(description="Training target values as a 1D list"),
    population_size: int = Field(default=200, ge=100, le=1000, description="Number of individuals in population"),
    generations: int = Field(default=50, ge=20, le=200, description="Number of evolutionary generations"),
    max_depth: int = Field(default=6, ge=3, le=10, description="Maximum depth of expression trees"),
    crossover_rate: float = Field(default=0.9, ge=0.7, le=0.95, description="Probability of crossover"),
    mutation_rate: float = Field(default=0.1, ge=0.05, le=0.2, description="Probability of mutation"),
    function_set: List[str] = Field(default=["+", "-", "*", "/"], description="Available operators: +, -, *, /, sin, cos, exp"),
    terminal_set: List[str] = Field(default=["x", "constants"], description="Terminal symbols: 'x' for variable, 'constants' for random constants"),
    parsimony_coefficient: float = Field(default=0.001, ge=0, le=0.01, description="Penalty for tree size (bloat control)")
) -> Dict[str, Any]:
    """
    Train Genetic Programming for symbolic regression.
    
    Automatically discovers mathematical expressions that fit the data.
    GP evolves tree-structured programs representing formulas like:
    ((x * 2.5) + sin(x)) or (x^2 - 3.14)
    
    **When to use:**
    - Discovering mathematical relationships in data
    - When you need interpretable formulas, not black-box models
    - Scientific modeling and equation discovery
    - Feature engineering (discovering useful transformations)
    
    **Function set options:**
    - Basic: ["+", "-", "*", "/"]
    - With trig: ["+", "-", "*", "/", "sin", "cos"]
    - With exp: ["+", "-", "*", "/", "exp"]
    
    **Parameter tuning:**
    - population_size: Larger = more exploration, slower
    - generations: More generations = better solutions, diminishing returns
    - max_depth: Deeper trees = more complex expressions
    - parsimony_coefficient: Higher = simpler expressions (bloat control)
    
    Args:
        X_train: Input values (1D list for single-variable regression).
        y_train: Target values to fit.
        population_size: Population size (100-1000). Default: 200.
        generations: Evolution generations (20-200). Default: 50.
        max_depth: Max tree depth (3-10). Default: 6.
        crossover_rate: Crossover probability (0.7-0.95). Default: 0.9.
        mutation_rate: Mutation probability (0.05-0.2). Default: 0.1.
        function_set: Available operators. Default: ["+", "-", "*", "/"].
        terminal_set: Terminal symbols. Default: ["x", "constants"].
        parsimony_coefficient: Size penalty (0-0.01). Default: 0.001.
    
    Returns:
        Dict containing:
            - model_id (str): Unique identifier
            - status (str): "success" or "error"
            - expression (str): Human-readable evolved formula
            - mse (float): Mean squared error on training data
            - tree_size (int): Number of nodes in expression
            - tree_depth (int): Depth of expression tree
    
    Example:
        >>> # Discover formula for y = x^2 + 2*x + 1
        >>> X = [i/10 for i in range(-50, 51)]
        >>> y = [x**2 + 2*x + 1 for x in X]
        >>> result = train_gp_tool(X_train=X, y_train=y, generations=100)
        >>> print(result['expression'])  # Something like: ((x * x) + ((x * 2.0) + 1.0))
    """
    try:
        X = np.array(X_train)
        y = np.array(y_train)
        
        if len(X.shape) != 1:
            return {"status": "error", "message": "X_train must be a 1D array"}
        if len(y.shape) != 1:
            return {"status": "error", "message": "y_train must be a 1D array"}
        if len(X) != len(y):
            return {"status": "error", "message": "X_train and y_train must have same length"}
        
        model = GeneticProgramming(
            population_size=population_size,
            generations=generations,
            max_depth=max_depth,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            function_set=function_set,
            terminal_set=terminal_set,
            parsimony_coefficient=parsimony_coefficient
        )
        
        best_tree = model.fit(X, y)
        
        model_id = f"gp_{uuid.uuid4().hex[:8]}"
        MODEL_STORE[model_id] = model
        
        # Calculate training metrics
        train_pred = model.predict(X)
        train_metrics = calculate_regression_metrics(y, train_pred)
        
        return {
            "status": "success",
            "message": "Genetic Programming completed successfully",
            "model_id": model_id,
            "expression": best_tree.to_string(),
            "mse": float(model._best_error),
            "tree_size": best_tree.size(),
            "tree_depth": best_tree.depth(),
            "n_samples": len(X),
            "training_r2": train_metrics["r2_score"],
            "convergence_history": model._history[-10:]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@tool
def inference_gp_tool(
    model_id: str = Field(description="The unique model ID returned from train_gp_tool"),
    X_test: List[float] = Field(description="Test input values as a 1D list"),
    y_true: Optional[List[float]] = Field(default=None, description="Optional ground truth values for computing regression metrics")
) -> Dict[str, Any]:
    """
    Make predictions using an evolved GP expression.
    
    Evaluates the discovered mathematical expression on new input values.
    
    Args:
        model_id: Unique identifier from train_gp_tool.
        X_test: Input values to evaluate (1D list).
    
    Returns:
        Dict containing:
            - status (str): "success" or "error"
            - predictions (List[float]): Predicted output values
            - expression (str): The formula being evaluated
            - n_samples (int): Number of predictions
            - metrics (Dict[str, Any]): when y_true is provided:
                - MSE, RMSE, MAE, R², MAPE
    
    Example:
        >>> result = inference_gp_tool(
        ...     model_id="gp_abc12345",
        ...     X_test=[0, 1, 2, 3, 4, 5]
        ... )
        >>> print(result['predictions'])
    """
    try:
        if model_id not in MODEL_STORE:
            return {"status": "error", "message": f"Model '{model_id}' not found."}
        
        model = MODEL_STORE[model_id]
        X = np.array(X_test)
        
        if len(X.shape) != 1:
            return {"status": "error", "message": "X_test must be a 1D array"}
        
        predictions = model.predict(X)
        
        result = {
            "status": "success",
            "message": f"Successfully predicted {len(predictions)} samples",
            "predictions": predictions.tolist(),
            "expression": model._best_program.to_string(),
            "n_samples": len(predictions)
        }
        
        # Calculate metrics if ground truth is provided
        if y_true is not None:
            y_true_arr = np.array(y_true)
            if len(y_true_arr) != len(predictions):
                return {"status": "error", "message": "y_true length must match X_test samples"}
            
            metrics = calculate_regression_metrics(y_true_arr, predictions)
            result["metrics"] = metrics
            result["message"] = f"Successfully predicted {len(predictions)} samples with R²={metrics['r2_score']:.3f}"
        
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}