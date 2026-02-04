import numpy as np
import uuid
from typing import List, Tuple, Optional, Any, Dict, Literal
from pydantic import Field
from langchain_core.tools import tool

# In-memory model storage
MODEL_STORE: Dict[str, Any] = {}


class SOM:
    """
    Kohonen Self-Organizing Map for clustering and visualization.
    
    A SOM is an unsupervised neural network that maps high-dimensional input
    data onto a low-dimensional (typically 2D) grid of neurons while preserving
    the topological relationships in the data. This makes it useful for:
    - Clustering similar data points
    - Visualizing high-dimensional data
    - Dimensionality reduction
    - Anomaly detection
    
    The network uses competitive learning where neurons compete to be activated,
    and the winner (Best Matching Unit) along with its neighbors are updated
    to become more similar to the input.
    
    Attributes:
        map_size (Tuple[int, int]): Grid dimensions (rows, cols).
        lr_initial (float): Initial learning rate.
        lr_final (float): Final learning rate after decay.
        radius_initial (float): Initial neighborhood radius.
        max_epochs (int): Number of training iterations.
        topology (str): Grid topology ("rectangular" or "hexagonal").
        weights (np.ndarray): Neuron weight vectors of shape (rows, cols, features).
    
    Example:
        >>> som = SOM(map_size=(10, 10), max_epochs=1000)
        >>> som.fit(data)
        >>> bmu = som.get_bmu(sample)  # Get cluster assignment
    """
    
    def __init__(
        self,
        map_size: Tuple[int, int] = (10, 10),
        learning_rate_initial: float = 0.5,
        learning_rate_final: float = 0.01,
        neighborhood_initial: float = 5.0,
        max_epochs: int = 1000,
        topology: str = "rectangular"
    ):
        """
        Initialize the Self-Organizing Map.
        
        Args:
            map_size: Grid dimensions as (rows, cols). Larger maps can capture
                more detail but require more data. Range: (5,5) to (50,50).
                Default: (10, 10).
            learning_rate_initial: Starting learning rate. Higher values lead
                to faster initial adaptation. Range: 0.1-1.0. Default: 0.5.
            learning_rate_final: Final learning rate after decay. Should be
                small for fine-tuning. Default: 0.01.
            neighborhood_initial: Initial neighborhood radius in grid units.
                Larger values cause more neurons to update initially.
                Default: 5.0.
            max_epochs: Number of training iterations. Each epoch presents
                one random sample. Range: 500-5000. Default: 1000.
            topology: Grid topology affecting neighbor distances:
                - "rectangular": Standard 4-connected grid
                - "hexagonal": 6-connected honeycomb pattern
                Default: "rectangular".
        """
        self.map_size = map_size
        self.lr_initial = learning_rate_initial
        self.lr_final = learning_rate_final
        self.radius_initial = neighborhood_initial
        self.max_epochs = max_epochs
        self.topology = topology
        self.weights: Optional[np.ndarray] = None
        self._coords = self._generate_coords()
        self._is_fitted: bool = False
        self._n_features: int = 0

    def _generate_coords(self) -> np.ndarray:
        """Pre-calculate 2D coordinates for neurons based on topology."""
        rows, cols = self.map_size
        coords = np.zeros((rows, cols, 2))
        
        for r in range(rows):
            for c in range(cols):
                if self.topology == "hexagonal":
                    # Offset every other row for hexagonal packing
                    x = c + (0.5 if r % 2 != 0 else 0)
                    y = r * (np.sqrt(3) / 2)
                    coords[r, c] = [x, y]
                else:  # rectangular
                    coords[r, c] = [r, c]
        
        return coords

    def fit(self, data: np.ndarray) -> 'SOM':
        """
        Train the SOM on the provided data.
        
        Uses online learning: each epoch presents one random sample and updates
        the BMU and its neighbors using a Gaussian neighborhood function.
        
        Args:
            data: Training data of shape (n_samples, n_features).
        
        Returns:
            SOM: The fitted model instance (self).
        """
        data = np.asarray(data)
        n_features = data.shape[1]
        self._n_features = n_features
        rows, cols = self.map_size
        
        # Initialize weights randomly from data range
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        self.weights = data_min + np.random.random((rows, cols, n_features)) * (data_max - data_min)
        
        # Time constant for exponential decay
        time_constant = self.max_epochs / np.log(max(self.radius_initial, 1))

        for epoch in range(self.max_epochs):
            # Decay learning rate exponentially
            lr = self.lr_initial * (self.lr_final / self.lr_initial) ** (epoch / self.max_epochs)
            
            # Decay neighborhood radius exponentially
            radius = self.radius_initial * np.exp(-epoch / time_constant)
            radius_sq = radius ** 2
            
            # Pick random sample
            idx = np.random.randint(0, data.shape[0])
            sample = data[idx]
            
            # Find Best Matching Unit (BMU)
            flat_w = self.weights.reshape(-1, n_features)
            dists = np.linalg.norm(flat_w - sample, axis=1)
            bmu_idx_flat = np.argmin(dists)
            bmu_idx = np.unravel_index(bmu_idx_flat, (rows, cols))
            
            # Get BMU location in grid space
            bmu_loc = self._coords[bmu_idx]
            
            # Calculate distance of all neurons to BMU in grid space
            dist_to_bmu_sq = np.sum((self._coords - bmu_loc) ** 2, axis=2)
            
            # Gaussian neighborhood function
            influence = np.exp(-dist_to_bmu_sq / (2 * radius_sq + 1e-10))
            
            # Mask neurons with negligible influence (optimization)
            mask = influence > 0.001
            
            # Update weights: W += lr * influence * (X - W)
            influence_expanded = influence[..., np.newaxis]
            delta = sample - self.weights
            self.weights[mask] += lr * influence_expanded[mask] * delta[mask]
        
        self._is_fitted = True
        return self

    def get_bmu(self, sample: np.ndarray) -> Tuple[int, int]:
        """
        Find the Best Matching Unit for a sample.
        
        Args:
            sample: Feature vector of shape (n_features,).
        
        Returns:
            Tuple[int, int]: (row, col) indices of the BMU in the grid.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before finding BMU.")
        
        flat_w = self.weights.reshape(-1, self.weights.shape[-1])
        dists = np.linalg.norm(flat_w - sample, axis=1)
        return np.unravel_index(np.argmin(dists), self.map_size)
    
    def quantization_error(self, data: np.ndarray) -> float:
        """
        Calculate the average quantization error.
        
        The quantization error is the average distance from each data point
        to its BMU, measuring how well the SOM represents the data.
        
        Args:
            data: Data matrix of shape (n_samples, n_features).
        
        Returns:
            float: Average quantization error (lower is better).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        
        total_error = 0.0
        for sample in data:
            bmu_idx = self.get_bmu(sample)
            total_error += np.linalg.norm(sample - self.weights[bmu_idx])
        
        return total_error / len(data)


@tool
def train_som_tool(
    X_train: List[List[float]] = Field(description="Training data matrix as a 2D list of shape (n_samples, n_features)"),
    map_size: Tuple[int, int] = Field(default=(10, 10), description="Grid dimensions as (rows, cols). Larger maps capture more detail"),
    learning_rate_initial: float = Field(default=0.5, ge=0.1, le=1.0, description="Initial learning rate. Higher values mean faster initial adaptation"),
    learning_rate_final: float = Field(default=0.01, ge=0.001, le=0.1, description="Final learning rate after decay"),
    neighborhood_initial: float = Field(default=5.0, ge=1.0, le=20.0, description="Initial neighborhood radius in grid units"),
    max_epochs: int = Field(default=1000, ge=500, le=5000, description="Number of training iterations"),
    topology: Literal["rectangular", "hexagonal"] = Field(default="rectangular", description="Grid topology: 'rectangular' or 'hexagonal'")
) -> Dict[str, Any]:
    """
    Train a Kohonen Self-Organizing Map (SOM) for clustering and visualization.
    
    SOMs are unsupervised neural networks that project high-dimensional data onto
    a 2D grid while preserving topological relationships. Neurons that are close
    in the grid represent similar data patterns.
    
    **When to use:**
    - Clustering when you don't know the number of clusters
    - Visualizing high-dimensional data on a 2D map
    - Finding natural groupings in unlabeled data
    - Anomaly detection (samples with distant BMUs are unusual)
    - Data exploration and understanding
    
    **Map size recommendations:**
    - Small datasets (< 500 samples): (10, 10) or smaller
    - Medium datasets (500-5000): (15, 15) to (20, 20)
    - Large datasets (> 5000): (25, 25) to (50, 50)
    - Rule of thumb: 5 * sqrt(n_samples) neurons total
    
    **Topology choice:**
    - Rectangular: Standard grid, easier to interpret
    - Hexagonal: Better neighbor connectivity, smoother maps
    
    **Training tips:**
    - Normalize your data before training (zero mean, unit variance)
    - Use more epochs for larger maps
    - Start with larger neighborhood_initial for smoother organization
    
    Args:
        X_train: Training data as a 2D list (n_samples, n_features).
        map_size: Grid dimensions (rows, cols). Default: (10, 10).
        learning_rate_initial: Starting learning rate (0.1-1.0). Default: 0.5.
        learning_rate_final: Final learning rate (0.001-0.1). Default: 0.01.
        neighborhood_initial: Initial radius (1.0-20.0). Default: 5.0.
        max_epochs: Training iterations (500-5000). Default: 1000.
        topology: Grid topology. Default: "rectangular".
    
    Returns:
        Dict containing:
            - model_id (str): Unique identifier for the trained model
            - status (str): "success" or "error"
            - message (str): Status message
            - map_size (Tuple[int, int]): Grid dimensions
            - n_neurons (int): Total number of neurons
            - n_features (int): Number of input features
            - n_samples (int): Number of training samples
            - quantization_error (float): Average distance to BMUs
    
    Example:
        >>> # Cluster customer data
        >>> result = train_som_tool(
        ...     X_train=customer_features,
        ...     map_size=(15, 15),
        ...     topology="hexagonal",
        ...     max_epochs=2000
        ... )
        >>> print(f"Trained with quantization error: {result['quantization_error']:.4f}")
    """
    try:
        X = np.array(X_train)
        
        if len(X.shape) != 2:
            return {"status": "error", "message": "X_train must be a 2D array"}
        if X.shape[0] < map_size[0] * map_size[1]:
            return {
                "status": "error", 
                "message": f"Not enough samples ({X.shape[0]}) for map size {map_size}"
            }
        
        # Create and train model
        model = SOM(
            map_size=map_size,
            learning_rate_initial=learning_rate_initial,
            learning_rate_final=learning_rate_final,
            neighborhood_initial=neighborhood_initial,
            max_epochs=max_epochs,
            topology=topology
        )
        model.fit(X)
        
        # Store model
        model_id = f"som_{uuid.uuid4().hex[:8]}"
        MODEL_STORE[model_id] = model
        
        # Calculate quantization error
        q_error = model.quantization_error(X)
        
        return {
            "status": "success",
            "message": f"SOM trained successfully with {map_size[0]}x{map_size[1]} = {map_size[0]*map_size[1]} neurons",
            "model_id": model_id,
            "map_size": map_size,
            "n_neurons": map_size[0] * map_size[1],
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "topology": topology,
            "quantization_error": float(q_error)
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool  
def inference_som_tool(
    model_id: str = Field(description="The unique model ID returned from train_som_tool"),
    X_test: List[List[float]] = Field(description="Test data matrix as a 2D list of shape (n_samples, n_features)")
) -> Dict[str, Any]:
    """
    Find Best Matching Units (clusters) for new samples using a trained SOM.
    
    Maps each test sample to its closest neuron (BMU) in the SOM grid.
    Samples mapping to the same or nearby neurons are similar in the
    original feature space.
    
    **Interpreting results:**
    - BMU coordinates indicate cluster membership
    - Nearby BMUs (in grid space) represent similar data patterns
    - Samples with distant BMUs from training data may be anomalies
    
    **Usage:**
    1. Train a SOM using train_som_tool to get a model_id
    2. Use this tool to find cluster assignments for new data
    
    Args:
        model_id: Unique identifier from train_som_tool.
        X_test: Test data as a 2D list (n_samples, n_features).
    
    Returns:
        Dict containing:
            - status (str): "success" or "error"
            - message (str): Status message  
            - bmu_indices (List[Tuple[int, int]]): BMU (row, col) for each sample
            - distances (List[float]): Distance from each sample to its BMU
            - n_samples (int): Number of samples processed
    
    Example:
        >>> result = inference_som_tool(
        ...     model_id="som_abc12345",
        ...     X_test=[[age, income, spending], ...]
        ... )
        >>> for i, (bmu, dist) in enumerate(zip(result['bmu_indices'], result['distances'])):
        ...     print(f"Sample {i}: Cluster {bmu}, distance {dist:.3f}")
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
        
        bmu_indices = []
        distances = []
        
        for sample in X:
            bmu = model.get_bmu(sample)
            bmu_indices.append(tuple(map(int, bmu)))
            distances.append(float(np.linalg.norm(sample - model.weights[bmu])))
        
        return {
            "status": "success",
            "message": f"Found BMUs for {len(bmu_indices)} samples",
            "bmu_indices": bmu_indices,
            "distances": distances,
            "n_samples": len(bmu_indices)
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}