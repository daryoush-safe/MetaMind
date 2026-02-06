import numpy as np
import uuid
from typing import List, Tuple, Optional, Any, Dict, Literal
from pydantic import Field, BaseModel
from langchain_core.tools import tool

# In-memory model storage
MODEL_STORE: Dict[str, Any] = {}


def calculate_clustering_metrics(X: np.ndarray, labels: np.ndarray, 
                                  y_true: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive clustering quality metrics.
    
    Internal metrics (no ground truth needed):
        - Silhouette Score: Cohesion vs separation (-1 to 1, higher is better)
        - Davies-Bouldin Index: Cluster similarity (lower is better)
        - Calinski-Harabasz Index: Variance ratio (higher is better)
        - Inertia: Within-cluster sum of squares (lower is better)
    
    External metrics (when true labels available):
        - Adjusted Rand Index: Agreement with true labels (0 to 1)
        - Normalized Mutual Information: Information shared with true labels (0 to 1)
        - Purity: Fraction of correctly assigned samples
    
    Args:
        X: Data points of shape (n_samples, n_features)
        labels: Cluster assignments
        y_true: Optional ground truth labels for external validation
    
    Returns:
        Dictionary with comprehensive clustering metrics
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = len(X)
    
    cluster_sizes = {int(label): int(np.sum(labels == label)) for label in unique_labels}
    
    # Pre-compute centroids and cluster data
    centroids = []
    cluster_points_list = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        cluster_points_list.append(cluster_points)
        centroids.append(np.mean(cluster_points, axis=0))
    centroids = np.array(centroids)
    global_centroid = np.mean(X, axis=0)
    
    # --- Inertia: Within-cluster sum of squares ---
    inertia = 0.0
    intra_distances = []
    for i, label in enumerate(unique_labels):
        cluster_points = cluster_points_list[i]
        dists_sq = np.sum((cluster_points - centroids[i]) ** 2, axis=1)
        inertia += np.sum(dists_sq)
        if len(cluster_points) > 1:
            intra_distances.append(np.mean(np.sqrt(dists_sq)))
    
    avg_intra_distance = float(np.mean(intra_distances)) if intra_distances else 0.0
    
    # Inter-cluster distances
    inter_distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            inter_distances.append(np.linalg.norm(centroids[i] - centroids[j]))
    avg_inter_distance = float(np.mean(inter_distances)) if inter_distances else 0.0
    
    # --- Davies-Bouldin Index (lower is better) ---
    db_index = 0.0
    if n_clusters > 1:
        for i, label_i in enumerate(unique_labels):
            max_ratio = 0.0
            si = np.mean(np.linalg.norm(cluster_points_list[i] - centroids[i], axis=1)) if len(cluster_points_list[i]) > 0 else 0
            
            for j, label_j in enumerate(unique_labels):
                if i != j:
                    sj = np.mean(np.linalg.norm(cluster_points_list[j] - centroids[j], axis=1)) if len(cluster_points_list[j]) > 0 else 0
                    dij = np.linalg.norm(centroids[i] - centroids[j])
                    if dij > 0:
                        ratio = (si + sj) / dij
                        max_ratio = max(max_ratio, ratio)
            db_index += max_ratio
        db_index /= n_clusters
    
    # --- Calinski-Harabasz Index (higher is better) ---
    ch_index = 0.0
    if n_clusters > 1 and n_clusters < n_samples:
        # Between-cluster dispersion
        bgss = 0.0
        for i, label in enumerate(unique_labels):
            n_k = len(cluster_points_list[i])
            bgss += n_k * np.sum((centroids[i] - global_centroid) ** 2)
        
        # Within-cluster dispersion
        wgss = inertia
        
        if wgss > 0:
            ch_index = (bgss / (n_clusters - 1)) / (wgss / (n_samples - n_clusters))
    
    # --- Silhouette Score (-1 to 1, higher is better) ---
    silhouette_score = 0.0
    if n_clusters > 1 and n_clusters < n_samples:
        silhouette_values = np.zeros(n_samples)
        
        for idx in range(n_samples):
            own_label = labels[idx]
            own_cluster_mask = labels == own_label
            own_cluster_size = np.sum(own_cluster_mask)
            
            # a(i): mean intra-cluster distance
            if own_cluster_size > 1:
                own_cluster_points = X[own_cluster_mask]
                a_i = np.mean(np.linalg.norm(own_cluster_points - X[idx], axis=1))
            else:
                a_i = 0.0
            
            # b(i): min mean distance to other clusters
            b_i = float('inf')
            for label in unique_labels:
                if label == own_label:
                    continue
                other_points = X[labels == label]
                mean_dist = np.mean(np.linalg.norm(other_points - X[idx], axis=1))
                b_i = min(b_i, mean_dist)
            
            if b_i == float('inf'):
                b_i = 0.0
            
            max_ab = max(a_i, b_i)
            silhouette_values[idx] = (b_i - a_i) / max_ab if max_ab > 0 else 0.0
        
        silhouette_score = float(np.mean(silhouette_values))
    
    result = {
        "n_clusters": n_clusters,
        "n_samples": n_samples,
        "cluster_sizes": cluster_sizes,
        "silhouette_score": silhouette_score,
        "davies_bouldin_index": float(db_index),
        "calinski_harabasz_index": float(ch_index),
        "inertia": float(inertia),
        "avg_intra_cluster_distance": avg_intra_distance,
        "avg_inter_cluster_distance": avg_inter_distance,
        "compactness_ratio": float(avg_intra_distance / avg_inter_distance) if avg_inter_distance > 0 else float('inf')
    }
    
    # --- External validation metrics (when ground truth available) ---
    if y_true is not None:
        y_true = np.asarray(y_true).flatten()
        
        # Purity
        total_correct = 0
        for i, label in enumerate(unique_labels):
            cluster_mask = labels == label
            if np.sum(cluster_mask) > 0:
                cluster_true = y_true[cluster_mask]
                most_common = np.bincount(cluster_true.astype(int)).argmax()
                total_correct += np.sum(cluster_true == most_common)
        purity = total_correct / n_samples
        
        # --- Adjusted Rand Index ---
        ari = _compute_adjusted_rand_index(y_true.astype(int), labels.astype(int))
        
        # --- Normalized Mutual Information ---
        nmi = _compute_nmi(y_true.astype(int), labels.astype(int))
        
        result["purity"] = float(purity)
        result["adjusted_rand_index"] = float(ari)
        result["normalized_mutual_information"] = float(nmi)
        result["external_validation_available"] = True
    else:
        result["external_validation_available"] = False
    
    return result


def _compute_adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index (ARI).
    Measures agreement between two clusterings, adjusted for chance.
    Range: -0.5 to 1.0 (1.0 = perfect agreement, 0 = random)
    """
    n = len(labels_true)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    
    # Contingency table
    contingency = np.zeros((len(classes), len(clusters)), dtype=int)
    class_map = {c: i for i, c in enumerate(classes)}
    cluster_map = {c: i for i, c in enumerate(clusters)}
    
    for i in range(n):
        contingency[class_map[labels_true[i]], cluster_map[labels_pred[i]]] += 1
    
    # Row and column sums
    a = contingency.sum(axis=1)
    b = contingency.sum(axis=0)
    
    # Combinations
    def comb2(x):
        return x * (x - 1) / 2
    
    sum_comb_nij = np.sum(comb2(contingency))
    sum_comb_a = np.sum(comb2(a))
    sum_comb_b = np.sum(comb2(b))
    comb_n = comb2(n)
    
    expected = sum_comb_a * sum_comb_b / comb_n if comb_n > 0 else 0
    max_index = (sum_comb_a + sum_comb_b) / 2
    
    denominator = max_index - expected
    if denominator == 0:
        return 0.0 if max_index == 0 else 1.0
    
    return float((sum_comb_nij - expected) / denominator)


def _compute_nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Normalized Mutual Information (NMI).
    Measures information shared between two clusterings.
    Range: 0 to 1 (1.0 = perfect agreement)
    """
    n = len(labels_true)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    
    # Contingency table
    contingency = np.zeros((len(classes), len(clusters)), dtype=float)
    class_map = {c: i for i, c in enumerate(classes)}
    cluster_map = {c: i for i, c in enumerate(clusters)}
    
    for i in range(n):
        contingency[class_map[labels_true[i]], cluster_map[labels_pred[i]]] += 1
    
    # Marginals
    p_true = contingency.sum(axis=1) / n
    p_pred = contingency.sum(axis=0) / n
    p_joint = contingency / n
    
    # Entropy
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    
    h_true = entropy(p_true)
    h_pred = entropy(p_pred)
    
    # Mutual information
    mi = 0.0
    for i in range(len(classes)):
        for j in range(len(clusters)):
            if p_joint[i, j] > 0:
                mi += p_joint[i, j] * np.log(p_joint[i, j] / (p_true[i] * p_pred[j]))
    
    # Normalize
    denominator = (h_true + h_pred) / 2
    if denominator == 0:
        return 0.0
    
    return float(mi / denominator)


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
    
    def topographic_error(self, data: np.ndarray) -> float:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        errors = 0
        for sample in data:
            flat_w = self.weights.reshape(-1, self.weights.shape[-1])
            dists = np.linalg.norm(flat_w - sample, axis=1)
            sorted_indices = np.argsort(dists)
            bmu1 = np.unravel_index(sorted_indices[0], self.map_size)
            bmu2 = np.unravel_index(sorted_indices[1], self.map_size)
            if abs(bmu1[0] - bmu2[0]) + abs(bmu1[1] - bmu2[1]) > 1:
                errors += 1
        return errors / len(data)


class TrainSOMInput(BaseModel):
    X_train: List[List[float]] = Field(description="Training data matrix as a 2D list of shape (n_samples, n_features)")
    map_size: Tuple[int, int] = Field(default=(10, 10), description="Grid dimensions as (rows, cols)")
    learning_rate_initial: float = Field(default=0.5, ge=0.1, le=1.0, description="Initial learning rate")
    learning_rate_final: float = Field(default=0.01, ge=0.001, le=0.1, description="Final learning rate after decay")
    neighborhood_initial: float = Field(default=5.0, ge=1.0, le=20.0, description="Initial neighborhood radius in grid units")
    max_epochs: int = Field(default=1000, ge=500, le=5000, description="Number of training iterations")
    topology: Literal["rectangular", "hexagonal"] = Field(default="rectangular", description="Grid topology")


class InferenceSOMInput(BaseModel):
    model_id: str = Field(description="The unique model ID returned from train_som_tool"),
    X_test: List[List[float]] = Field(description="Test data matrix as a 2D list of shape (n_samples, n_features)"),
    y_true: Optional[List[int]] = Field(default=None, description="Optional ground truth cluster labels for external validation metrics")


@tool(args_schema=TrainSOMInput)
def train_som_tool(
    X_train: List[List[float]],
    map_size: Tuple[int, int] = (10, 10),
    learning_rate_initial: float = 0.5,
    learning_rate_final: float = 0.01,
    neighborhood_initial: float = 5.0,
    max_epochs: int = 1000,
    topology: Literal["rectangular", "hexagonal"] = "rectangular"
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
    """
    try:
        X = np.array(X_train)
        
        if len(X.shape) != 2:
            return {"status": "error", "message": "X_train must be a 2D array"}
        if X.shape[0] < map_size[0] * map_size[1]:
            return {"status": "error", "message": f"Not enough samples ({X.shape[0]}) for map size {map_size}"}
        
        model = SOM(
            map_size=map_size,
            learning_rate_initial=learning_rate_initial,
            learning_rate_final=learning_rate_final,
            neighborhood_initial=neighborhood_initial,
            max_epochs=max_epochs,
            topology=topology
        )
        model.fit(X)
        
        model_id = f"som_{uuid.uuid4().hex[:8]}"
        MODEL_STORE[model_id] = model
        
        q_error = model.quantization_error(X)
        t_error = model.topographic_error(X)
        
        return {
            "status": "success",
            "message": f"SOM trained successfully with {map_size[0]}x{map_size[1]} = {map_size[0]*map_size[1]} neurons",
            "model_id": model_id,
            "map_size": map_size,
            "n_neurons": map_size[0] * map_size[1],
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "topology": topology,
            "quantization_error": float(q_error),
            "topographic_error": float(t_error)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool  
def inference_som_tool(
    model_id: str,
    X_test: List[List[float]],
    y_true: Optional[List[int]] = None,
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
    
    Returns:
        Dict containing:
            - status: "success" or "error"
            - message: Status message  
            - bmu_indices: BMU (row, col) for each sample
            - distances: Distance from each sample to its BMU
            - n_samples: Number of samples processed
            - metrics: Metrics when y_true is provided
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
        
        bmu_indices = []
        distances = []
        cluster_labels = []
        
        for sample in X:
            bmu = model.get_bmu(sample)
            bmu_indices.append(tuple(map(int, bmu)))
            distances.append(float(np.linalg.norm(sample - model.weights[bmu])))
            cluster_labels.append(bmu[0] * model.map_size[1] + bmu[1])
        
        result = {
            "status": "success",
            "message": f"Found BMUs for {len(bmu_indices)} samples",
            "bmu_indices": bmu_indices,
            "distances": distances,
            "cluster_labels": cluster_labels,
            "n_samples": len(bmu_indices)
        }
        
        # Calculate comprehensive clustering metrics
        cluster_arr = np.array(cluster_labels)
        y_true_arr = np.array(y_true) if y_true is not None else None
        metrics = calculate_clustering_metrics(X, cluster_arr, y_true_arr)
        result["metrics"] = metrics
        
        msg = f"Found BMUs for {len(bmu_indices)} samples. Silhouette: {metrics['silhouette_score']:.3f}, DB: {metrics['davies_bouldin_index']:.3f}"
        if y_true is not None:
            msg += f", ARI: {metrics['adjusted_rand_index']:.3f}, NMI: {metrics['normalized_mutual_information']:.3f}"
        result["message"] = msg
        
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}