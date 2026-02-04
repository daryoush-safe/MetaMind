import numpy as np
from typing import Tuple, Literal
from pydantic import Field
from langchain_core.tools import tool


class SOM:
    def __init__(self, map_size=(10, 10), learning_rate_initial=0.5, learning_rate_final=0.01,
                 neighborhood_initial=5.0, max_epochs=1000, topology="rectangular"):
        self.map_size = map_size
        self.lr_initial = learning_rate_initial
        self.lr_final = learning_rate_final
        self.radius_initial = neighborhood_initial
        self.max_epochs = max_epochs
        self.topology = topology
        self.weights = None
        self._coords = self._generate_coords()

    def _generate_coords(self):
        """Pre-calculate coordinates for neurons based on topology."""
        rows, cols = self.map_size
        coords = np.zeros((rows, cols, 2))
        for r in range(rows):
            for c in range(cols):
                if self.topology == "hexagonal":
                    # Offset every other row
                    x = c + (0.5 if r % 2 != 0 else 0)
                    y = r * (np.sqrt(3) / 2)
                    coords[r, c] = [x, y]
                else: # rectangular
                    coords[r, c] = [r, c]
        return coords

    def fit(self, data):
        n_features = data.shape[1]
        rows, cols = self.map_size
        
        # Initialize weights randomly
        self.weights = np.random.random((rows, cols, n_features))
        
        # Time constant for decay
        time_constant = self.max_epochs / np.log(self.radius_initial)

        for epoch in range(self.max_epochs):
            # Decay learning rate and radius
            lr = self.lr_initial * (self.lr_final / self.lr_initial) ** (epoch / self.max_epochs)
            radius = self.radius_initial * np.exp(-epoch / time_constant)
            radius_sq = radius ** 2
            
            # Pick random sample
            idx = np.random.randint(0, data.shape[0])
            sample = data[idx]
            
            # Find Best Matching Unit (BMU)
            # Flatten weights to calculate distance efficiently
            flat_w = self.weights.reshape(-1, n_features)
            dists = np.linalg.norm(flat_w - sample, axis=1)
            bmu_idx_flat = np.argmin(dists)
            bmu_idx = np.unravel_index(bmu_idx_flat, (rows, cols))
            
            # Update weights of BMU and neighbors
            bmu_loc = self._coords[bmu_idx]
            
            # Calculate distance of all neurons to BMU in grid space
            dist_to_bmu_sq = np.sum((self._coords - bmu_loc) ** 2, axis=2)
            
            # Gaussian neighborhood function
            influence = np.exp(-dist_to_bmu_sq / (2 * radius_sq))
            
            # Mask neurons outside radius (optimization)
            # Note: Gaussian technically infinite, but we can cut off small values
            mask = influence > 0.001
            
            # Update rule: W += lr * influence * (X - W)
            # Expand dimensions for broadcasting
            influence_expanded = influence[..., np.newaxis]
            self.weights[mask] += lr * influence_expanded[mask] * (sample - self.weights[mask])

    def get_bmu(self, sample):
        """Returns the (row, col) of the Best Matching Unit."""
        flat_w = self.weights.reshape(-1, self.weights.shape[-1])
        dists = np.linalg.norm(flat_w - sample, axis=1)
        return np.unravel_index(np.argmin(dists), self.map_size)


@tool
def som_tool(
    map_size: Tuple[int, int] = Field(default=(10, 10)),
    learning_rate_initial: float = Field(default=0.5, ge=0.1, le=1.0),
    learning_rate_final: float = Field(default=0.01),
    neighborhood_initial: float = Field(default=5.0),
    max_epochs: int = Field(default=1000, ge=500, le=5000),
    topology: Literal["rectangular", "hexagonal"] = Field(default="rectangular"),
):
    """Creates a Self-Organizing Map (SOM) tool with specified hyperparameters."""
    pass