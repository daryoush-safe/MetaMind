import numpy as np
from pydantic import Field
from langchain_core.tools import tool


class HopfieldNetwork:
    def __init__(self, max_iterations=100, threshold=0.0, async_update=True, energy_threshold=1e-6):
        """
        Args:
            threshold (float): Firing threshold for neurons (default 0.0).
            async_update (bool): True for random sequential updates, False for simultaneous.
        """
        self.max_iter = max_iterations
        self.threshold = threshold
        self.async_update = async_update
        self.energy_tol = energy_threshold
        self.weights = None
        self.n_neurons = 0

    def train(self, patterns):
        """
        Patterns must be N vectors of shape (M,), containing binary values {-1, 1}.
        """
        n_patterns, self.n_neurons = patterns.shape
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        
        # Hebbian rule: W = sum(x_i * x_j) / N
        for p in patterns:
            self.weights += np.outer(p, p)
        
        self.weights /= self.n_neurons
        
        # Set diagonal to zero (no self-connection)
        np.fill_diagonal(self.weights, 0)

    def predict(self, pattern):
        state = pattern.copy()
        
        last_energy = self._energy(state)
        
        for _ in range(self.max_iter):
            if self.async_update:
                # Update neurons in random order
                indices = np.random.permutation(self.n_neurons)
                for idx in indices:
                    raw_output = np.dot(self.weights[idx], state) - self.threshold
                    state[idx] = 1 if raw_output >= 0 else -1
            else:
                # Synchronous update (all at once)
                raw_output = np.dot(self.weights, state) - self.threshold
                state = np.where(raw_output >= 0, 1, -1)
            
            # Check energy convergence
            current_energy = self._energy(state)
            if abs(last_energy - current_energy) < self.energy_tol:
                break
            last_energy = current_energy
            
        return state

    def _energy(self, state):
        # E = -0.5 * s^T * W * s + theta * sum(s)
        return -0.5 * np.dot(state.T, np.dot(self.weights, state)) + np.sum(state * self.threshold)    


@tool
def hopfield_tool(
    max_iterations: int = Field(default=100, ge=50, le=500),
    threshold: float = Field(default=0.0),
    async_update: bool = Field(default=True),
    energy_threshold: float = Field(default=1e-6),
):
    """Creates a Hopfield Network tool with specified hyperparameters."""
    pass