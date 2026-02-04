import numpy as np
import uuid
from typing import List, Optional, Any, Dict
from pydantic import Field
from langchain_core.tools import tool

# In-memory model storage
MODEL_STORE: Dict[str, Any] = {}


class HopfieldNetwork:
    """
    Hopfield Network for associative memory and pattern completion.
    
    A Hopfield Network is a recurrent neural network that can store a set of
    binary patterns and recall them when presented with partial or noisy versions.
    It acts as a content-addressable memory where stored patterns are attractors
    in the network's energy landscape.
    
    Key properties:
    - Stores binary patterns using Hebbian learning
    - Can retrieve stored patterns from corrupted/partial inputs
    - Uses energy-based dynamics to converge to stable states
    - Storage capacity: approximately 0.14 * n_neurons patterns
    
    Applications:
    - Pattern recognition and completion
    - Associative memory / content-addressable storage
    - Optimization problems (when formulated as energy minimization)
    - Error correction in binary codes
    
    Attributes:
        max_iter (int): Maximum iterations for convergence.
        threshold (float): Neuron firing threshold.
        async_update (bool): Whether to use asynchronous updates.
        energy_tol (float): Convergence threshold for energy change.
        weights (np.ndarray): Symmetric weight matrix.
        n_neurons (int): Number of neurons (pattern length).
    
    Example:
        >>> hopfield = HopfieldNetwork()
        >>> hopfield.train(stored_patterns)  # patterns with values {-1, 1}
        >>> recalled = hopfield.predict(noisy_pattern)
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        threshold: float = 0.0,
        async_update: bool = True,
        energy_threshold: float = 1e-6
    ):
        """
        Initialize the Hopfield Network.
        
        Args:
            max_iterations: Maximum iterations for state updates during recall.
                Network stops early if energy converges. Range: 50-500.
                Default: 100.
            threshold: Neuron firing threshold (theta in update rule).
                Neurons fire (output 1) if weighted input >= threshold.
                Default: 0.0.
            async_update: Update mode for neurons:
                - True: Update neurons one at a time in random order (more stable)
                - False: Update all neurons simultaneously (faster but may oscillate)
                Default: True (recommended).
            energy_threshold: Convergence criterion. Stop when energy change
                is less than this value. Default: 1e-6.
        """
        self.max_iter = max_iterations
        self.threshold = threshold
        self.async_update = async_update
        self.energy_tol = energy_threshold
        self.weights: Optional[np.ndarray] = None
        self.n_neurons: int = 0
        self._is_fitted: bool = False
        self._n_patterns: int = 0

    def train(self, patterns: np.ndarray) -> 'HopfieldNetwork':
        """
        Store patterns in the network using Hebbian learning.
        
        Implements the Hebbian learning rule: W = (1/N) * sum(p_i * p_i^T)
        with zero diagonal (no self-connections).
        
        Args:
            patterns: Binary patterns to store, shape (n_patterns, n_neurons).
                Values must be -1 or 1. Common convention:
                - 1 represents "on" or "white" or "active"
                - -1 represents "off" or "black" or "inactive"
        
        Returns:
            HopfieldNetwork: The trained network (self).
        
        Raises:
            ValueError: If patterns don't contain only {-1, 1} values.
        
        Note:
            Storage capacity is approximately 0.14 * n_neurons patterns.
            Storing more patterns leads to spurious states.
        """
        patterns = np.asarray(patterns)
        
        # Validate patterns are bipolar
        if not np.all(np.isin(patterns, [-1, 1])):
            raise ValueError("Patterns must contain only -1 and 1 values")
        
        self._n_patterns, self.n_neurons = patterns.shape
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        
        # Hebbian learning: W = sum(outer products) / n_neurons
        for p in patterns:
            self.weights += np.outer(p, p)
        
        self.weights /= self.n_neurons
        
        # Remove self-connections
        np.fill_diagonal(self.weights, 0)
        
        self._is_fitted = True
        return self

    def predict(self, pattern: np.ndarray) -> np.ndarray:
        """
        Recall a stored pattern from a (possibly corrupted) input.
        
        Iteratively updates the network state until it converges to a
        stable state (attractor), which should be one of the stored patterns.
        
        Args:
            pattern: Input pattern of shape (n_neurons,) with values {-1, 1}.
                Can be a noisy or partial version of a stored pattern.
        
        Returns:
            np.ndarray: Recalled pattern of shape (n_neurons,).
        
        Raises:
            RuntimeError: If the network hasn't been trained.
        """
        if not self._is_fitted:
            raise RuntimeError("Network must be trained before recall. Call train() first.")
        
        state = pattern.copy()
        last_energy = self._energy(state)
        
        for _ in range(self.max_iter):
            if self.async_update:
                # Asynchronous: update neurons one at a time in random order
                indices = np.random.permutation(self.n_neurons)
                for idx in indices:
                    h = np.dot(self.weights[idx], state) - self.threshold
                    state[idx] = 1 if h >= 0 else -1
            else:
                # Synchronous: update all neurons simultaneously
                h = np.dot(self.weights, state) - self.threshold
                state = np.where(h >= 0, 1, -1)
            
            # Check for energy convergence
            current_energy = self._energy(state)
            if abs(last_energy - current_energy) < self.energy_tol:
                break
            last_energy = current_energy
        
        return state

    def _energy(self, state: np.ndarray) -> float:
        """
        Compute the energy of a network state.
        
        E = -0.5 * s^T * W * s + theta * sum(s)
        
        Lower energy means more stable state. Stored patterns should be
        local energy minima.
        """
        return -0.5 * np.dot(state.T, np.dot(self.weights, state)) + self.threshold * np.sum(state)
    
    def pattern_energy(self, pattern: np.ndarray) -> float:
        """Get the energy of a pattern (public interface)."""
        return self._energy(np.asarray(pattern))


@tool
def train_hopfield_tool(
    patterns: List[List[int]] = Field(description="Binary patterns to store as a 2D list of shape (n_patterns, n_neurons). Values must be -1 or 1"),
    max_iterations: int = Field(default=100, ge=50, le=500, description="Maximum iterations for pattern recall convergence"),
    threshold: float = Field(default=0.0, ge=-1.0, le=1.0, description="Neuron firing threshold"),
    async_update: bool = Field(default=True, description="Use asynchronous (random order) updates. More stable but slower"),
    energy_threshold: float = Field(default=1e-6, ge=1e-10, le=1e-3, description="Energy change threshold for convergence")
) -> Dict[str, Any]:
    """
    Train a Hopfield Network to store binary patterns for associative memory.
    
    Hopfield Networks are recurrent neural networks that function as associative
    (content-addressable) memories. They can store binary patterns and recall
    them even when given partial or noisy inputs.
    
    **When to use:**
    - Pattern completion (recovering full pattern from partial input)
    - Error correction in binary codes
    - Associative memory applications
    - Optimization problems (when cast as energy minimization)
    
    **Capacity and limitations:**
    - Storage capacity: ~0.14 * n_neurons reliable patterns
    - For 100 neurons: store up to ~14 patterns
    - Storing more patterns causes interference and "spurious states"
    - Patterns should be sufficiently different from each other
    
    **Pattern encoding:**
    - Use bipolar encoding: -1 for "off/black", 1 for "on/white"
    - For images: flatten to 1D array, threshold to {-1, 1}
    - Pattern length determines number of neurons
    
    **Example use cases:**
    - Store letter images (e.g., 7x7 = 49 neuron network for alphabet)
    - Store binary codewords for error correction
    - Store prototype patterns for classification
    
    Args:
        patterns: Binary patterns to store, shape (n_patterns, n_neurons).
            Each inner list is one pattern with values -1 or 1.
        max_iterations: Max updates during recall (50-500). Default: 100.
        threshold: Firing threshold for neurons (-1.0 to 1.0). Default: 0.0.
        async_update: Use asynchronous updates (recommended). Default: True.
        energy_threshold: Convergence criterion. Default: 1e-6.
    
    Returns:
        Dict containing:
            - model_id (str): Unique identifier for the trained network
            - status (str): "success" or "error"
            - message (str): Status message
            - n_patterns (int): Number of stored patterns
            - n_neurons (int): Number of neurons (pattern length)
            - capacity (int): Estimated reliable storage capacity
            - pattern_energies (List[float]): Energy of each stored pattern
    
    Example:
        >>> # Store 3 simple patterns
        >>> patterns = [
        ...     [1, 1, -1, -1, 1, 1, -1, -1],  # Pattern A
        ...     [-1, -1, 1, 1, -1, -1, 1, 1],  # Pattern B
        ...     [1, -1, 1, -1, 1, -1, 1, -1],  # Pattern C
        ... ]
        >>> result = train_hopfield_tool(patterns=patterns)
        >>> print(f"Stored {result['n_patterns']} patterns in {result['n_neurons']} neurons")
    """
    try:
        P = np.array(patterns)
        
        if len(P.shape) != 2:
            return {"status": "error", "message": "patterns must be a 2D array"}
        
        # Validate bipolar encoding
        if not np.all(np.isin(P, [-1, 1])):
            return {"status": "error", "message": "Patterns must contain only -1 and 1 values"}
        
        n_patterns, n_neurons = P.shape
        capacity = int(0.14 * n_neurons)
        
        if n_patterns > capacity:
            warning = f"Warning: Storing {n_patterns} patterns exceeds recommended capacity of {capacity}. May have recall errors."
        else:
            warning = None
        
        # Create and train network
        model = HopfieldNetwork(
            max_iterations=max_iterations,
            threshold=threshold,
            async_update=async_update,
            energy_threshold=energy_threshold
        )
        model.train(P)
        
        # Store model
        model_id = f"hopfield_{uuid.uuid4().hex[:8]}"
        MODEL_STORE[model_id] = model
        
        # Calculate energy of each stored pattern
        pattern_energies = [model.pattern_energy(p) for p in P]
        
        result = {
            "status": "success",
            "message": f"Hopfield network trained with {n_patterns} patterns of length {n_neurons}",
            "model_id": model_id,
            "n_patterns": n_patterns,
            "n_neurons": n_neurons,
            "capacity": capacity,
            "pattern_energies": pattern_energies
        }
        
        if warning:
            result["warning"] = warning
        
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool
def inference_hopfield_tool(
    model_id: str = Field(description="The unique model ID returned from train_hopfield_tool"),
    pattern: List[int] = Field(description="Input pattern as a 1D list of values (-1 or 1). Can be noisy/partial version of stored pattern")
) -> Dict[str, Any]:
    """
    Recall a stored pattern from a (possibly corrupted) input using a trained Hopfield Network.
    
    Given a noisy or partial input pattern, the network iteratively updates
    its state until it converges to a stable attractor, which should be
    one of the stored patterns.
    
    **How it works:**
    1. Initialize network state with the input pattern
    2. Iteratively update neurons based on weighted inputs
    3. Network converges to a local energy minimum (stored pattern)
    4. Return the recalled pattern
    
    **Usage scenarios:**
    - **Noise correction**: Input a pattern with random flipped bits
    - **Pattern completion**: Input a pattern with unknown bits set to random
    - **Associative recall**: Input a related pattern to recall associated memory
    
    **Interpreting results:**
    - recalled_pattern: The stable state the network converged to
    - input_energy vs output_energy: Lower output energy means more stable result
    - n_different: How many bits changed from input to output
    
    Args:
        model_id: Unique identifier from train_hopfield_tool.
        pattern: Input pattern as a 1D list with values -1 or 1.
            Length must match the stored patterns.
    
    Returns:
        Dict containing:
            - status (str): "success" or "error"
            - message (str): Status message
            - recalled_pattern (List[int]): The recalled/corrected pattern
            - input_energy (float): Energy of the input pattern
            - output_energy (float): Energy of the recalled pattern
            - n_different (int): Number of bits that changed
    
    Example:
        >>> # Recall with 20% noise
        >>> noisy_input = [1, 1, 1, -1, 1, 1, -1, -1]  # Some bits flipped
        >>> result = inference_hopfield_tool(
        ...     model_id="hopfield_abc12345",
        ...     pattern=noisy_input
        ... )
        >>> print(f"Corrected {result['n_different']} bits")
        >>> print(f"Recalled: {result['recalled_pattern']}")
    """
    try:
        if model_id not in MODEL_STORE:
            return {"status": "error", "message": f"Model '{model_id}' not found. Train a model first."}
        
        model = MODEL_STORE[model_id]
        p = np.array(pattern)
        
        if len(p.shape) != 1:
            return {"status": "error", "message": "pattern must be a 1D array"}
        if len(p) != model.n_neurons:
            return {
                "status": "error",
                "message": f"Pattern length {len(p)} doesn't match network size {model.n_neurons}"
            }
        if not np.all(np.isin(p, [-1, 1])):
            return {"status": "error", "message": "Pattern must contain only -1 and 1 values"}
        
        # Calculate input energy
        input_energy = model.pattern_energy(p)
        
        # Recall pattern
        recalled = model.predict(p)
        
        # Calculate output energy and differences
        output_energy = model.pattern_energy(recalled)
        n_different = int(np.sum(p != recalled))
        
        return {
            "status": "success",
            "message": f"Pattern recalled. {n_different} bits changed.",
            "recalled_pattern": recalled.tolist(),
            "input_energy": float(input_energy),
            "output_energy": float(output_energy),
            "n_different": n_different
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}