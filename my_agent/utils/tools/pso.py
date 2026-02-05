import numpy as np
from typing import List, Optional, Any, Dict, Tuple, Callable
from pydantic import Field
from langchain_core.tools import tool


def calculate_optimization_metrics(best_pos: np.ndarray, best_score: float,
                                    known_optimal: float = 0.0,
                                    convergence_history: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Calculate continuous optimization metrics.
    
    Args:
        best_pos: Best position found
        best_score: Best fitness value found
        known_optimal: Known global minimum value
        convergence_history: History of best fitness values
    
    Returns:
        Dictionary with optimization metrics
    """
    dimensions = len(best_pos)
    
    metrics = {
        "best_fitness": float(best_score),
        "known_optimal": float(known_optimal),
        "dimensions": dimensions,
        "position_norm": float(np.linalg.norm(best_pos)),
    }
    
    # Gap metrics
    gap = best_score - known_optimal
    metrics["absolute_gap"] = float(gap)
    
    if abs(known_optimal) > 1e-10:
        gap_percentage = (gap / abs(known_optimal)) * 100
        metrics["relative_gap_percentage"] = float(gap_percentage)
    else:
        metrics["relative_gap_percentage"] = float(best_score * 100)  # For optimal=0
    
    # Performance rating based on gap
    if gap < 1e-6:
        metrics["performance_rating"] = "excellent"
        metrics["solution_quality"] = "optimal"
    elif gap < 0.1:
        metrics["performance_rating"] = "excellent"
        metrics["solution_quality"] = "near-optimal"
    elif gap < 1.0:
        metrics["performance_rating"] = "good"
        metrics["solution_quality"] = "good"
    elif gap < 10.0:
        metrics["performance_rating"] = "acceptable"
        metrics["solution_quality"] = "acceptable"
    else:
        metrics["performance_rating"] = "poor"
        metrics["solution_quality"] = "suboptimal"
    
    # Convergence analysis
    if convergence_history and len(convergence_history) > 1:
        metrics["initial_fitness"] = float(convergence_history[0])
        metrics["improvement"] = float(convergence_history[0] - best_score)
        metrics["improvement_factor"] = float(convergence_history[0] / (best_score + 1e-10))
        
        # Check for early convergence
        last_10 = convergence_history[-10:] if len(convergence_history) >= 10 else convergence_history
        if len(last_10) > 1:
            variance = np.var(last_10)
            metrics["convergence_variance"] = float(variance)
            metrics["converged"] = variance < 1e-10
        
        # Iterations to reach 90% of improvement
        total_improvement = convergence_history[0] - best_score
        if total_improvement > 0:
            threshold = convergence_history[0] - 0.9 * total_improvement
            for i, val in enumerate(convergence_history):
                if val <= threshold:
                    metrics["iterations_to_90_percent"] = i
                    break
    
    return metrics


class PSO:
    """
    Particle Swarm Optimization for continuous optimization.
    
    PSO simulates a swarm of particles moving through the search space,
    influenced by their personal best positions and the global best position.
    Effective for continuous, multimodal optimization problems.
    
    Attributes:
        n_particles (int): Number of particles in the swarm.
        max_iter (int): Maximum iterations.
        w_start (float): Initial inertia weight.
        c1 (float): Cognitive coefficient (personal best attraction).
        c2 (float): Social coefficient (global best attraction).
        w_decay (bool): Whether to decay inertia weight.
        v_clamp_frac (float): Velocity clamping as fraction of range.
    """
    
    def __init__(
        self,
        n_particles: int = 50,
        max_iterations: int = 500,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        w_decay: bool = True,
        velocity_clamp: float = 0.5
    ):
        self.n_particles = n_particles
        self.max_iter = max_iterations
        self.w_start = w
        self.c1 = c1
        self.c2 = c2
        self.w_decay = w_decay
        self.v_clamp_frac = velocity_clamp
        
        self.global_best_pos: Optional[np.ndarray] = None
        self.global_best_score: float = float('inf')
        self._history: List[float] = []

    def optimize(
        self,
        objective_func: Callable,
        bounds: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize the objective function within given bounds.
        
        Args:
            objective_func: Function to minimize, takes array of shape (n_dim,).
            bounds: List of (min, max) tuples for each dimension.
        
        Returns:
            Tuple[np.ndarray, float]: Best position and best score.
        """
        bounds = np.array(bounds)
        n_dim = len(bounds)
        range_width = bounds[:, 1] - bounds[:, 0]
        v_max = range_width * self.v_clamp_frac
        v_min = -v_max

        # Initialize particles
        pos = bounds[:, 0] + np.random.rand(self.n_particles, n_dim) * range_width
        vel = (np.random.rand(self.n_particles, n_dim) - 0.5) * range_width * 0.1
        
        # Personal bests
        p_best_pos = pos.copy()
        p_best_scores = np.full(self.n_particles, float('inf'))

        # Evaluate initial population
        for i in range(self.n_particles):
            score = objective_func(pos[i])
            p_best_scores[i] = score
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_pos = pos[i].copy()

        self._history = [self.global_best_score]

        # Optimization loop
        w = self.w_start
        for t in range(self.max_iter):
            # Decay inertia weight linearly
            if self.w_decay:
                w = self.w_start - (self.w_start - 0.4) * (t / self.max_iter)

            # Random coefficients
            r1 = np.random.rand(self.n_particles, n_dim)
            r2 = np.random.rand(self.n_particles, n_dim)

            # Update velocity
            cognitive = self.c1 * r1 * (p_best_pos - pos)
            social = self.c2 * r2 * (self.global_best_pos - pos)
            vel = w * vel + cognitive + social

            # Clamp velocity
            vel = np.clip(vel, v_min, v_max)

            # Update position
            pos += vel

            # Enforce bounds (clipping)
            pos = np.clip(pos, bounds[:, 0], bounds[:, 1])

            # Evaluate and update bests
            for i in range(self.n_particles):
                score = objective_func(pos[i])
                
                if score < p_best_scores[i]:
                    p_best_scores[i] = score
                    p_best_pos[i] = pos[i].copy()
                    
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_pos = pos[i].copy()
            
            self._history.append(self.global_best_score)

        return self.global_best_pos, self.global_best_score


# Benchmark functions
def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function - highly multimodal, global min at origin = 0."""
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def ackley(x: np.ndarray) -> float:
    """Ackley function - flat outer region, deep hole at origin = 0."""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function - narrow curved valley, global min at (1,1,...) = 0."""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def sphere(x: np.ndarray) -> float:
    """Sphere function - simple unimodal, global min at origin = 0."""
    return np.sum(x**2)


BENCHMARK_FUNCTIONS = {
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "ackley": (ackley, (-5, 5)),
    "rosenbrock": (rosenbrock, (-5, 10)),
    "sphere": (sphere, (-5.12, 5.12))
}


@tool
def pso_tool(
    function_name: str = Field(default="rastrigin", description="Benchmark function to optimize: 'rastrigin', 'ackley', 'rosenbrock', or 'sphere'"),
    dimensions: int = Field(default=10, ge=2, le=50, description="Number of dimensions for the optimization problem"),
    n_particles: int = Field(default=50, ge=20, le=200, description="Number of particles in the swarm"),
    max_iterations: int = Field(default=500, ge=100, le=2000, description="Maximum number of iterations"),
    w: float = Field(default=0.7, ge=0.4, le=0.9, description="Inertia weight - controls momentum of particles"),
    c1: float = Field(default=1.5, ge=1.0, le=2.5, description="Cognitive coefficient - attraction to personal best"),
    c2: float = Field(default=1.5, ge=1.0, le=2.5, description="Social coefficient - attraction to global best"),
    w_decay: bool = Field(default=True, description="Whether to linearly decrease inertia weight over iterations"),
    velocity_clamp: float = Field(default=0.5, ge=0.1, le=1.0, description="Velocity clamping as fraction of search range"),
    custom_bounds: Optional[List[List[float]]] = Field(default=None, description="Custom bounds as list of [min, max] for each dimension. If None, uses function defaults")
) -> Dict[str, Any]:
    """
    Solve continuous optimization problems using Particle Swarm Optimization.
    
    PSO simulates a swarm of particles searching for the global minimum.
    Each particle is influenced by its personal best position and the
    swarm's global best position.
    
    **When to use:**
    - Continuous function optimization
    - Multimodal problems (many local minima)
    - Neural network training
    - Parameter tuning
    - Engineering design optimization
    
    **Benchmark functions available:**
    - rastrigin: Highly multimodal, tests global search ability
    - ackley: Large flat region with deep hole at center
    - rosenbrock: Narrow curved valley, tests convergence
    - sphere: Simple unimodal, baseline comparison
    
    **How PSO works:**
    1. Initialize particles at random positions with random velocities
    2. Evaluate fitness of each particle
    3. Update personal best if current position is better
    4. Update global best if any particle found better solution
    5. Update velocities based on inertia + cognitive + social components
    6. Move particles to new positions
    7. Repeat until convergence or max iterations
    
    **Parameter tuning:**
    - w (inertia): Higher = more exploration, lower = more exploitation
    - c1 (cognitive): Higher = particles trust their own experience more
    - c2 (social): Higher = particles follow the swarm more
    - w_decay: Usually helps convergence by reducing exploration over time
    - velocity_clamp: Prevents particles from moving too fast
    
    Args:
        function_name: Benchmark function to optimize. Default: "rastrigin".
        dimensions: Problem dimensionality (2-50). Default: 10.
        n_particles: Swarm size (20-200). Default: 50.
        max_iterations: Max iterations (100-2000). Default: 500.
        w: Inertia weight (0.4-0.9). Default: 0.7.
        c1: Cognitive coefficient (1.0-2.5). Default: 1.5.
        c2: Social coefficient (1.0-2.5). Default: 1.5.
        w_decay: Decay inertia weight. Default: True.
        velocity_clamp: Velocity limit (0.1-1.0). Default: 0.5.
        custom_bounds: Optional custom search bounds.
    
    Returns:
        Dict containing:
            - status (str): "success" or "error"
            - best_position (List[float]): Best solution found
            - best_fitness (float): Fitness value at best position
            - known_optimal (float): Known global minimum value
            - gap (float): Difference from known optimal
            - dimensions (int): Problem dimensions
            - convergence_history (List[float]): Best fitness per iteration
    
    Example:
        >>> # Optimize 20-dimensional Rastrigin function
        >>> result = pso_tool(
        ...     function_name="rastrigin",
        ...     dimensions=20,
        ...     n_particles=100,
        ...     max_iterations=1000
        ... )
        >>> print(f"Found minimum: {result['best_fitness']:.6f}")
        >>> print(f"Gap from optimal: {result['gap']:.6f}")
    """
    try:
        if function_name not in BENCHMARK_FUNCTIONS:
            return {
                "status": "error",
                "message": f"Unknown function '{function_name}'. Available: {list(BENCHMARK_FUNCTIONS.keys())}"
            }
        
        func, default_bounds = BENCHMARK_FUNCTIONS[function_name]
        
        if custom_bounds is not None:
            bounds = [(b[0], b[1]) for b in custom_bounds]
            if len(bounds) != dimensions:
                return {"status": "error", "message": f"custom_bounds length must match dimensions"}
        else:
            bounds = [default_bounds] * dimensions
        
        known_optimal = 0.0
        
        pso = PSO(
            n_particles=n_particles,
            max_iterations=max_iterations,
            w=w,
            c1=c1,
            c2=c2,
            w_decay=w_decay,
            velocity_clamp=velocity_clamp
        )
        
        best_pos, best_score = pso.optimize(func, bounds)
        
        # Calculate metrics
        metrics = calculate_optimization_metrics(
            best_pos, best_score, known_optimal, pso._history
        )
        
        result = {
            "status": "success",
            "message": f"PSO optimization completed for {dimensions}D {function_name} function",
            "best_position": best_pos.tolist(),
            "best_fitness": float(best_score),
            "known_optimal": known_optimal,
            "dimensions": dimensions,
            "n_particles": n_particles,
            "iterations_run": max_iterations,
            "convergence_history": pso._history[-20:],
            "metrics": metrics
        }
        
        result["message"] = f"PSO completed. Best: {best_score:.6f}, Gap: {metrics['absolute_gap']:.6f} ({metrics['performance_rating']})"
        
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}