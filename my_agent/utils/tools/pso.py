import numpy as np
import time
from typing import List, Optional, Any, Dict, Tuple, Callable
from pydantic import Field, BaseModel
from langchain_core.tools import tool


def calculate_optimization_metrics(best_pos: np.ndarray, best_score: float,
                                    known_optimal: float = 0.0,
                                    convergence_history: Optional[List[float]] = None,
                                    computation_time: Optional[float] = None,
                                    function_evaluations: int = 0,
                                    multi_run_results: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive continuous optimization metrics.
    
    Metrics computed:
        - Best Fitness: f(x*) - the best objective value found
        - Mean Fitness: Average of best fitness across runs
        - Std Dev: Standard deviation of best fitness across runs
        - Error: |f(x*) - f(x_opt)| - absolute error from known optimum
        - Success Rate: % of runs with error < 1e-4
        - Function Evaluations: Total calls to objective function
    
    Args:
        best_pos: Best position found
        best_score: Best fitness value found
        known_optimal: Known global minimum value
        convergence_history: History of best fitness values
        computation_time: Wall-clock time in seconds
        function_evaluations: Total number of objective function calls
        multi_run_results: Best fitness from multiple independent runs
    
    Returns:
        Dictionary with optimization metrics
    """
    dimensions = len(best_pos)
    
    # --- Best Fitness ---
    metrics = {
        "best_fitness": float(best_score),
        "known_optimal": float(known_optimal),
        "dimensions": dimensions,
        "position_norm": float(np.linalg.norm(best_pos)),
    }
    
    # --- Error: |f(x*) - f(x_opt)| ---
    error = abs(best_score - known_optimal)
    metrics["error"] = float(error)
    metrics["absolute_gap"] = float(best_score - known_optimal)
    
    if abs(known_optimal) > 1e-10:
        metrics["relative_gap_percentage"] = float((best_score - known_optimal) / abs(known_optimal) * 100)
    else:
        metrics["relative_gap_percentage"] = float(best_score * 100)
    
    # Performance rating
    if error < 1e-6:
        metrics["performance_rating"] = "excellent"
        metrics["solution_quality"] = "optimal"
    elif error < 0.1:
        metrics["performance_rating"] = "excellent"
        metrics["solution_quality"] = "near-optimal"
    elif error < 1.0:
        metrics["performance_rating"] = "good"
        metrics["solution_quality"] = "good"
    elif error < 10.0:
        metrics["performance_rating"] = "acceptable"
        metrics["solution_quality"] = "acceptable"
    else:
        metrics["performance_rating"] = "poor"
        metrics["solution_quality"] = "suboptimal"
    
    # --- Function Evaluations ---
    if function_evaluations > 0:
        metrics["function_evaluations"] = function_evaluations
    
    # --- Computation Time ---
    if computation_time is not None:
        metrics["computation_time_seconds"] = float(computation_time)
    
    # --- Multi-run Statistics: Mean Fitness, Std Dev, Success Rate ---
    if multi_run_results is not None and len(multi_run_results) > 0:
        metrics["mean_fitness"] = float(np.mean(multi_run_results))
        metrics["std_fitness"] = float(np.std(multi_run_results))
        metrics["median_fitness"] = float(np.median(multi_run_results))
        metrics["best_of_runs"] = float(np.min(multi_run_results))
        metrics["worst_of_runs"] = float(np.max(multi_run_results))
        metrics["n_runs"] = len(multi_run_results)
        
        # Success Rate: % of runs with error < 1e-4
        n_successful = sum(1 for r in multi_run_results if abs(r - known_optimal) < 1e-4)
        metrics["success_rate_percent"] = float(n_successful / len(multi_run_results) * 100)
        metrics["n_successful_runs"] = n_successful
    
    # --- Convergence Analysis ---
    if convergence_history and len(convergence_history) > 1:
        metrics["initial_fitness"] = float(convergence_history[0])
        metrics["improvement"] = float(convergence_history[0] - best_score)
        metrics["improvement_factor"] = float(convergence_history[0] / (best_score + 1e-10))
        
        # Convergence speed: iterations to reach 90% of improvement
        total_improvement = convergence_history[0] - best_score
        if total_improvement > 0:
            threshold_90 = convergence_history[0] - 0.9 * total_improvement
            for i, val in enumerate(convergence_history):
                if val <= threshold_90:
                    metrics["iterations_to_90_percent"] = i
                    break
        
        last_10 = convergence_history[-10:] if len(convergence_history) >= 10 else convergence_history
        if len(last_10) > 1:
            variance = np.var(last_10)
            metrics["convergence_variance"] = float(variance)
            metrics["converged"] = variance < 1e-10
    
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
        self._function_evaluations: int = 0

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

        self._function_evaluations = 0

        # Evaluate initial population
        for i in range(self.n_particles):
            score = objective_func(pos[i])
            self._function_evaluations += 1
            p_best_scores[i] = score
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_pos = pos[i].copy()

        self._history = [self.global_best_score]

        # Optimization loop
        w = self.w_start
        for t in range(self.max_iter):
            if self.w_decay:
                w = self.w_start - (self.w_start - 0.4) * (t / self.max_iter)

            r1 = np.random.rand(self.n_particles, n_dim)
            r2 = np.random.rand(self.n_particles, n_dim)

            cognitive = self.c1 * r1 * (p_best_pos - pos)
            social = self.c2 * r2 * (self.global_best_pos - pos)
            vel = w * vel + cognitive + social

            vel = np.clip(vel, v_min, v_max)
            pos += vel
            pos = np.clip(pos, bounds[:, 0], bounds[:, 1])

            for i in range(self.n_particles):
                score = objective_func(pos[i])
                self._function_evaluations += 1
                
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


class PSOInput(BaseModel):
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
    n_runs: int = Field(default=1, ge=1, le=10, description="Number of independent runs for computing multi-run statistics (mean, std, success rate)")


@tool(args_schema=PSOInput)
def pso_tool(
    function_name: str = "rastrigin",
    dimensions: int = 10,
    n_particles: int = 50,
    max_iterations: int = 500,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    w_decay: bool = True,
    velocity_clamp: float = 0.5,
    custom_bounds: Optional[List[List[float]]] = None,
    n_runs: int = 1,
) -> Dict[str, Any]:
    """
    Solve continuous optimization problems using Particle Swarm Optimization.
    
    PSO simulates a swarm of particles searching for the global minimum.
    Each particle is influenced by its personal best position and the
    swarm's global best position.
    
    Use for: Continuous function optimization, Multimodal problems, Engineering design optimization
    
    **Metrics computed:**
    - Best Fitness: f(x*) - best objective value found
    - Mean Fitness: Average best fitness across runs (when n_runs > 1)
    - Std Dev: Standard deviation of best fitness (when n_runs > 1)
    - Error: |f(x*) - f(x_opt)| - absolute error from known optimum
    - Success Rate: % of runs with error < 1e-4 (when n_runs > 1)
    - Function Evaluations: Total calls to objective function
    
    **Parameter tuning:**
    - w (inertia): Higher = more exploration, lower = more exploitation
    - c1 (cognitive): Higher = particles trust their own experience more
    - c2 (social): Higher = particles follow the swarm more
    - w_decay: Usually helps convergence by reducing exploration over time
    - velocity_clamp: Prevents particles from moving too fast
    
    Returns:
        Dict containing:
            - status: "success" or "error"
            - best_position: Best solution found
            - best_fitness: Fitness value at best position
            - error: Absolute error from known optimal
            - function_evaluations: Total objective function calls
            - computation_time_seconds: Wall-clock time
            - metrics: Comprehensive optimization metrics
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
        
        # Run optimization (potentially multiple times)
        multi_run_results = []
        best_overall_pos = None
        best_overall_score = float('inf')
        best_overall_history = None
        total_func_evals = 0
        total_time = 0.0
        
        for run_idx in range(n_runs):
            pso = PSO(
                n_particles=n_particles,
                max_iterations=max_iterations,
                w=w,
                c1=c1,
                c2=c2,
                w_decay=w_decay,
                velocity_clamp=velocity_clamp
            )
            
            start_time = time.time()
            best_pos, best_score = pso.optimize(func, bounds)
            elapsed = time.time() - start_time
            total_time += elapsed
            total_func_evals += pso._function_evaluations
            
            multi_run_results.append(best_score)
            
            if best_score < best_overall_score:
                best_overall_score = best_score
                best_overall_pos = best_pos
                best_overall_history = pso._history
        
        # Calculate metrics
        metrics = calculate_optimization_metrics(
            best_overall_pos, best_overall_score, known_optimal,
            convergence_history=best_overall_history,
            computation_time=total_time,
            function_evaluations=total_func_evals,
            multi_run_results=multi_run_results if n_runs > 1 else None
        )
        
        result = {
            "status": "success",
            "message": f"PSO optimization completed for {dimensions}D {function_name} function",
            "best_position": best_overall_pos.tolist(),
            "best_fitness": float(best_overall_score),
            "error": float(abs(best_overall_score - known_optimal)),
            "known_optimal": known_optimal,
            "dimensions": dimensions,
            "n_particles": n_particles,
            "iterations_run": max_iterations,
            "function_evaluations": total_func_evals,
            "computation_time_seconds": float(total_time),
            "convergence_history": best_overall_history[-20:] if best_overall_history else [],
            "metrics": metrics
        }
        
        msg = f"PSO completed. Best: {best_overall_score:.6f}, Error: {metrics['error']:.6f} ({metrics['performance_rating']}), FuncEvals: {total_func_evals}"
        if n_runs > 1:
            msg += f", Success Rate: {metrics.get('success_rate_percent', 0):.1f}%"
        result["message"] = msg
        
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}