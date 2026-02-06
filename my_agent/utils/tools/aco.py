import numpy as np
import time
from typing import List, Optional, Any, Dict, Tuple
from pydantic import Field, BaseModel
from langchain_core.tools import tool


def calculate_tsp_metrics(best_tour: List[int], best_fitness: float, 
                          distance_matrix: np.ndarray,
                          known_optimal: Optional[float] = None,
                          convergence_history: Optional[List[float]] = None,
                          computation_time: Optional[float] = None,
                          multi_run_results: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive TSP optimization metrics.
    
    Metrics computed:
        - Tour Length: Sum of edge distances in the best tour
        - Gap to Optimal: (found - optimal) / optimal * 100%
        - Computation Time: Seconds elapsed during optimization
        - Success Rate %: Percentage of runs within 5% of optimal
        - Convergence Speed: Iterations to reach 90% of final quality
    
    Args:
        best_tour: Best tour found (list of city indices)
        best_fitness: Length of best tour
        distance_matrix: NxN distance matrix
        known_optimal: Known optimal tour length (if available)
        convergence_history: History of best fitness values per iteration
        computation_time: Wall-clock time in seconds for the optimization run
        multi_run_results: Best fitness from multiple independent runs (for success rate)
    
    Returns:
        Dictionary with TSP optimization metrics
    """
    n_cities = len(best_tour)
    
    # --- Tour Length ---
    metrics = {
        "tour_length": float(best_fitness),
        "n_cities": n_cities,
        "avg_edge_length": float(best_fitness / n_cities),
    }
    
    edge_lengths = []
    for i in range(n_cities):
        j = (i + 1) % n_cities
        edge_lengths.append(distance_matrix[best_tour[i], best_tour[j]])
    
    metrics["min_edge"] = float(np.min(edge_lengths))
    metrics["max_edge"] = float(np.max(edge_lengths))
    metrics["edge_std"] = float(np.std(edge_lengths))
    
    # --- Computation Time ---
    if computation_time is not None:
        metrics["computation_time_seconds"] = float(computation_time)
    
    # --- Gap to Optimal ---
    if known_optimal is not None and known_optimal > 0:
        gap = best_fitness - known_optimal
        gap_percentage = (gap / known_optimal) * 100
        metrics["known_optimal"] = float(known_optimal)
        metrics["optimality_gap"] = float(gap)
        metrics["optimality_gap_percentage"] = float(gap_percentage)
        metrics["is_optimal"] = gap_percentage < 0.01
        
        if gap_percentage < 1:
            metrics["performance_rating"] = "excellent"
        elif gap_percentage < 5:
            metrics["performance_rating"] = "good"
        elif gap_percentage < 10:
            metrics["performance_rating"] = "acceptable"
        else:
            metrics["performance_rating"] = "poor"
        
        # --- Success Rate % (runs within 5% of optimal) ---
        if multi_run_results is not None and len(multi_run_results) > 0:
            threshold = known_optimal * 1.05
            n_successful = sum(1 for r in multi_run_results if r <= threshold)
            metrics["success_rate_percent"] = float(n_successful / len(multi_run_results) * 100)
            metrics["n_runs"] = len(multi_run_results)
            metrics["mean_tour_length"] = float(np.mean(multi_run_results))
            metrics["std_tour_length"] = float(np.std(multi_run_results))
            metrics["best_of_runs"] = float(np.min(multi_run_results))
            metrics["worst_of_runs"] = float(np.max(multi_run_results))
    
    # --- Convergence Speed ---
    if convergence_history and len(convergence_history) > 1:
        metrics["initial_fitness"] = float(convergence_history[0])
        metrics["improvement"] = float(convergence_history[0] - best_fitness)
        metrics["improvement_percentage"] = float(
            (convergence_history[0] - best_fitness) / convergence_history[0] * 100
        )
        
        # Iterations to reach 90% of final quality
        total_improvement = convergence_history[0] - best_fitness
        if total_improvement > 0:
            threshold_90 = convergence_history[0] - 0.9 * total_improvement
            for i, val in enumerate(convergence_history):
                if val <= threshold_90:
                    metrics["convergence_speed_iterations"] = i
                    metrics["convergence_speed_percent"] = float(
                        i / len(convergence_history) * 100
                    )
                    break
        
        last_10 = convergence_history[-10:] if len(convergence_history) >= 10 else convergence_history
        if len(last_10) > 1:
            variance = np.var(last_10)
            metrics["convergence_variance"] = float(variance)
            metrics["converged"] = variance < 1e-6
    
    return metrics


class AntColonyOptimization:
    """
    Ant Colony Optimization for combinatorial optimization.
    
    Simulates ants constructing solutions probabilistically based on
    pheromone trails and heuristic information. Particularly effective
    for graph-based routing problems like TSP.
    
    Attributes:
        n_ants (int): Number of ants in the colony.
        max_iter (int): Maximum iterations.
        alpha (float): Pheromone importance.
        beta (float): Heuristic importance.
        rho (float): Evaporation rate.
        Q (float): Pheromone deposit factor.
        init_phero (float): Initial pheromone level.
        do_local_search (bool): Whether to apply 2-opt improvement.
    """
    
    def __init__(
        self,
        n_ants: int = 50,
        max_iterations: int = 500,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.5,
        q: float = 1.0,
        initial_pheromone: float = 0.1,
        local_search: bool = True
    ):
        self.n_ants = n_ants
        self.max_iter = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = evaporation_rate
        self.Q = q
        self.init_phero = initial_pheromone
        self.do_local_search = local_search
        self._history: List[float] = []

    def fit(self, distance_matrix: np.ndarray) -> Tuple[List[int], float]:
        """
        Solve the TSP using Ant Colony Optimization.
        
        Args:
            distance_matrix: NxN matrix of distances between cities.
        
        Returns:
            Tuple[List[int], float]: Best tour and its length.
        """
        n_cities = len(distance_matrix)
        
        # Initialize pheromones
        pheromone = np.full((n_cities, n_cities), self.init_phero)
        
        # Pre-calculate heuristic (inverse distance)
        heuristic = 1.0 / (distance_matrix + 1e-10)
        np.fill_diagonal(heuristic, 0)

        best_path: Optional[List[int]] = None
        best_dist = float('inf')
        self._history = []

        for iteration in range(self.max_iter):
            all_paths = []
            all_dists = []

            # Each ant constructs a solution
            for k in range(self.n_ants):
                path = [np.random.randint(n_cities)]
                visited = set(path)
                
                # Build tour city by city
                for _ in range(n_cities - 1):
                    current = path[-1]
                    
                    # Calculate selection probabilities
                    probs = np.zeros(n_cities)
                    unvisited = [node for node in range(n_cities) if node not in visited]
                    
                    total = 0.0
                    for node in unvisited:
                        tau = pheromone[current, node] ** self.alpha
                        eta = heuristic[current, node] ** self.beta
                        val = tau * eta
                        probs[node] = val
                        total += val
                    
                    if total == 0:
                        next_node = np.random.choice(unvisited)
                    else:
                        probs = probs / total
                        next_node = np.random.choice(range(n_cities), p=probs)
                    
                    path.append(next_node)
                    visited.add(next_node)

                # Optional: Local Search (2-Opt)
                if self.do_local_search:
                    path = self._two_opt(path, distance_matrix)

                dist = self._path_length(path, distance_matrix)
                all_paths.append(path)
                all_dists.append(dist)

                if dist < best_dist:
                    best_dist = dist
                    best_path = list(path)
            
            self._history.append(best_dist)

            # Update pheromones
            pheromone *= (1 - self.rho)
            
            for i in range(self.n_ants):
                path = all_paths[i]
                dist = all_dists[i]
                deposit_amount = self.Q / dist
                
                for j in range(n_cities - 1):
                    u, v = path[j], path[j+1]
                    pheromone[u, v] += deposit_amount
                    pheromone[v, u] += deposit_amount
                
                u, v = path[-1], path[0]
                pheromone[u, v] += deposit_amount
                pheromone[v, u] += deposit_amount

        return best_path, best_dist

    def _path_length(self, path: List[int], dist_matrix: np.ndarray) -> float:
        """Calculate total tour length."""
        total = 0.0
        for i in range(len(path) - 1):
            total += dist_matrix[path[i], path[i+1]]
        total += dist_matrix[path[-1], path[0]]
        return total

    def _two_opt(self, path: List[int], dist_matrix: np.ndarray) -> List[int]:
        """Apply 2-opt local search for tour improvement."""
        best_path = list(path)
        improved = True
        n = len(path)
        
        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue
                    
                    d_current = (dist_matrix[best_path[i-1], best_path[i]] + 
                                dist_matrix[best_path[j], best_path[(j+1) % n]])
                    d_new = (dist_matrix[best_path[i-1], best_path[j]] + 
                            dist_matrix[best_path[i], best_path[(j+1) % n]])

                    if d_new < d_current:
                        best_path[i:j+1] = best_path[i:j+1][::-1]
                        improved = True
        
        return best_path


class TrainACOInput(BaseModel):
    distance_matrix: List[List[float]] = Field(description="Distance/cost matrix as a 2D list. For TSP, distance_matrix[i][j] is the distance from city i to city j")
    n_ants: int = Field(default=50, ge=10, le=100, description="Number of ants in the colony")
    max_iterations: int = Field(default=500, ge=100, le=2000, description="Maximum number of iterations")
    alpha: float = Field(default=1.0, ge=0.5, le=2.0, description="Pheromone importance - higher values make ants follow pheromone trails more strongly")
    beta: float = Field(default=2.0, ge=1.0, le=5.0, description="Heuristic importance - higher values make ants prefer shorter edges")
    evaporation_rate: float = Field(default=0.5, ge=0.1, le=0.9, description="Pheromone evaporation rate - higher values mean faster forgetting of old trails")
    q: float = Field(default=1.0, ge=0.1, le=10.0, description="Pheromone deposit factor - amount of pheromone deposited")
    initial_pheromone: float = Field(default=0.1, ge=0.01, le=1.0, description="Initial pheromone level on all edges")
    local_search: bool = Field(default=True, description="Apply 2-opt local search improvement to each ant's tour")
    known_optimal: Optional[float] = Field(default=None, description="Known optimal tour length for computing optimality gap")
    n_runs: int = Field(default=1, ge=1, le=10, description="Number of independent runs for computing success rate statistics")


@tool(args_schema=TrainACOInput)
def aco_tool(
    distance_matrix: List[List[float]],
    n_ants: int = 50,
    max_iterations: int = 500,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation_rate: float = 0.5,
    q: float = 1.0,
    initial_pheromone: float = 0.1,
    local_search: bool = True,
    known_optimal: Optional[float] = None,
    n_runs: int = 1
) -> Dict[str, Any]:
    """
    Solve the Traveling Salesman Problem using Ant Colony Optimization.
    
    ACO simulates ants constructing tours probabilistically based on
    pheromone trails (historical success) and heuristic information
    (edge distances). It's naturally suited for graph-based routing problems.
    
    Use for: TSP, vehicle routing, network routing, graph path problems.
    
    **Parameter tuning:**
    - alpha (pheromone importance): Higher = ants follow trails more
    - beta (heuristic importance): Higher = ants prefer short edges
    - evaporation_rate: Higher = faster adaptation, lower = more exploitation
    - Typical good values: alpha=1, beta=2-5, evaporation=0.5
    
    **Metrics computed:**
    - Tour Length: Total distance of best tour
    - Gap to Optimal: Percentage above known optimal (when provided)
    - Computation Time: Wall-clock seconds elapsed
    - Success Rate %: Percentage of runs within 5% of optimal (when n_runs > 1)
    - Convergence Speed: Iterations to reach 90% of final quality
    
    Returns:
        Dict containing:
            - status: "success" or "error"
            - best_tour: Best route found (city indices)
            - best_fitness: Tour length of best solution
            - n_cities: Number of cities
            - computation_time_seconds: Wall-clock time
            - convergence_history: Best fitness per iteration
            - metrics: Comprehensive TSP metrics
    """
    try:
        dist_matrix = np.array(distance_matrix)
        
        if len(dist_matrix.shape) != 2:
            return {"status": "error", "message": "distance_matrix must be a 2D array"}
        if dist_matrix.shape[0] != dist_matrix.shape[1]:
            return {"status": "error", "message": "distance_matrix must be square"}
        
        n_cities = dist_matrix.shape[0]
        
        multi_run_results = []
        best_overall_tour = None
        best_overall_fitness = float('inf')
        best_overall_history = None
        total_time = 0.0
        
        for run_idx in range(n_runs):
            aco = AntColonyOptimization(
                n_ants=n_ants,
                max_iterations=max_iterations,
                alpha=alpha,
                beta=beta,
                evaporation_rate=evaporation_rate,
                q=q,
                initial_pheromone=initial_pheromone,
                local_search=local_search
            )
            
            start_time = time.time()
            best_tour, best_fitness = aco.fit(dist_matrix)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            multi_run_results.append(best_fitness)
            
            if best_fitness < best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_tour = best_tour
                best_overall_history = aco._history
        
        metrics = calculate_tsp_metrics(
            best_overall_tour, best_overall_fitness, dist_matrix,
            known_optimal=known_optimal,
            convergence_history=best_overall_history,
            computation_time=total_time,
            multi_run_results=multi_run_results if n_runs > 1 else None
        )
        
        result = {
            "status": "success",
            "message": f"ACO optimization completed for {n_cities}-city TSP",
            "best_tour": best_overall_tour,
            "best_fitness": float(best_overall_fitness),
            "n_cities": n_cities,
            "n_ants": n_ants,
            "iterations_run": max_iterations,
            "local_search_applied": local_search,
            "computation_time_seconds": float(total_time),
            "convergence_history": best_overall_history[-20:] if best_overall_history else [],
            "metrics": metrics
        }
        
        if "performance_rating" in metrics:
            result["message"] = f"ACO completed. Tour: {best_overall_fitness:.2f}, Gap: {metrics['optimality_gap_percentage']:.2f}% ({metrics['performance_rating']}), Time: {total_time:.2f}s"
        
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}