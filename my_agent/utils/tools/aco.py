import numpy as np
from typing import List, Optional, Any, Dict, Tuple
from pydantic import Field
from langchain_core.tools import tool


def calculate_tsp_metrics(best_tour: List[int], best_fitness: float, 
                          distance_matrix: np.ndarray,
                          known_optimal: Optional[float] = None,
                          convergence_history: Optional[List[float]] = None) -> Dict[str, Any]:
    """Calculate TSP optimization metrics."""
    n_cities = len(best_tour)
    
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
    
    if convergence_history and len(convergence_history) > 1:
        metrics["initial_fitness"] = float(convergence_history[0])
        metrics["improvement"] = float(convergence_history[0] - best_fitness)
        metrics["improvement_percentage"] = float((convergence_history[0] - best_fitness) / convergence_history[0] * 100)
        
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
            distance_matrix: N×N matrix of distances between cities.
        
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
            # 1. Evaporation
            pheromone *= (1 - self.rho)
            
            # 2. Deposit pheromone
            for i in range(self.n_ants):
                path = all_paths[i]
                dist = all_dists[i]
                deposit_amount = self.Q / dist
                
                for j in range(n_cities - 1):
                    u, v = path[j], path[j+1]
                    pheromone[u, v] += deposit_amount
                    pheromone[v, u] += deposit_amount  # Symmetric TSP
                
                # Close the loop
                u, v = path[-1], path[0]
                pheromone[u, v] += deposit_amount
                pheromone[v, u] += deposit_amount

        return best_path, best_dist

    def _path_length(self, path: List[int], dist_matrix: np.ndarray) -> float:
        """Calculate total tour length."""
        total = 0.0
        for i in range(len(path) - 1):
            total += dist_matrix[path[i], path[i+1]]
        total += dist_matrix[path[-1], path[0]]  # Return to start
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
                    
                    # Calculate change in distance
                    d_current = (dist_matrix[best_path[i-1], best_path[i]] + 
                                dist_matrix[best_path[j], best_path[(j+1) % n]])
                    d_new = (dist_matrix[best_path[i-1], best_path[j]] + 
                            dist_matrix[best_path[i], best_path[(j+1) % n]])

                    if d_new < d_current:
                        # Reverse segment from i to j
                        best_path[i:j+1] = best_path[i:j+1][::-1]
                        improved = True
        
        return best_path


@tool
def aco_tool(
    distance_matrix: List[List[float]] = Field(description="Distance/cost matrix as a 2D list. For TSP, distance_matrix[i][j] is the distance from city i to city j"),
    n_ants: int = Field(default=50, ge=10, le=100, description="Number of ants in the colony"),
    max_iterations: int = Field(default=500, ge=100, le=2000, description="Maximum number of iterations"),
    alpha: float = Field(default=1.0, ge=0.5, le=2.0, description="Pheromone importance - higher values make ants follow pheromone trails more strongly"),
    beta: float = Field(default=2.0, ge=1.0, le=5.0, description="Heuristic importance - higher values make ants prefer shorter edges"),
    evaporation_rate: float = Field(default=0.5, ge=0.1, le=0.9, description="Pheromone evaporation rate - higher values mean faster forgetting of old trails"),
    q: float = Field(default=1.0, ge=0.1, le=10.0, description="Pheromone deposit factor - amount of pheromone deposited"),
    initial_pheromone: float = Field(default=0.1, ge=0.01, le=1.0, description="Initial pheromone level on all edges"),
    local_search: bool = Field(default=True, description="Apply 2-opt local search improvement to each ant's tour"),
    known_optimal: Optional[float] = Field(default=None, description="Known optimal tour length for computing optimality gap")
) -> Dict[str, Any]:
    """
    Solve the Traveling Salesman Problem using Ant Colony Optimization.
    
    ACO simulates ants constructing tours probabilistically based on
    pheromone trails (historical success) and heuristic information
    (edge distances). It's naturally suited for graph-based routing problems.
    
    **When to use:**
    - Traveling Salesman Problem (TSP)
    - Vehicle routing problems
    - Network routing
    - Any problem that can be modeled as finding paths in a graph
    
    **How ACO works:**
    1. Initialize pheromone trails on all edges
    2. Each ant constructs a complete tour:
       - At each step, choose next city probabilistically
       - Probability depends on pheromone (τ^α) and distance (η^β)
    3. After all ants finish, update pheromones:
       - Evaporate existing pheromone: τ = (1-ρ)τ
       - Deposit new pheromone on good paths: τ += Q/tour_length
    4. Repeat until convergence or max iterations
    
    **Parameter tuning:**
    - alpha (pheromone importance): Higher = ants follow trails more
    - beta (heuristic importance): Higher = ants prefer short edges
    - evaporation_rate: Higher = faster adaptation, lower = more exploitation
    - Typical good values: alpha=1, beta=2-5, evaporation=0.5
    
    **2-opt local search:**
    - Improves each ant's tour by reversing segments
    - Significantly improves solution quality
    - Recommended to keep enabled (default)
    
    Args:
        distance_matrix: N×N matrix where element [i][j] is distance from i to j.
        n_ants: Number of ants (10-100). Default: 50.
        max_iterations: Max iterations (100-2000). Default: 500.
        alpha: Pheromone importance (0.5-2.0). Default: 1.0.
        beta: Heuristic importance (1.0-5.0). Default: 2.0.
        evaporation_rate: Evaporation rate (0.1-0.9). Default: 0.5.
        q: Pheromone deposit factor (0.1-10.0). Default: 1.0.
        initial_pheromone: Initial pheromone (0.01-1.0). Default: 0.1.
        local_search: Apply 2-opt. Default: True.
    
    Returns:
        Dict containing:
            - status (str): "success" or "error"
            - best_tour (List[int]): Best route found (city indices)
            - best_fitness (float): Tour length of best solution
            - n_cities (int): Number of cities
            - convergence_history (List[float]): Best fitness per iteration
            - metrics (Dict[str, Any]): When known_optimal is provided
                - Optimality gap: Difference from known optimal
                - Gap percentage: (solution - optimal) / optimal * 100
                - Performance rating: excellent/good/acceptable/poor
    
    Example:
        >>> # Solve 10-city TSP
        >>> distances = generate_distance_matrix(10)  # Your distance matrix
        >>> result = aco_tool(
        ...     distance_matrix=distances,
        ...     n_ants=30,
        ...     max_iterations=500,
        ...     beta=3.0,  # Emphasize short edges
        ...     local_search=True
        ... )
        >>> print(f"Best tour: {result['best_tour']}")
        >>> print(f"Tour length: {result['best_fitness']}")
    """
    try:
        dist_matrix = np.array(distance_matrix)
        
        if len(dist_matrix.shape) != 2:
            return {"status": "error", "message": "distance_matrix must be a 2D array"}
        if dist_matrix.shape[0] != dist_matrix.shape[1]:
            return {"status": "error", "message": "distance_matrix must be square"}
        
        n_cities = dist_matrix.shape[0]
        
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
        
        best_tour, best_fitness = aco.fit(dist_matrix)
        
        metrics = calculate_tsp_metrics(
            best_tour, best_fitness, dist_matrix,
            known_optimal=known_optimal,
            convergence_history=aco._history
        )
        
        result = {
            "status": "success",
            "message": f"ACO optimization completed for {n_cities}-city TSP",
            "best_tour": best_tour,
            "best_fitness": float(best_fitness),
            "n_cities": n_cities,
            "n_ants": n_ants,
            "iterations_run": max_iterations,
            "local_search_applied": local_search,
            "convergence_history": aco._history[-20:],
            "metrics": metrics
        }
        
        if "performance_rating" in metrics:
            result["message"] = f"ACO optimization completed. Tour length: {best_fitness:.2f}, Gap: {metrics['optimality_gap_percentage']:.2f}% ({metrics['performance_rating']})"
        
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}