import numpy as np
from pydantic import Field
from langchain_core.tools import tool


class AntColonyOptimization:
    def __init__(self, n_ants=50, max_iterations=500, alpha=1.0, beta=2.0, 
                 evaporation_rate=0.5, q=1.0, initial_pheromone=0.1, local_search=True):
        self.n_ants = n_ants
        self.max_iter = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = evaporation_rate
        self.Q = q
        self.init_phero = initial_pheromone
        self.do_local_search = local_search

    def fit(self, distance_matrix):
        n_cities = len(distance_matrix)
        
        # Initialize Pheromones
        pheromone = np.full((n_cities, n_cities), self.init_phero)
        
        # Pre-calculate Heuristic (Inverse Distance)
        # Add small epsilon to avoid divide by zero
        heuristic = 1.0 / (distance_matrix + 1e-10)
        np.fill_diagonal(heuristic, 0)

        best_path = None
        best_dist = float('inf')

        for iteration in range(self.max_iter):
            all_paths = []
            all_dists = []

            # Construct solutions for each ant
            for k in range(self.n_ants):
                path = [np.random.randint(n_cities)]
                visited = set(path)
                
                for _ in range(n_cities - 1):
                    current = path[-1]
                    
                    # Calculate Probabilities
                    # P_ij = (tau^alpha) * (eta^beta)
                    probs = np.zeros(n_cities)
                    unvisited = [node for node in range(n_cities) if node not in visited]
                    
                    den = 0
                    for node in unvisited:
                        tau = pheromone[current, node] ** self.alpha
                        eta = heuristic[current, node] ** self.beta
                        val = tau * eta
                        probs[node] = val
                        den += val
                        
                    if den == 0: # Should not happen unless disconnected
                        next_node = np.random.choice(unvisited)
                    else:
                        probs = probs / den
                        # Select next node based on probability
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

            # Update Pheromones
            # 1. Evaporation
            pheromone *= (1 - self.rho)
            
            # 2. Deposit
            for i in range(self.n_ants):
                path = all_paths[i]
                dist = all_dists[i]
                deposit_amount = self.Q / dist
                for j in range(n_cities - 1):
                    u, v = path[j], path[j+1]
                    pheromone[u, v] += deposit_amount
                    pheromone[v, u] += deposit_amount # Symmetric TSP
                # Close the loop
                u, v = path[-1], path[0]
                pheromone[u, v] += deposit_amount
                pheromone[v, u] += deposit_amount

        return best_path, best_dist

    def _path_length(self, path, dist_matrix):
        total = 0
        for i in range(len(path) - 1):
            total += dist_matrix[path[i], path[i+1]]
        total += dist_matrix[path[-1], path[0]] # Return to start
        return total

    def _two_opt(self, path, dist_matrix):
        """Simple First-Improvement 2-opt"""
        best_path = path
        improved = True
        n = len(path)
        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n):
                    if j - i == 1: continue 
                    
                    # Check if swap improves distance
                    # Current edges: (i-1, i) and (j, j+1)
                    # New edges: (i-1, j) and (i, j+1)
                    # Note: handling circular index for j+1 is needed for full 2-opt, 
                    # simplified here for linear path segments
                    
                    p = best_path
                    d_current = dist_matrix[p[i-1], p[i]] + dist_matrix[p[j], p[(j+1)%n]]
                    d_new = dist_matrix[p[i-1], p[j]] + dist_matrix[p[i], p[(j+1)%n]]

                    if d_new < d_current:
                        # Reverse segment from i to j
                        best_path = p[:i] + p[i:j+1][::-1] + p[j+1:]
                        improved = True
        return best_path


@tool
def aco_tool(
    n_ants: int = Field(default=50, ge=10, le=100),
    max_iterations: int = Field(default=500, ge=100, le=2000),
    alpha: float = Field(default=1.0, ge=0.5, le=2.0, description="Pheromone importance"),
    beta: float = Field(default=2.0, ge=1.0, le=5.0, description="Heuristic importance"),
    evaporation_rate: float = Field(default=0.5, ge=0.1, le=0.9),
    q: float = Field(default=1.0, description="Pheromone deposit factor"),
    initial_pheromone: float = Field(default=0.1),
    local_search: bool = Field(default=True, description="Apply 2-opt improvement"),
):
    """Creates an Ant Colony Optimization tool with specified hyperparameters."""
    pass