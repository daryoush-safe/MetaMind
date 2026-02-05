import numpy as np
from typing import List, Optional, Any, Dict, Literal, Callable, Tuple
from pydantic import Field
from langchain_core.tools import tool


def calculate_tsp_metrics(best_tour: np.ndarray, best_fitness: float, 
                          distance_matrix: np.ndarray,
                          known_optimal: Optional[float] = None,
                          convergence_history: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Calculate TSP optimization metrics.
    
    Args:
        best_tour: Best tour found
        best_fitness: Length of best tour
        distance_matrix: Distance matrix
        known_optimal: Known optimal solution (if available)
        convergence_history: History of best fitness values
    
    Returns:
        Dictionary with optimization metrics
    """
    n_cities = len(best_tour)
    
    metrics = {
        "tour_length": float(best_fitness),
        "n_cities": n_cities,
        "avg_edge_length": float(best_fitness / n_cities),
    }
    
    # Calculate tour statistics
    edge_lengths = []
    for i in range(n_cities):
        j = (i + 1) % n_cities
        edge_lengths.append(distance_matrix[best_tour[i], best_tour[j]])
    
    metrics["min_edge"] = float(np.min(edge_lengths))
    metrics["max_edge"] = float(np.max(edge_lengths))
    metrics["edge_std"] = float(np.std(edge_lengths))
    
    # Optimality metrics if known optimal is provided
    if known_optimal is not None and known_optimal > 0:
        gap = best_fitness - known_optimal
        gap_percentage = (gap / known_optimal) * 100
        metrics["known_optimal"] = float(known_optimal)
        metrics["optimality_gap"] = float(gap)
        metrics["optimality_gap_percentage"] = float(gap_percentage)
        metrics["is_optimal"] = gap_percentage < 0.01  # Within 0.01%
        
        # Performance rating
        if gap_percentage < 1:
            metrics["performance_rating"] = "excellent"
        elif gap_percentage < 5:
            metrics["performance_rating"] = "good"
        elif gap_percentage < 10:
            metrics["performance_rating"] = "acceptable"
        else:
            metrics["performance_rating"] = "poor"
    
    # Convergence analysis
    if convergence_history and len(convergence_history) > 1:
        metrics["initial_fitness"] = float(convergence_history[0])
        metrics["improvement"] = float(convergence_history[0] - best_fitness)
        metrics["improvement_percentage"] = float((convergence_history[0] - best_fitness) / convergence_history[0] * 100)
        
        # Check for early convergence
        last_10 = convergence_history[-10:] if len(convergence_history) >= 10 else convergence_history
        if len(last_10) > 1:
            variance = np.var(last_10)
            metrics["convergence_variance"] = float(variance)
            metrics["converged"] = variance < 1e-6
    
    return metrics


class GeneticAlgorithm:
    """
    Genetic Algorithm for combinatorial optimization.
    
    Evolves a population of candidate solutions using selection, crossover,
    and mutation operators. Particularly effective for permutation-based
    problems like TSP, scheduling, and assignment.
    
    Attributes:
        pop_size (int): Population size.
        generations (int): Number of generations.
        cx_rate (float): Crossover probability.
        mut_rate (float): Mutation probability.
        selection_method (str): Selection strategy.
        tourn_size (int): Tournament size (if using tournament selection).
        elitism (int): Number of best individuals to preserve.
        cx_type (str): Crossover operator type.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 500,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        selection: str = "tournament",
        tournament_size: int = 3,
        elitism: int = 2,
        crossover_type: str = "pmx"
    ):
        self.pop_size = population_size
        self.generations = generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.selection_method = selection
        self.tourn_size = tournament_size
        self.elitism = elitism
        self.cx_type = crossover_type
        self.population: List[np.ndarray] = []
        self._history: List[float] = []

    def fit(self, fitness_func: Callable, genome_length: int) -> Tuple[np.ndarray, float]:
        """
        Run the genetic algorithm to minimize the fitness function.
        
        Args:
            fitness_func: Function that takes a permutation and returns fitness (lower is better).
            genome_length: Length of the permutation vector.
        
        Returns:
            Tuple[np.ndarray, float]: Best solution found and its fitness.
        """
        # Initialize population with random permutations
        self.population = [np.random.permutation(genome_length) for _ in range(self.pop_size)]
        
        best_global_score = float('inf')
        best_global_genome = None
        self._history = []

        for gen in range(self.generations):
            # Evaluate fitness
            scores = np.array([fitness_func(ind) for ind in self.population])
            
            # Track best
            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < best_global_score:
                best_global_score = scores[min_score_idx]
                best_global_genome = self.population[min_score_idx].copy()
            
            self._history.append(best_global_score)

            # Elitism: keep best N
            sorted_indices = np.argsort(scores)
            new_pop = [self.population[i].copy() for i in sorted_indices[:self.elitism]]

            # Create rest of population
            while len(new_pop) < self.pop_size:
                p1 = self._select(scores)
                p2 = self._select(scores)

                if np.random.random() < self.cx_rate:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                self._mutate(c1)
                self._mutate(c2)

                new_pop.extend([c1, c2])

            self.population = new_pop[:self.pop_size]

        return best_global_genome, best_global_score

    def _select(self, scores: np.ndarray) -> np.ndarray:
        """Select a parent using the configured selection method."""
        if self.selection_method == "tournament":
            candidates = np.random.choice(len(self.population), self.tourn_size, replace=False)
            best_idx = candidates[np.argmin(scores[candidates])]
            return self.population[best_idx].copy()
        elif self.selection_method == "roulette":
            inv_scores = 1.0 / (scores + 1e-6)
            probs = inv_scores / np.sum(inv_scores)
            return self.population[np.random.choice(len(self.population), p=probs)].copy()
        elif self.selection_method == "rank":
            ranks = np.argsort(np.argsort(scores)) + 1
            probs = (len(scores) - ranks + 1) / np.sum(np.arange(1, len(scores) + 1))
            return self.population[np.random.choice(len(self.population), p=probs)].copy()
        return self.population[0].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover between two parents."""
        size = len(p1)
        c1, c2 = -np.ones(size, dtype=int), -np.ones(size, dtype=int)
        
        if self.cx_type in ["pmx", "ox"]:
            pt1, pt2 = sorted(np.random.choice(range(size), 2, replace=False))
            c1[pt1:pt2+1] = p1[pt1:pt2+1]
            c2[pt1:pt2+1] = p2[pt1:pt2+1]
            
            if self.cx_type == "ox":
                # Order Crossover
                current_idx1 = (pt2 + 1) % size
                current_idx2 = (pt2 + 1) % size
                
                for i in range(size):
                    cand1 = p2[(pt2 + 1 + i) % size]
                    cand2 = p1[(pt2 + 1 + i) % size]
                    if cand1 not in c1:
                        c1[current_idx1] = cand1
                        current_idx1 = (current_idx1 + 1) % size
                    if cand2 not in c2:
                        c2[current_idx2] = cand2
                        current_idx2 = (current_idx2 + 1) % size
            else:
                # PMX - fill remaining with order from other parent
                remaining1 = [x for x in p2 if x not in c1]
                remaining2 = [x for x in p1 if x not in c2]
                j1, j2 = 0, 0
                for i in range(size):
                    if c1[i] == -1:
                        c1[i] = remaining1[j1]
                        j1 += 1
                    if c2[i] == -1:
                        c2[i] = remaining2[j2]
                        j2 += 1
        else:
            # Default to single-point crossover
            pt = np.random.randint(1, size)
            c1[:pt] = p1[:pt]
            remaining = [x for x in p2 if x not in c1[:pt]]
            c1[pt:] = remaining[:size-pt]
            c2[:pt] = p2[:pt]
            remaining = [x for x in p1 if x not in c2[:pt]]
            c2[pt:] = remaining[:size-pt]

        return c1, c2

    def _mutate(self, genome: np.ndarray) -> None:
        """Apply mutation to a genome."""
        if np.random.random() < self.mut_rate:
            # Swap mutation
            i, j = np.random.choice(range(len(genome)), 2, replace=False)
            genome[i], genome[j] = genome[j], genome[i]


def _calculate_tour_length(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    """Calculate total tour length for TSP."""
    total = 0.0
    for i in range(len(tour) - 1):
        total += distance_matrix[tour[i], tour[i+1]]
    total += distance_matrix[tour[-1], tour[0]]  # Return to start
    return total


@tool
def ga_tool(
    distance_matrix: List[List[float]] = Field(description="Distance/cost matrix as a 2D list"),
    population_size: int = Field(default=100, ge=50, le=500, description="Number of individuals in the population"),
    generations: int = Field(default=500, ge=100, le=2000, description="Number of evolutionary generations"),
    crossover_rate: float = Field(default=0.8, ge=0.6, le=0.95, description="Probability of crossover"),
    mutation_rate: float = Field(default=0.1, ge=0.01, le=0.3, description="Probability of mutation"),
    selection: Literal["tournament", "roulette", "rank"] = Field(default="tournament", description="Selection method"),
    tournament_size: int = Field(default=3, ge=2, le=10, description="Tournament size"),
    elitism: int = Field(default=2, ge=0, le=10, description="Number of best individuals to preserve"),
    crossover_type: Literal["pmx", "ox", "cx", "single", "two_point", "uniform"] = Field(default="pmx", description="Crossover operator type"),
    known_optimal: Optional[float] = Field(default=None, description="Known optimal tour length for computing optimality gap")
) -> Dict[str, Any]:
    """
    Solve combinatorial optimization problems using Genetic Algorithm.
    
    GA evolves a population of candidate solutions through selection, crossover,
    and mutation. This implementation is optimized for permutation-based problems
    like the Traveling Salesman Problem (TSP).
    
    **When to use:**
    - Traveling Salesman Problem (TSP)
    - Vehicle routing problems
    - Job shop scheduling
    - Assignment problems
    - Any problem where solutions are permutations
    
    **How GA works:**
    1. Initialize random population of permutations
    2. Evaluate fitness of each individual
    3. Select parents based on fitness
    4. Apply crossover to create offspring
    5. Apply mutation for diversity
    6. Repeat for specified generations
    
    **Selection methods:**
    - tournament: Select best from random subset (recommended)
    - roulette: Probability proportional to fitness
    - rank: Probability based on fitness ranking
    
    **Crossover types for permutations:**
    - pmx: Partially Mapped Crossover (preserves relative positions)
    - ox: Order Crossover (preserves relative order)
    
    **Parameter tuning:**
    - population_size: Larger = more exploration, slower
    - generations: More = better solutions, diminishing returns
    - crossover_rate: 0.8-0.9 typical, too low reduces exploration
    - mutation_rate: 0.05-0.2 typical, too high = random search
    - elitism: 1-5 preserves good solutions
    
    Args:
        distance_matrix: N×N matrix where element [i][j] is distance from i to j.
        population_size: Population size (50-500). Default: 100.
        generations: Number of generations (100-2000). Default: 500.
        crossover_rate: Crossover probability (0.6-0.95). Default: 0.8.
        mutation_rate: Mutation probability (0.01-0.3). Default: 0.1.
        selection: Selection method. Default: "tournament".
        tournament_size: Tournament size (2-10). Default: 3.
        elitism: Elite individuals (0-10). Default: 2.
        crossover_type: Crossover operator. Default: "pmx".
    
    Returns:
        Dict containing:
            - status (str): "success" or "error"
            - best_tour (List[int]): Best route found (city indices)
            - best_fitness (float): Tour length of best solution
            - n_cities (int): Number of cities
            - convergence_history (List[float]): Best fitness per generation
    
    Example:
        >>> # Solve 5-city TSP
        >>> distances = [
        ...     [0, 10, 15, 20, 25],
        ...     [10, 0, 35, 25, 30],
        ...     [15, 35, 0, 30, 20],
        ...     [20, 25, 30, 0, 15],
        ...     [25, 30, 20, 15, 0]
        ... ]
        >>> result = ga_tool(
        ...     distance_matrix=distances,
        ...     population_size=100,
        ...     generations=500
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
        
        def fitness_func(tour):
            return _calculate_tour_length(tour, dist_matrix)
        
        ga = GeneticAlgorithm(
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            selection=selection,
            tournament_size=tournament_size,
            elitism=elitism,
            crossover_type=crossover_type
        )
        
        best_tour, best_fitness = ga.fit(fitness_func, n_cities)
        
        # Calculate metrics
        metrics = calculate_tsp_metrics(
            best_tour, best_fitness, dist_matrix,
            known_optimal=known_optimal,
            convergence_history=ga._history
        )
        
        result = {
            "status": "success",
            "message": f"GA optimization completed for {n_cities}-city problem",
            "best_tour": best_tour.tolist(),
            "best_fitness": float(best_fitness),
            "n_cities": n_cities,
            "generations_run": generations,
            "convergence_history": ga._history[-20:],
            "metrics": metrics
        }
        
        # Add summary message based on metrics
        if "performance_rating" in metrics:
            result["message"] = f"GA optimization completed. Tour length: {best_fitness:.2f}, Gap: {metrics['optimality_gap_percentage']:.2f}% ({metrics['performance_rating']})"
        
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}