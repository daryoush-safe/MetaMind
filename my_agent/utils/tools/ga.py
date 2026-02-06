import numpy as np
import time
from typing import List, Optional, Any, Dict, Literal, Callable, Tuple
from pydantic import Field, BaseModel
from langchain_core.tools import tool


def calculate_tsp_metrics(best_tour: np.ndarray, best_fitness: float, 
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
        best_tour: Best tour found
        best_fitness: Length of best tour
        distance_matrix: Distance matrix
        known_optimal: Known optimal solution (if available)
        convergence_history: History of best fitness values
        computation_time: Wall-clock time in seconds
        multi_run_results: Best fitness from multiple independent runs
    
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
        self.population = [np.random.permutation(genome_length) for _ in range(self.pop_size)]
        
        best_global_score = float('inf')
        best_global_genome = None
        self._history = []

        for gen in range(self.generations):
            scores = np.array([fitness_func(ind) for ind in self.population])
            
            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < best_global_score:
                best_global_score = scores[min_score_idx]
                best_global_genome = self.population[min_score_idx].copy()
            
            self._history.append(best_global_score)

            sorted_indices = np.argsort(scores)
            new_pop = [self.population[i].copy() for i in sorted_indices[:self.elitism]]

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
            i, j = np.random.choice(range(len(genome)), 2, replace=False)
            genome[i], genome[j] = genome[j], genome[i]


def _calculate_tour_length(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    """Calculate total tour length for TSP."""
    total = 0.0
    for i in range(len(tour) - 1):
        total += distance_matrix[tour[i], tour[i+1]]
    total += distance_matrix[tour[-1], tour[0]]
    return total


class GAInput(BaseModel):
    distance_matrix: List[List[float]] = Field(description="Distance/cost matrix as a 2D list")
    population_size: int = Field(default=100, ge=50, le=500, description="Number of individuals in the population")
    generations: int = Field(default=500, ge=100, le=2000, description="Number of evolutionary generations")
    crossover_rate: float = Field(default=0.8, ge=0.6, le=0.95, description="Probability of crossover")
    mutation_rate: float = Field(default=0.1, ge=0.01, le=0.3, description="Probability of mutation")
    selection: Literal["tournament", "roulette", "rank"] = Field(default="tournament", description="Selection method")
    tournament_size: int = Field(default=3, ge=2, le=10, description="Tournament size")
    elitism: int = Field(default=2, ge=0, le=10, description="Number of best individuals to preserve")
    crossover_type: Literal["pmx", "ox", "cx", "single", "two_point", "uniform"] = Field(default="pmx", description="Crossover operator type")
    known_optimal: Optional[float] = Field(default=None, description="Known optimal tour length for computing optimality gap")
    n_runs: int = Field(default=1, ge=1, le=10, description="Number of independent runs for computing success rate statistics")


@tool(args_schema=GAInput)
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
    known_optimal: Optional[float] = Field(default=None, description="Known optimal tour length for computing optimality gap"),
    n_runs: int = Field(default=1, ge=1, le=10, description="Number of independent runs for success rate statistics")
) -> Dict[str, Any]:
    """
    Solve combinatorial optimization problems using Genetic Algorithm.
    
    GA evolves a population of candidate solutions through selection, crossover,
    and mutation. This implementation is optimized for permutation-based problems
    like the Traveling Salesman Problem (TSP).
    
    Use for: TSP, Vehicle routing problems, scheduling, Assignment problems
    
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
            - convergence_history: Best fitness per generation
            - metrics: Comprehensive TSP metrics
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
        
        multi_run_results = []
        best_overall_tour = None
        best_overall_fitness = float('inf')
        best_overall_history = None
        total_time = 0.0
        
        for run_idx in range(n_runs):
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
            
            start_time = time.time()
            best_tour, best_fitness = ga.fit(fitness_func, n_cities)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            multi_run_results.append(best_fitness)
            
            if best_fitness < best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_tour = best_tour
                best_overall_history = ga._history
        
        metrics = calculate_tsp_metrics(
            best_overall_tour, best_overall_fitness, dist_matrix,
            known_optimal=known_optimal,
            convergence_history=best_overall_history,
            computation_time=total_time,
            multi_run_results=multi_run_results if n_runs > 1 else None
        )
        
        result = {
            "status": "success",
            "message": f"GA optimization completed for {n_cities}-city problem",
            "best_tour": best_overall_tour.tolist(),
            "best_fitness": float(best_overall_fitness),
            "n_cities": n_cities,
            "generations_run": generations,
            "computation_time_seconds": float(total_time),
            "convergence_history": best_overall_history[-20:] if best_overall_history else [],
            "metrics": metrics
        }
        
        if "performance_rating" in metrics:
            result["message"] = f"GA completed. Tour: {best_overall_fitness:.2f}, Gap: {metrics['optimality_gap_percentage']:.2f}% ({metrics['performance_rating']}), Time: {total_time:.2f}s"
        
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}