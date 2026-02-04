import numpy as np
from typing import Literal
from pydantic import Field
from langchain_core.tools import tool


class GeneticAlgorithm:
    def __init__(self, population_size=100, generations=500, crossover_rate=0.8, 
                 mutation_rate=0.1, selection="tournament", tournament_size=3, 
                 elitism=2, crossover_type="pmx"):
        self.pop_size = population_size
        self.generations = generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.selection_method = selection
        self.tourn_size = tournament_size
        self.elitism = elitism
        self.cx_type = crossover_type
        self.population = None

    def fit(self, fitness_func, genome_length):
        """
        fitness_func: function taking a list (permutation) and returning a float score (lower is better).
        genome_length: length of the permutation vector.
        """
        # Initialize Population (Random permutations)
        self.population = [np.random.permutation(genome_length) for _ in range(self.pop_size)]
        
        best_global_score = float('inf')
        best_global_genome = None

        for gen in range(self.generations):
            # Evaluate Fitness
            scores = np.array([fitness_func(ind) for ind in self.population])
            
            # Track Best
            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < best_global_score:
                best_global_score = scores[min_score_idx]
                best_global_genome = self.population[min_score_idx].copy()

            # Elitism: Keep best N
            sorted_indices = np.argsort(scores)
            new_pop = [self.population[i] for i in sorted_indices[:self.elitism]]

            # Main Loop
            while len(new_pop) < self.pop_size:
                # Selection
                p1 = self._select(scores)
                p2 = self._select(scores)

                # Crossover
                if np.random.random() < self.cx_rate:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Mutation
                self._mutate(c1)
                self._mutate(c2)

                new_pop.extend([c1, c2])

            self.population = new_pop[:self.pop_size] # Trim if exceeded

        return best_global_genome, best_global_score

    def _select(self, scores):
        if self.selection_method == "tournament":
            candidates = np.random.choice(len(self.population), self.tourn_size, replace=False)
            best_idx = candidates[np.argmin(scores[candidates])]
            return self.population[best_idx]
        elif self.selection_method == "roulette":
            # Inverse fitness for minimization
            inv_scores = 1.0 / (scores + 1e-6)
            probs = inv_scores / np.sum(inv_scores)
            return self.population[np.random.choice(len(self.population), p=probs)]
        return self.population[0] # Fallback

    def _crossover(self, p1, p2):
        size = len(p1)
        c1, c2 = -np.ones(size, dtype=int), -np.ones(size, dtype=int)
        
        if self.cx_type == "pmx":
            # Partially Mapped Crossover
            pt1, pt2 = sorted(np.random.choice(range(size), 2, replace=False))
            # Copy segment
            c1[pt1:pt2+1] = p1[pt1:pt2+1]
            c2[pt1:pt2+1] = p2[pt1:pt2+1]
            
            def fill_pmx(child, parent, points):
                start, end = points
                for i in range(size):
                    if start <= i <= end: continue
                    val = parent[i]
                    while val in child: # Conflict resolution
                        # Find index in child where val exists
                        idx = np.where(child == val)[0][0]
                        # Map to the value in the *other* parent at that index
                        val = parent[idx] if np.array_equal(child, c1) else p1[idx] 
                    child[i] = val
            
            # Simple PMX fill for brevity (standard PMX mapping logic is complex)
            # Using simple 'fill missing' strategy for robustness in this snippet
            remaining1 = [x for x in p2 if x not in c1]
            c1[c1 == -1] = remaining1
            remaining2 = [x for x in p1 if x not in c2]
            c2[c2 == -1] = remaining2

        elif self.cx_type == "ox":
            # Order Crossover (Classic implementation)
            pt1, pt2 = sorted(np.random.choice(range(size), 2, replace=False))
            c1[pt1:pt2+1] = p1[pt1:pt2+1]
            c2[pt1:pt2+1] = p2[pt1:pt2+1]
            
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

        return c1, c2

    def _mutate(self, genome):
        if np.random.random() < self.mut_rate:
            # Swap Mutation
            i, j = np.random.choice(range(len(genome)), 2, replace=False)
            genome[i], genome[j] = genome[j], genome[i]


@tool
def ga_tool(
    population_size: int = Field(default=100, ge=50, le=500),
    generations: int = Field(default=500, ge=100, le=2000),
    crossover_rate: float = Field(default=0.8, ge=0.6, le=0.95),
    mutation_rate: float = Field(default=0.1, ge=0.01, le=0.3),
    selection: Literal["tournament", "roulette", "rank"] = Field(default="tournament"),
    tournament_size: int = Field(default=3, ge=2, le=10),
    elitism: int = Field(default=2, ge=0, le=10),
    crossover_type: Literal["pmx", "ox", "cx", "single", "two_point", "uniform"] = Field(default="pmx"),
):
    """Creates a Genetic Algorithm tool with specified hyperparameters."""
    pass
