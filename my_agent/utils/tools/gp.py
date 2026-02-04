import numpy as np
import copy
import random
from typing import List
from pydantic import Field
from langchain_core.tools import tool


class GPNode:
    def __init__(self, val, children=None):
        self.val = val # Can be operator (+, *) or terminal (x, 1.5)
        self.children = children if children else []
        self.is_terminal = len(self.children) == 0

    def evaluate(self, x):
        if self.is_terminal:
            if self.val == 'x': return x
            return self.val # Constant
        
        args = [c.evaluate(x) for c in self.children]
        if self.val == '+': return args[0] + args[1]
        if self.val == '-': return args[0] - args[1]
        if self.val == '*': return args[0] * args[1]
        if self.val == '/': 
            return args[0] / (args[1] if abs(args[1]) > 1e-6 else 1) # Protected division
        if self.val == 'sin': return np.sin(args[0])
        if self.val == 'cos': return np.cos(args[0])
        return 0

    def size(self):
        return 1 + sum(c.size() for c in self.children)

class GeneticProgramming:
    def __init__(self, population_size=200, generations=50, max_depth=6, 
                 crossover_rate=0.9, mutation_rate=0.1, 
                 function_set=["+", "-", "*", "/"], terminal_set=["x", "constants"],
                 parsimony_coefficient=0.001):
        self.pop_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.funcs = function_set
        self.terms = terminal_set
        self.parsimony = parsimony_coefficient
        self.population = []

    def _random_tree(self, depth, type="full"):
        if depth == 0 or (type == "grow" and random.random() < 0.5):
            # Terminal
            term = random.choice(self.terms)
            val = random.uniform(-5, 5) if term == "constants" else 'x'
            return GPNode(val)
        else:
            # Function
            func = random.choice(self.funcs)
            arity = 1 if func in ['sin', 'cos', 'exp'] else 2
            children = [self._random_tree(depth - 1, type) for _ in range(arity)]
            return GPNode(func, children)

    def fit(self, X, y):
        # Initialize Population (Ramp Half-and-Half)
        self.population = []
        for _ in range(self.pop_size):
            depth = random.randint(2, self.max_depth)
            self.population.append(self._random_tree(depth, "grow"))

        best_prog = None
        best_error = float('inf')

        for gen in range(self.generations):
            scores = []
            for individual in self.population:
                # Calculate MSE
                preds = np.array([individual.evaluate(xi) for xi in X])
                mse = np.mean((y - preds) ** 2)
                # Parsimony pressure
                fitness = mse + self.parsimony * individual.size()
                scores.append(fitness)

                if mse < best_error:
                    best_error = mse
                    best_prog = copy.deepcopy(individual)

            # Tournament Selection
            new_pop = [best_prog] # Elitism (1)
            
            while len(new_pop) < self.pop_size:
                parent1 = self._tournament(scores)
                parent2 = self._tournament(scores)
                
                if random.random() < self.cx_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)
                
                if random.random() < self.mut_rate:
                    child = self._mutate(child)
                    
                new_pop.append(child)
            
            self.population = new_pop

        return best_prog

    def _tournament(self, scores):
        indices = np.random.choice(len(self.population), 3, replace=False)
        best_idx = indices[np.argmin([scores[i] for i in indices])]
        return self.population[best_idx]

    def _crossover(self, p1, p2):
        child = copy.deepcopy(p1)
        # Select random node in child to replace
        nodes = self._get_nodes(child)
        target = random.choice(nodes)
        # Select random subtree from p2
        source = random.choice(self._get_nodes(p2))
        
        # Swap content (simplified)
        target.val = source.val
        target.children = copy.deepcopy(source.children)
        return child

    def _mutate(self, p):
        child = copy.deepcopy(p)
        nodes = self._get_nodes(child)
        target = random.choice(nodes)
        # Replace with random subtree
        new_subtree = self._random_tree(2, "grow")
        target.val = new_subtree.val
        target.children = new_subtree.children
        return child

    def _get_nodes(self, node):
        nodes = [node]
        for c in node.children:
            nodes.extend(self._get_nodes(c))
        return nodes


@tool
def gp_tool(
    population_size: int = Field(default=200, ge=100, le=1000),
    generations: int = Field(default=50, ge=20, le=200),
    max_depth: int = Field(default=6, ge=3, le=10),
    crossover_rate: float = Field(default=0.9, ge=0.7, le=0.95),
    mutation_rate: float = Field(default=0.1, ge=0.05, le=0.2),
    function_set: List[str] = Field(default=["+", "-", "*", "/"]),
    terminal_set: List[str] = Field(default=["x", "constants"]),
    parsimony_coefficient: float = Field(default=0.001, ge=0, le=0.01),
):
    """Creates a Genetic Programming tool with specified hyperparameters."""
    pass