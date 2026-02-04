import numpy as np
from pydantic import Field
from langchain_core.tools import tool


class PSO:
    def __init__(self, n_particles=50, max_iterations=500, w=0.7, c1=1.5, c2=1.5, 
                 w_decay=True, velocity_clamp=0.5):
        self.n_particles = n_particles
        self.max_iter = max_iterations
        self.w_start = w
        self.c1 = c1
        self.c2 = c2
        self.w_decay = w_decay
        self.v_clamp_frac = velocity_clamp
        
        self.global_best_pos = None
        self.global_best_score = float('inf')

    def optimize(self, objective_func, bounds):
        """
        bounds: list of tuples [(min, max), (min, max), ...] for each dimension
        """
        bounds = np.array(bounds)
        n_dim = len(bounds)
        range_width = bounds[:, 1] - bounds[:, 0]
        v_max = range_width * self.v_clamp_frac
        v_min = -v_max

        # Initialize Particles
        # Positions: random within bounds
        pos = bounds[:, 0] + np.random.rand(self.n_particles, n_dim) * range_width
        # Velocities: random fraction of range
        vel = (np.random.rand(self.n_particles, n_dim) - 0.5) * range_width * 0.1
        
        # Best positions
        p_best_pos = pos.copy()
        p_best_scores = np.full(self.n_particles, float('inf'))

        # Evaluate initial population
        for i in range(self.n_particles):
            score = objective_func(pos[i])
            p_best_scores[i] = score
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_pos = pos[i].copy()

        # Optimization Loop
        w = self.w_start
        for t in range(self.max_iter):
            # Linearly decay inertia weight
            if self.w_decay:
                w = self.w_start - (self.w_start - 0.4) * (t / self.max_iter)

            # Random coefficients
            r1 = np.random.rand(self.n_particles, n_dim)
            r2 = np.random.rand(self.n_particles, n_dim)

            # Update Velocity
            # v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
            cognitive = self.c1 * r1 * (p_best_pos - pos)
            social = self.c2 * r2 * (self.global_best_pos - pos)
            vel = w * vel + cognitive + social

            # Clamp Velocity
            vel = np.clip(vel, v_min, v_max)

            # Update Position
            pos += vel

            # Enforce Bounds (Reflective or Clipping)
            # Here we clip to stay valid
            pos = np.clip(pos, bounds[:, 0], bounds[:, 1])

            # Evaluate
            for i in range(self.n_particles):
                score = objective_func(pos[i])
                
                if score < p_best_scores[i]:
                    p_best_scores[i] = score
                    p_best_pos[i] = pos[i].copy()
                    
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_pos = pos[i].copy()

        return self.global_best_pos, self.global_best_score


@tool
def pso_tool(
    n_particles: int = Field(default=50, ge=20, le=200),
    max_iterations: int = Field(default=500, ge=100, le=2000),
    w: float = Field(default=0.7, ge=0.4, le=0.9, description="Inertia weight"),
    c1: float = Field(default=1.5, ge=1.0, le=2.5, description="Cognitive coefficient"),
    c2: float = Field(default=1.5, ge=1.0, le=2.5, description="Social coefficient"),
    w_decay: bool = Field(default=True, description="Linearly decrease inertia weight"),
    velocity_clamp: float = Field(default=0.5, ge=0.1, le=1.0, description="Fraction of search range"),
):
    """Creates a Particle Swarm Optimization (PSO) tool with specified hyperparameters."""
    pass