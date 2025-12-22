"""
State extractor for RL-DAS with Stick-with-Winner behavior.

Simplified state focused on switching decision:
- 5 LA features (landscape)
- 1 last_improved flag
- 4 current algorithm (one-hot)
- 4 algorithm success rates
- 4 algorithm freshness

Total: 18 features
"""

from typing import Any, List, Tuple
import numpy as np


class StateExtractor:
    """
    Extract state representation for algorithm selection.
    
    State dimension: 18 features
    - 5 LA: cost, budget, diversity, improvement_potential, convergence
    - 1 last_improved: binary flag if last step improved
    - 4 current_algo: one-hot encoding
    - 4 success_rates: historical success rate per algorithm
    - 4 freshness: how long since each algorithm was used
    """
    
    NUM_LA_FEATURES = 5
    
    def __init__(self, num_algorithms: int):
        """
        Initialize state extractor.
        
        Args:
            num_algorithms: Number of algorithms in the pool (typically 4)
        """
        self.num_algorithms = num_algorithms
        self.last_improved = False
        self.current_algo_idx = None
        
        # Algorithm tracking
        self.algo_attempts = np.zeros(num_algorithms)
        self.algo_successes = np.zeros(num_algorithms)
        self.algo_last_used = np.full(num_algorithms, -10)
        self.current_step = 0
        
    @property
    def state_dim(self) -> int:
        """Total dimension of state vector."""
        # 5 LA + 1 last_improved + 4 current + 4 success + 4 freshness = 18
        return self.NUM_LA_FEATURES + 1 + 3 * self.num_algorithms
    
    def reset(self):
        """Reset internal state for new episode."""
        self.last_improved = False
        self.current_algo_idx = None
        self.algo_attempts = np.zeros(self.num_algorithms)
        self.algo_successes = np.zeros(self.num_algorithms)
        self.algo_last_used = np.full(self.num_algorithms, -10)
        self.current_step = 0
    
    def get_success_rates(self) -> np.ndarray:
        """Get success rate for each algorithm."""
        rates = np.zeros(self.num_algorithms, dtype=np.float32)
        for i in range(self.num_algorithms):
            if self.algo_attempts[i] > 0:
                rates[i] = self.algo_successes[i] / self.algo_attempts[i]
            else:
                rates[i] = 0.5  # Unknown = neutral
        return rates
    
    def get_freshness(self) -> np.ndarray:
        """Get freshness score for each algorithm."""
        freshness = np.zeros(self.num_algorithms, dtype=np.float32)
        for i in range(self.num_algorithms):
            steps_since = self.current_step - self.algo_last_used[i]
            freshness[i] = min(steps_since / 10.0, 1.0)
        return freshness
    
    def extract_la_features(
        self,
        problem: Any,
        current_best_cost: float,
        population: List[Tuple[Any, float]],
        current_fes: int,
        max_fes: int
    ) -> np.ndarray:
        """
        Extract 5 essential landscape analysis features.
        
        Returns:
            numpy array of 5 LA features, all in [0, 1]
        """
        features = np.zeros(self.NUM_LA_FEATURES, dtype=np.float32)
        
        # Feature 1: Normalized cost [0,1]
        features[0] = problem.normalize_cost(current_best_cost)
        
        # Feature 2: Budget consumed [0,1]
        features[1] = current_fes / max_fes
        
        # Feature 3: Diversity (coefficient of variation of population costs)
        if len(population) > 1:
            costs = np.array([c for _, c in population])
            mean_cost = np.mean(costs)
            if mean_cost > 1e-9:
                cv = np.std(costs) / mean_cost
                features[2] = np.clip(cv, 0.0, 1.0)
        
        # Feature 4: Improvement potential (gap to lower bound)
        lb, ub = problem.get_cost_bounds()
        if ub > lb:
            gap = (current_best_cost - lb) / (ub - lb)
            features[3] = np.clip(gap, 0.0, 1.0)
        
        # Feature 5: Convergence (inverse of diversity)
        features[4] = 1.0 - features[2]
        
        return features
    
    def extract_state(
        self,
        problem: Any,
        current_best_solution: Any,
        current_best_cost: float,
        population: List[Tuple[Any, float]],
        current_fes: int,
        max_fes: int,
        current_algo_idx: int = None
    ) -> np.ndarray:
        """
        Extract complete state vector.
        
        Args:
            problem: Problem instance
            current_best_solution: Current best solution
            current_best_cost: Cost of current best
            population: List of (solution, cost) tuples
            current_fes: Current function evaluations used
            max_fes: Maximum function evaluations budget
            current_algo_idx: Index of currently running algorithm
            
        Returns:
            State vector of dimension 18
        """
        # LA features (5)
        la_features = self.extract_la_features(
            problem, current_best_cost, population, current_fes, max_fes
        )
        
        # Last improved flag (1)
        last_improved = np.array([1.0 if self.last_improved else 0.0], dtype=np.float32)
        
        # Current algorithm one-hot (4)
        current_algo = np.zeros(self.num_algorithms, dtype=np.float32)
        if current_algo_idx is not None and 0 <= current_algo_idx < self.num_algorithms:
            current_algo[current_algo_idx] = 1.0
        
        # Success rates (4)
        success_rates = self.get_success_rates()
        
        # Freshness (4)
        freshness = self.get_freshness()
        
        # Concatenate all features (Dim 18)
        state = np.concatenate([la_features, last_improved, current_algo, success_rates, freshness])
        
        return state
    
    def update_stagnation(self, algo_idx: int, improved: bool, improvement_magnitude: float = 0.0):
        """
        Update tracking after algorithm execution.
        
        Args:
            algo_idx: Index of algorithm that was executed
            improved: Whether the algorithm improved the best cost
            improvement_magnitude: Not used in this version
        """
        if algo_idx < 0 or algo_idx >= self.num_algorithms:
            return
        
        self.current_step += 1
        self.last_improved = improved
        self.current_algo_idx = algo_idx
        
        # Update algorithm tracking
        self.algo_attempts[algo_idx] += 1
        self.algo_last_used[algo_idx] = self.current_step
        
        if improved:
            self.algo_successes[algo_idx] += 1
