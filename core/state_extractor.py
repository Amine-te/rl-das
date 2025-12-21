"""
Minimal state extractor for RL-DAS.

Simplified to 13 essential features:
- 5 LA features (landscape)
- 4 current algorithm (one-hot)
- 4 stagnation counters (per algorithm)
"""

from typing import Any, List, Tuple
import numpy as np


class StateExtractor:
    """
    Extract minimal state representation for algorithm selection.
    
    State dimension: 13 features
    - 5 LA: cost, budget, diversity, improvement_potential, convergence
    - 4 algo: one-hot encoding of current algorithm
    - 4 stagnation: intervals since each algorithm improved
    """
    
    NUM_LA_FEATURES = 5
    
    def __init__(self, num_algorithms: int):
        """
        Initialize state extractor.
        
        Args:
            num_algorithms: Number of algorithms in the pool (typically 4)
        """
        self.num_algorithms = num_algorithms
        self.stagnation_counters = [0] * num_algorithms
        
    @property
    def state_dim(self) -> int:
        """Total dimension of state vector."""
        # 5 LA + 4 algo onehot + 4 stagnation = 13
        return self.NUM_LA_FEATURES + self.num_algorithms + self.num_algorithms
    
    def reset(self):
        """Reset stagnation counters for new episode."""
        self.stagnation_counters = [0] * self.num_algorithms
    
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
            current_algo_idx: Index of currently running algorithm (None if first step)
            
        Returns:
            State vector of dimension 13
        """
        # LA features (5)
        la_features = self.extract_la_features(
            problem, current_best_cost, population, current_fes, max_fes
        )
        
        # Current algorithm one-hot (4)
        algo_onehot = np.zeros(self.num_algorithms, dtype=np.float32)
        if current_algo_idx is not None:
            algo_onehot[current_algo_idx] = 1.0
        
        # Stagnation counters (4), normalized to [0,1]
        # Cap at 10 intervals to prevent unbounded growth
        stagnation = np.array([
            min(self.stagnation_counters[i] / 10.0, 1.0)
            for i in range(self.num_algorithms)
        ], dtype=np.float32)
        
        # Concatenate all features
        state = np.concatenate([la_features, algo_onehot, stagnation])
        
        return state
    
    def update_stagnation(self, algo_idx: int, improved: bool):
        """
        Update stagnation counters after algorithm execution.
        
        Args:
            algo_idx: Index of algorithm that was executed
            improved: Whether the algorithm improved the best cost
        """
        if algo_idx < 0 or algo_idx >= self.num_algorithms:
            return
        
        # Reset counter for executed algorithm if it improved
        if improved:
            self.stagnation_counters[algo_idx] = 0
        else:
            self.stagnation_counters[algo_idx] += 1
        
        # Increment counters for all other algorithms
        for i in range(self.num_algorithms):
            if i != algo_idx:
                self.stagnation_counters[i] += 1
