"""
State extractor for RL-DAS with Improvement Magnitude Tracking.

Features:
- 5 LA features (landscape analysis)
- 4 Activity levels (recency-weighted algorithm usage)
- 4 Stagnation counters (intervals since improvement)
- 4 Credibility scores (success rate)
- 4 Recent improvement magnitudes (HOW MUCH each algo improved)

Total: 21 features
"""

from typing import Any, List, Tuple
import numpy as np


class StateExtractor:
    """
    Extract state representation for algorithm selection.
    
    State dimension: 21 features
    - 5 LA: cost, budget, diversity, improvement_potential, convergence
    - 4 activity: recency-weighted algorithm usage
    - 4 stagnation: intervals since each algorithm improved
    - 4 credibility: success rate per algorithm
    - 4 improvement_magnitude: how much each algo improved (decaying)
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
        self.activity_levels = np.zeros(num_algorithms, dtype=np.float32)
        self.activity_decay = 0.6
        self.credibility_scores = np.full(num_algorithms, 0.5, dtype=np.float32)  # Start neutral
        self.recent_improvements = np.zeros(num_algorithms, dtype=np.float32)  # NEW: improvement magnitudes
        
    @property
    def state_dim(self) -> int:
        """Total dimension of state vector."""
        # 5 LA + 4 Activity + 4 Stagnation + 4 Credibility + 4 Improvements = 21
        return self.NUM_LA_FEATURES + 4 * self.num_algorithms
    
    def reset(self):
        """Reset internal state for new episode."""
        self.stagnation_counters = [0] * self.num_algorithms
        self.activity_levels = np.zeros(self.num_algorithms, dtype=np.float32)
        self.credibility_scores = np.full(self.num_algorithms, 0.5, dtype=np.float32)
        self.recent_improvements = np.zeros(self.num_algorithms, dtype=np.float32)
    
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
            State vector of dimension 21
        """
        # LA features (5)
        la_features = self.extract_la_features(
            problem, current_best_cost, population, current_fes, max_fes
        )
        
        # Current algorithm memory (Activity Vector)
        # Update activity: decay all, set current to 1.0
        if current_algo_idx is not None:
            self.activity_levels *= self.activity_decay
            self.activity_levels[current_algo_idx] = 1.0
            
        # Copy to features
        algo_activity = self.activity_levels.copy()
        
        # Stagnation counters (4), normalized to [0,1]
        # Cap at 10 intervals to prevent unbounded growth
        stagnation = np.array([
            min(self.stagnation_counters[i] / 10.0, 1.0)
            for i in range(self.num_algorithms)
        ], dtype=np.float32)
        
        # Credibility scores (4)
        credibility = self.credibility_scores.copy()
        
        # Recent improvement magnitudes (4) - already in [0,1]
        improvements = self.recent_improvements.copy()
        
        # Concatenate all features (Dim 21)
        state = np.concatenate([la_features, algo_activity, stagnation, credibility, improvements])
        
        return state
    
    def update_stagnation(self, algo_idx: int, improved: bool, improvement_magnitude: float = 0.0):
        """
        Update stagnation counters, credibility scores, and improvement magnitudes.
        
        Args:
            algo_idx: Index of algorithm that was executed
            improved: Whether the algorithm improved the best cost
            improvement_magnitude: Normalized improvement (prev - curr) / initial_cost
        """
        if algo_idx < 0 or algo_idx >= self.num_algorithms:
            return
        
        # Decay all improvement magnitudes (memory fades)
        self.recent_improvements *= 0.7
        
        if improved:
            # Track HOW MUCH it improved (scaled and clipped)
            # Multiply by 10 to make small improvements more visible, clip at 1.0
            self.recent_improvements[algo_idx] = min(improvement_magnitude * 10, 1.0)
            
            # Boost credibility (cap at 1.0)
            self.credibility_scores[algo_idx] = min(self.credibility_scores[algo_idx] + 0.15, 1.0)
            
            # Reset stagnation
            self.stagnation_counters[algo_idx] = 0
        else:
            # Decay improvement memory faster on failure
            self.recent_improvements[algo_idx] *= 0.5
            
            # Penalize credibility - faster decay for repeated failures
            decay = 0.15 + 0.03 * min(self.stagnation_counters[algo_idx], 5)
            self.credibility_scores[algo_idx] = max(self.credibility_scores[algo_idx] - decay, 0.0)
            
            # Increment stagnation
            self.stagnation_counters[algo_idx] += 1
        
        # Increment counters for all other algorithms (they're getting "stale")
        for i in range(self.num_algorithms):
            if i != algo_idx:
                self.stagnation_counters[i] += 1
