"""
State extractor for RL-DAS.

Extracts problem-agnostic state features for the RL agent:
- Landscape Analysis (LA) features: Problem characteristics
- Algorithm History (AH) features: Past algorithm performance
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


class StateExtractor:
    """
    Extract problem-agnostic state features for RL agent.
    
    Features are designed to be:
    - Normalized to [0, 1] range
    - Problem-size independent
    - Generalizable across problem types
    
    LA Features (9):
        1. best_cost_normalized: Current best cost [0,1]
        2. fdc: Fitness-Distance Correlation [0,1]
        3. gap_to_bound: Gap to lower bound [0,1]
        4. diversity: Population diversity [0,1]
        5. convergence: Convergence indicator [0,1]
        6. dispersion_diff: Elite vs population dispersion [0,1]
        7. improvement_potential: % neighbors that improve [0,1]
        8. search_difficulty: Local optima density [0,1]
        9. budget_consumed: FEs / MaxFEs [0,1]
    
    AH Features (2 × num_algorithms):
        For each algorithm l:
        - shift_best_l: Avg edit distance of best before/after
        - shift_worst_l: Avg edit distance of worst before/after
    """
    
    NUM_LA_FEATURES = 9
    
    def __init__(self, num_algorithms: int, neighbor_sample_size: int = 20):
        """
        Initialize state extractor.
        
        Args:
            num_algorithms: Number of algorithms in the pool
            neighbor_sample_size: Number of neighbors to sample for LA features
        """
        self.num_algorithms = num_algorithms
        self.neighbor_sample_size = neighbor_sample_size
        
        # Initialize algorithm history tracking
        self.algorithm_history: Dict[int, Dict[str, float]] = {
            i: {
                'best_shift': 0.0,
                'worst_shift': 0.0,
                'call_count': 0,
                'intervals_since_improvement': 0,
                'recent_reward': 0.0,
                'last_cost_improvement': 0.0
            }
            for i in range(num_algorithms)
        }
        
        # Exponential moving average factor for history (increased for faster adaptation)
        self.ema_alpha = 0.6
        
        # Episode tracking
        self.current_step = 0
        self.last_algo_used = None
        
    @property
    def state_dim(self) -> int:
        """
        Total dimension of state vector.
        
        LA features: 9
        AH features per algorithm: 4 (best_shift, worst_shift, intervals_since_improvement, recent_reward)
        Total: 9 + 4*L
        """
        return self.NUM_LA_FEATURES + 4 * self.num_algorithms
    
    def reset(self):
        """Reset history for new episode."""
        self.algorithm_history = {
            i: {
                'best_shift': 0.0,
                'worst_shift': 0.0,
                'call_count': 0,
                'intervals_since_improvement': 0,
                'recent_reward': 0.0,
                'last_cost_improvement': 0.0
            }
            for i in range(self.num_algorithms)
        }
        self.current_step = 0
        self.last_algo_used = None
    
    def extract_la_features(
        self,
        problem: Any,
        current_best_solution: Any,
        current_best_cost: float,
        population: List[Tuple[Any, float]],
        current_fes: int,
        max_fes: int
    ) -> np.ndarray:
        """
        Extract landscape analysis features.
        
        Args:
            problem: Problem instance (implements BaseProblem)
            current_best_solution: Current best solution
            current_best_cost: Cost of current best solution
            population: List of (solution, cost) tuples representing search state
            current_fes: Current function evaluations used
            max_fes: Maximum function evaluations budget
            
        Returns:
            numpy array of 9 LA features, all in [0, 1]
        """
        features = np.zeros(self.NUM_LA_FEATURES, dtype=np.float32)
        
        # Feature 1: Best cost normalized
        features[0] = problem.normalize_cost(current_best_cost)
        
        # Feature 2: Fitness-Distance Correlation (FDC)
        features[1] = self._compute_fdc(problem, current_best_solution, population)
        
        # Feature 3: Gap to known lower bound
        lb, ub = problem.get_cost_bounds()
        if ub > lb:
            features[2] = (current_best_cost - lb) / (ub - lb)
            features[2] = np.clip(features[2], 0.0, 1.0)
        else:
            features[2] = 0.5
        
        # Features 4, 5, 6: Diversity, convergence, dispersion difference
        div, conv, disp_diff = self._compute_population_features(problem, population)
        features[3] = div
        features[4] = conv
        features[5] = disp_diff
        
        # Features 7, 8: Improvement potential and search difficulty
        imp_pot, search_diff = self._compute_evolvability_features(
            problem, current_best_solution, current_best_cost
        )
        features[6] = imp_pot
        features[7] = search_diff
        
        # Feature 9: Budget consumed
        features[8] = current_fes / max_fes if max_fes > 0 else 1.0
        
        return features
    
    def _compute_fdc(
        self,
        problem: Any,
        best_solution: Any,
        population: List[Tuple[Any, float]]
    ) -> float:
        """
        Compute Fitness-Distance Correlation.
        
        FDC measures correlation between solution quality and distance to best.
        High FDC (close to 1) → easier to optimize (gradient towards optimum)
        Low/negative FDC → harder (deceptive landscape)
        
        Returns value normalized to [0, 1] where 0.5 is no correlation.
        """
        if len(population) < 3:
            return 0.5  # Not enough data
        
        distances = []
        costs = []
        
        for sol, cost in population:
            dist = problem.solution_distance(sol, best_solution)
            distances.append(dist)
            costs.append(cost)
        
        if len(set(distances)) < 2 or len(set(costs)) < 2:
            return 0.5  # No variance
        
        try:
            correlation, _ = stats.pearsonr(distances, costs)
            if np.isnan(correlation):
                return 0.5
            # Normalize from [-1, 1] to [0, 1]
            return (correlation + 1) / 2
        except:
            return 0.5
    
    def _compute_population_features(
        self,
        problem: Any,
        population: List[Tuple[Any, float]]
    ) -> Tuple[float, float, float]:
        """
        Compute population-based features.
        
        Returns:
            diversity: Average pairwise distance [0, 1]
            convergence: Max distance in population [0, 1]
            dispersion_diff: Difference between elite and full dispersion [0, 1]
        """
        if len(population) < 2:
            return 0.5, 0.5, 0.5
        
        solutions = [sol for sol, _ in population]
        costs = [cost for _, cost in population]
        
        # Compute pairwise distances (sample if too many)
        sample_size = min(len(solutions), 20)
        if len(solutions) > sample_size:
            indices = np.random.choice(len(solutions), sample_size, replace=False)
            sample_solutions = [solutions[i] for i in indices]
            sample_costs = [costs[i] for i in indices]
        else:
            sample_solutions = solutions
            sample_costs = costs
        
        distances = []
        for i in range(len(sample_solutions)):
            for j in range(i + 1, len(sample_solutions)):
                dist = problem.solution_distance(sample_solutions[i], sample_solutions[j])
                distances.append(dist)
        
        if not distances:
            return 0.5, 0.5, 0.5
        
        # Diversity: mean pairwise distance
        diversity = np.mean(distances)
        
        # Convergence: max distance (1 - max for convergence indicator)
        convergence = 1.0 - np.max(distances)
        
        # Dispersion difference: elite vs full population
        # Elite = top 20% by cost
        n_elite = max(1, len(sample_costs) // 5)
        elite_indices = np.argsort(sample_costs)[:n_elite]
        
        if n_elite >= 2:
            elite_distances = []
            for i in range(len(elite_indices)):
                for j in range(i + 1, len(elite_indices)):
                    dist = problem.solution_distance(
                        sample_solutions[elite_indices[i]],
                        sample_solutions[elite_indices[j]]
                    )
                    elite_distances.append(dist)
            elite_dispersion = np.mean(elite_distances) if elite_distances else diversity
        else:
            elite_dispersion = 0.0
        
        # Difference normalized with sigmoid
        diff = diversity - elite_dispersion
        dispersion_diff = 1 / (1 + np.exp(-5 * diff))  # Sigmoid centered at 0
        
        return float(diversity), float(convergence), float(dispersion_diff)
    
    def _compute_evolvability_features(
        self,
        problem: Any,
        current_solution: Any,
        current_cost: float
    ) -> Tuple[float, float]:
        """
        Compute evolvability features by sampling neighbors.
        
        This DOES consume function evaluations.
        
        Returns:
            improvement_potential: Fraction of neighbors that improve [0, 1]
            search_difficulty: Estimate of local optima density [0, 1]
        """
        neighbors = problem.generate_neighbors(current_solution, self.neighbor_sample_size)
        
        improvements = 0
        neutral = 0
        deteriorations = 0
        
        neighbor_costs = []
        for neighbor in neighbors:
            cost = problem.evaluate(neighbor)
            neighbor_costs.append(cost)
            
            if cost < current_cost - 1e-9:
                improvements += 1
            elif cost > current_cost + 1e-9:
                deteriorations += 1
            else:
                neutral += 1
        
        k = len(neighbors)
        if k == 0:
            return 0.5, 0.5
        
        # Improvement potential
        improvement_potential = improvements / k
        
        # Search difficulty: high neutral ratio + low improvement = likely stuck
        # Use combination of neutral and deterioration ratios
        neutral_ratio = neutral / k
        improvement_ratio = improvements / k
        
        # High difficulty if few improvements and many neutral moves (local optimum)
        search_difficulty = 1.0 - improvement_ratio
        # Adjust by neutral ratio (flat landscapes are hard)
        if neutral_ratio > 0.3:
            search_difficulty = min(1.0, search_difficulty + 0.2)
        
        return float(improvement_potential), float(search_difficulty)
    
    def extract_ah_features(self) -> np.ndarray:
        """
        Extract algorithm history features.
        
        Returns:
            numpy array of 4*L features per algorithm:
            - best_shift: Average edit distance of best solution
            - worst_shift: Average edit distance of worst solution
            - intervals_since_improvement: Normalized stagnation indicator [0,1]
            - recent_reward: Reward from last use (normalized)
        """
        features = np.zeros(4 * self.num_algorithms, dtype=np.float32)
        
        for i in range(self.num_algorithms):
            history = self.algorithm_history[i]
            features[4 * i] = history['best_shift']
            features[4 * i + 1] = history['worst_shift']
            
            # Normalize intervals_since_improvement (cap at 10 intervals)
            features[4 * i + 2] = min(history['intervals_since_improvement'] / 10.0, 1.0)
            
            # Recent reward (already normalized)
            features[4 * i + 3] = np.clip(history['recent_reward'], 0.0, 1.0)
        
        return features
    
    def update_algorithm_history(
        self,
        algo_idx: int,
        best_before: Any,
        best_after: Any,
        worst_before: Any,
        worst_after: Any,
        problem: Any,
        cost_before: float = None,
        cost_after: float = None,
        reward: float = 0.0
    ):
        """
        Update algorithm history after algorithm execution.
        
        Uses exponential moving average to track shifts over time.
        Also tracks stagnation and recent performance.
        
        Args:
            algo_idx: Index of the algorithm that was executed
            best_before: Best solution before execution
            best_after: Best solution after execution
            worst_before: Worst solution in population before
            worst_after: Worst solution in population after
            problem: Problem instance for computing distances
            cost_before: Cost before execution (for improvement tracking)
            cost_after: Cost after execution (for improvement tracking)
            reward: Reward received for this algorithm execution
        """
        if algo_idx < 0 or algo_idx >= self.num_algorithms:
            return
        
        # Compute shifts (distances)
        best_shift = problem.solution_distance(best_before, best_after)
        worst_shift = problem.solution_distance(worst_before, worst_after)
        
        # Update with EMA
        history = self.algorithm_history[algo_idx]
        if history['call_count'] == 0:
            history['best_shift'] = best_shift
            history['worst_shift'] = worst_shift
        else:
            history['best_shift'] = (
                self.ema_alpha * best_shift + 
                (1 - self.ema_alpha) * history['best_shift']
            )
            history['worst_shift'] = (
                self.ema_alpha * worst_shift + 
                (1 - self.ema_alpha) * history['worst_shift']
            )
        
        # Track improvement and stagnation
        if cost_before is not None and cost_after is not None:
            improvement = cost_before - cost_after
            history['last_cost_improvement'] = improvement
            
            # Update stagnation counter
            if improvement > 1e-9:  # Meaningful improvement
                history['intervals_since_improvement'] = 0
            else:
                history['intervals_since_improvement'] += 1
        
        # Update recent reward
        history['recent_reward'] = reward
        
        # Update intervals_since_improvement for other algorithms
        for i in range(self.num_algorithms):
            if i != algo_idx:
                self.algorithm_history[i]['intervals_since_improvement'] += 1
        
        history['call_count'] += 1
        self.current_step += 1
        self.last_algo_used = algo_idx
    
    def extract_state(
        self,
        problem: Any,
        current_best_solution: Any,
        current_best_cost: float,
        population: List[Tuple[Any, float]],
        current_fes: int,
        max_fes: int
    ) -> np.ndarray:
        """
        Extract full state vector (LA + AH features).
        
        Args:
            problem: Problem instance
            current_best_solution: Current best solution
            current_best_cost: Cost of best solution
            population: List of (solution, cost) tuples
            current_fes: Current FEs used
            max_fes: Maximum FEs budget
            
        Returns:
            numpy array of shape (state_dim,) with all features in [0, 1]
        """
        la_features = self.extract_la_features(
            problem, current_best_solution, current_best_cost,
            population, current_fes, max_fes
        )
        ah_features = self.extract_ah_features()
        
        return np.concatenate([la_features, ah_features])
