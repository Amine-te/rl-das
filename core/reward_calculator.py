"""
Reward calculator for RL-DAS with Stick-with-Winner behavior.

Simple, clear reward signals:
- Reward staying with improving algorithm
- Punish staying when stuck
- Reward switching to proven/fresh algorithm when stuck
"""

import numpy as np
from typing import List


class RewardCalculator:
    """
    Calculate rewards for algorithm selection.
    
    Behavioral design:
    - Stick with winner: +1.0 for staying when improving
    - Switch when stuck: -1.0 for staying when not improving
    - Smart switching: bonus for switching to proven/fresh algo
    """
    
    def __init__(self, max_fes: int, initial_cost: float = None, num_algorithms: int = 4):
        """
        Initialize reward calculator.
        
        Args:
            max_fes: Maximum function evaluations budget
            initial_cost: Initial cost for normalization (set in reset())
            num_algorithms: Number of algorithms in pool
        """
        self.max_fes = max_fes
        self.initial_cost = initial_cost
        self.num_algorithms = num_algorithms
        
        # Track algorithm success history
        self.algo_attempts = np.zeros(num_algorithms)
        self.algo_successes = np.zeros(num_algorithms)
        self.algo_last_used = np.zeros(num_algorithms)  # Step when last used
        self.current_step = 0
        
    def reset(self, initial_cost: float):
        """Reset for new episode."""
        self.initial_cost = initial_cost
        self.algo_attempts = np.zeros(self.num_algorithms)
        self.algo_successes = np.zeros(self.num_algorithms)
        self.algo_last_used = np.full(self.num_algorithms, -10)  # All "stale" initially
        self.current_step = 0
    
    def get_success_rates(self) -> np.ndarray:
        """Get success rate for each algorithm."""
        rates = np.zeros(self.num_algorithms)
        for i in range(self.num_algorithms):
            if self.algo_attempts[i] > 0:
                rates[i] = self.algo_successes[i] / self.algo_attempts[i]
            else:
                rates[i] = 0.5  # Unknown = neutral
        return rates
    
    def get_freshness(self) -> np.ndarray:
        """Get freshness score for each algorithm (how long since used)."""
        freshness = np.zeros(self.num_algorithms)
        for i in range(self.num_algorithms):
            steps_since_used = self.current_step - self.algo_last_used[i]
            # Normalize: 0 = just used, 1 = unused for 10+ steps
            freshness[i] = min(steps_since_used / 10.0, 1.0)
        return freshness
    
    def compute_step_reward(
        self,
        prev_cost: float,
        curr_cost: float,
        algo_idx: int,
        switched: bool,
        prev_improved: bool
    ) -> float:
        """
        Compute reward for a single step.
        
        Args:
            prev_cost: Cost before algorithm execution
            curr_cost: Cost after algorithm execution
            algo_idx: Index of the algorithm that was selected
            switched: Whether agent switched algorithms this step
            prev_improved: Whether the PREVIOUS step improved
            
        Returns:
            Reward value
        """
        self.current_step += 1
        
        # Track this attempt
        self.algo_attempts[algo_idx] += 1
        self.algo_last_used[algo_idx] = self.current_step
        
        # Did we improve?
        improved = curr_cost < prev_cost - 1e-9
        if improved:
            self.algo_successes[algo_idx] += 1
        
        # Get algorithm quality metrics
        success_rates = self.get_success_rates()
        freshness = self.get_freshness()
        
        # === REWARD LOGIC ===
        
        if improved:
            # === CASE 1: We improved ===
            # Calculate relative improvement magnitude
            rel_imp = 0.0
            if self.initial_cost and self.initial_cost > 0:
                rel_imp = (prev_cost - curr_cost) / self.initial_cost
            
            # Scale factor: 1% improvement (0.01) -> +1.0 added reward
            # This makes the reward relative to how much we improved
            imp_bonus = rel_imp * 100.0
            
            if switched:
                # Switching to an algorithm that improves -> MAX REWARD + BONUS
                return 1.0 + imp_bonus
            else:
                # Sticking with an algorithm that improves -> HIGH REWARD + BONUS
                return 0.7 + imp_bonus
        else:
            # === CASE 2: We did NOT improve ===
            if not switched:
                # Sticking with an algorithm that doesn't improve -> HIGH NEGATIVE REWARD
                # This is the critical signal to force switching
                return -1.0
            else:
                # Switching to an algorithm that doesn't improve -> LOW NEGATIVE REWARD
                # We tried to switch (good behavior) but it didn't work (bad outcome)
                # Differentiate slightly based on whether it was a "smart" switch
                algo_success = success_rates[algo_idx]
                algo_fresh = freshness[algo_idx]
                
                if algo_fresh > 0.5:
                    # Switch to fresh algo (exploration)
                    return -0.1
                elif algo_success > 0.5:
                    # Switch to proven algo
                    return -0.2
                else:
                    # Switch to random/bad algo
                    return -0.3
