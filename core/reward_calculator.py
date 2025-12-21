"""
Minimal reward calculator for RL-DAS.

Simplified reward: improvement or penalty, no complex factors.
"""

import numpy as np
from typing import List, Tuple


class RewardCalculator:
    """
    Calculate rewards for algorithm selection.
    
    Simple design:
    - Positive reward for improvement (scaled by 100x)
    - Small penalty for no improvement (-0.1)
    - No complex factors (speed, feasibility, milestones)
    """
    
    def __init__(self, max_fes: int, initial_cost: float = None):
        """
        Initialize reward calculator.
        
        Args:
            max_fes: Maximum function evaluations budget
            initial_cost: Initial cost for normalization (set in reset())
        """
        self.max_fes = max_fes
        self.initial_cost = initial_cost
        self.improvement_scale = 100.0  # Scale up for clear learning signal
        
    def reset(self, initial_cost: float):
        """
        Reset for new episode.
        
        Args:
            initial_cost: Cost of initial solution
        """
        self.initial_cost = initial_cost
    
    def compute_step_reward(
        self,
        prev_cost: float,
        curr_cost: float
    ) -> float:
        """
        Compute reward for a single step.
        
        Simple formula:
        - If improved: (improvement / initial_cost) * 100
        - If not improved: -0.1 (penalty for wasting FEs)
        
        Args:
            prev_cost: Cost before algorithm execution
            curr_cost: Cost after algorithm execution
            
        Returns:
            Reward value
        """
        if self.initial_cost is None or self.initial_cost <= 0:
            # Fallback if not initialized
            improvement = prev_cost - curr_cost
        else:
            # Normalized improvement
            improvement = (prev_cost - curr_cost) / self.initial_cost
        
        # Reward or penalty
        if improvement > 1e-9:
            # Positive reward for improvement
            return improvement * self.improvement_scale
        else:
            # Penalty for no improvement (wasting FEs)
            return -0.1