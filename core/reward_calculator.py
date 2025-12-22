"""
Reward calculator for RL-DAS with Performance-Gated Diversity.

Key design principles:
1. Reward switching ONLY when it leads to improvement
2. Graduated penalties based on algorithm credibility
3. Early exploration bonus that decays over progress
4. Freshness bonus for trying under-used algorithms
"""

import numpy as np
from typing import List, Tuple


class RewardCalculator:
    """
    Calculate rewards for algorithm selection.
    
    Design:
    - Positive reward for improvement (scaled by 100x)
    - Switch bonus ONLY on successful switch (+0.2)
    - Freshness bonus for trying underused algorithms
    - Graduated penalty based on credibility
    - Early exploration multiplier that decays
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
        self.improvement_scale = 100.0  # Scale up for clear learning signal
        self.num_algorithms = num_algorithms
        
        # Track algorithm usage for freshness bonus
        self.algo_usage_counts = [0] * num_algorithms
        self.total_steps = 0
        
    def reset(self, initial_cost: float):
        """
        Reset for new episode.
        
        Args:
            initial_cost: Cost of initial solution
        """
        self.initial_cost = initial_cost
        self.algo_usage_counts = [0] * self.num_algorithms
        self.total_steps = 0
    
    def compute_step_reward(
        self,
        prev_cost: float,
        curr_cost: float,
        fes_used: int = 1000,
        progress: float = 0.5,
        switched: bool = False,
        credibility: float = 0.5,
        algo_idx: int = 0
    ) -> float:
        """
        Compute reward for a single step with Performance-Gated Diversity.
        
        Args:
            prev_cost: Cost before algorithm execution
            curr_cost: Cost after algorithm execution
            fes_used: Number of FEs used in this step
            progress: Episode progress (0.0 to 1.0)
            switched: Whether agent switched algorithms this step
            credibility: Credibility score of selected algorithm [0,1]
            algo_idx: Index of the algorithm that was selected
            
        Returns:
            Reward value
        """
        # Update tracking
        self.total_steps += 1
        if 0 <= algo_idx < self.num_algorithms:
            self.algo_usage_counts[algo_idx] += 1
        
        if self.initial_cost is None or self.initial_cost <= 0:
            # Fallback if not initialized
            improvement = prev_cost - curr_cost
        else:
            # Normalized improvement
            improvement = (prev_cost - curr_cost) / self.initial_cost
        
        # === IMPROVEMENT CASE ===
        if improvement > 1e-9:
            base_reward = improvement * self.improvement_scale
            
            # Phase 1: Efficiency (< 30% progress)
            # Favor algorithms that achieve improvement quickly
            if progress < 0.3:
                efficiency_factor = 1000.0 / max(fes_used, 1)
                base_reward = base_reward * efficiency_factor
            
            # PERFORMANCE-GATED SWITCH BONUS
            # Only reward switching when it actually helped
            if switched:
                # Successful switch deserves a meaningful bonus
                switch_success_bonus = 0.2
                base_reward += switch_success_bonus
            
            return base_reward
        
        # === NO IMPROVEMENT CASE ===
        else:
            # Base penalty for wasting FEs
            base_penalty = -0.2
            
            # Credibility-weighted penalty:
            # - High credibility (0.8+): algo usually works, just bad luck → -0.2
            # - Medium credibility (0.5): normal penalty → -0.25
            # - Low credibility (0.0): you chose a known-bad algo → -0.3
            credibility_penalty = -0.1 * (1.0 - credibility)
            
            # FRESHNESS BONUS (preserves diversity even when no improvement)
            # If this algorithm has been used less than average, reduce penalty
            # This encourages trying under-explored algorithms
            freshness_bonus = 0.0
            if self.total_steps > 0:
                avg_usage = self.total_steps / self.num_algorithms
                algo_usage = self.algo_usage_counts[algo_idx] if 0 <= algo_idx < self.num_algorithms else avg_usage
                
                if algo_usage < avg_usage * 0.5:
                    # Under-used algorithm: reduce penalty to encourage exploration
                    freshness_bonus = 0.1
                elif algo_usage > avg_usage * 1.5:
                    # Over-used algorithm: slight extra penalty
                    freshness_bonus = -0.05
            
            # EARLY EXPLORATION MULTIPLIER
            # Early in episode: softer penalties to encourage exploration
            # Late in episode: full penalties to focus on best algorithms
            if progress < 0.2:
                exploration_multiplier = 0.5  # Half penalty early
            elif progress < 0.4:
                exploration_multiplier = 0.75
            else:
                exploration_multiplier = 1.0  # Full penalty later
            
            total_penalty = (base_penalty + credibility_penalty) * exploration_multiplier + freshness_bonus
            
            return total_penalty
