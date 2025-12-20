"""
Reward calculator for RL-DAS.

Implements the paper's reward design with:
- Base improvement reward
- Speed factor (applied at episode end)
- Feasibility factor for constrained problems
- Milestone bonuses
"""

from typing import List, Optional, Tuple
import numpy as np


class RewardCalculator:
    """
    Calculate RL rewards with speed factor and feasibility handling.
    
    The key insight from the paper is that rewards should encourage
    *efficient* improvement, not just improvement. This is achieved
    by multiplying final rewards by a speed factor.
    
    Reward formula:
        r = base_improvement × speed_factor × feasibility_factor × milestone_bonus
    
    Where:
        - base_improvement = (cost_prev - cost_curr) / cost_initial
        - speed_factor = MaxFEs / FEs_at_termination (applied at episode end)
        - feasibility_factor = 1.0 if feasible, else penalty
        - milestone_bonus = 1.2 if reached new global best
    """
    
    def __init__(
        self,
        max_fes: int,
        feasibility_penalty_base: float = 0.3,
        milestone_bonus: float = 1.2,
        improvement_scale: float = 10.0
    ):
        """
        Initialize reward calculator.
        
        Args:
            max_fes: Maximum function evaluations for the episode
            feasibility_penalty_base: Base penalty for infeasible solutions
            milestone_bonus: Multiplier for reaching new best
            improvement_scale: Scaling factor for improvements (paper uses normalization)
        """
        self.max_fes = max_fes
        self.feasibility_penalty_base = feasibility_penalty_base
        self.milestone_bonus_factor = milestone_bonus
        self.improvement_scale = improvement_scale
        
        # Episode state
        self.initial_cost: Optional[float] = None
        self.previous_best_cost: Optional[float] = None
        self.episode_rewards: List[Tuple[float, int]] = []  # (raw_reward, fes_at_step)
        self.global_best_cost: Optional[float] = None
    
    def reset(self, initial_cost: float):
        """
        Reset for new episode.
        
        Args:
            initial_cost: Cost of initial solution
        """
        self.initial_cost = initial_cost
        self.previous_best_cost = initial_cost
        self.global_best_cost = initial_cost
        self.episode_rewards = []
    
    def compute_step_reward(
        self,
        prev_cost: float,
        curr_cost: float,
        fes_used: int,
        is_feasible: bool = True,
        is_global_best: bool = False
    ) -> float:
        """
        Compute immediate reward for a step.
        
        Note: This does NOT include the speed factor yet.
        The speed factor is applied at episode end.
        
        Args:
            prev_cost: Cost before this step
            curr_cost: Cost after this step
            fes_used: Total FEs used so far
            is_feasible: Whether current solution is feasible
            is_global_best: Whether this is a new global best
            
        Returns:
            Raw reward (without speed factor)
        """
        if self.initial_cost is None:
            raise ValueError("Must call reset() before computing rewards")
        
        # Base improvement (normalized by initial cost)
        if self.initial_cost > 0:
            improvement = (prev_cost - curr_cost) / self.initial_cost
        else:
            improvement = prev_cost - curr_cost
        
        # Scale improvements to reasonable range
        base_reward = improvement * self.improvement_scale
        
        # Feasibility factor
        if is_feasible:
            feasibility_factor = 1.0
        else:
            # Penalty decreases as we run out of budget
            # (early infeasibility is less penalized, late is more)
            fes_remaining_ratio = max(0, (self.max_fes - fes_used) / self.max_fes)
            feasibility_factor = (
                self.feasibility_penalty_base + 
                (1 - self.feasibility_penalty_base) * fes_remaining_ratio
            )
        
        # Milestone bonus for new global best
        if is_global_best and curr_cost < self.global_best_cost:
            milestone_factor = self.milestone_bonus_factor
            self.global_best_cost = curr_cost
        else:
            milestone_factor = 1.0
        
        # Combine factors
        raw_reward = base_reward * feasibility_factor * milestone_factor
        
        # Store for later speed factor application
        self.episode_rewards.append((raw_reward, fes_used))
        
        # Update tracking
        self.previous_best_cost = min(self.previous_best_cost, curr_cost)
        
        return raw_reward
    
    def finalize_episode(self, final_fes: int) -> List[float]:
        """
        Apply speed factor to all stored rewards at episode end.
        
        The speed factor rewards efficient optimization:
        - Fast termination (few FEs) → higher multiplier
        - Slow termination (many FEs) → lower multiplier
        
        Args:
            final_fes: Total FEs used at episode end
            
        Returns:
            List of final rewards with speed factor applied
        """
        if final_fes <= 0:
            final_fes = 1  # Avoid division by zero
        
        # Speed factor: MaxFEs / FEs_at_termination
        # Capped to prevent extreme values
        speed_factor = min(self.max_fes / final_fes, 5.0)
        
        # Apply speed factor to all rewards
        final_rewards = [
            raw_reward * speed_factor 
            for raw_reward, _ in self.episode_rewards
        ]
        
        return final_rewards
    
    def get_current_reward_sum(self) -> float:
        """Get sum of raw rewards so far (without speed factor)."""
        return sum(r for r, _ in self.episode_rewards)
    
    def compute_terminal_reward(
        self,
        final_cost: float,
        final_fes: int,
        is_feasible: bool = True
    ) -> float:
        """
        Compute a single terminal reward for the entire episode.
        
        Alternative to step-by-step rewards. Useful for simpler training.
        
        Args:
            final_cost: Final solution cost
            final_fes: Total FEs used
            is_feasible: Whether final solution is feasible
            
        Returns:
            Terminal reward value
        """
        if self.initial_cost is None:
            return 0.0
        
        # Total improvement
        if self.initial_cost > 0:
            improvement = (self.initial_cost - final_cost) / self.initial_cost
        else:
            improvement = self.initial_cost - final_cost
        
        # Speed factor
        speed_factor = min(self.max_fes / max(final_fes, 1), 5.0)
        
        # Feasibility
        feasibility_factor = 1.0 if is_feasible else self.feasibility_penalty_base
        
        return improvement * speed_factor * feasibility_factor * self.improvement_scale