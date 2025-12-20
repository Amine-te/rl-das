from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class BaseAlgorithm(ABC):
    """Abstract base class for optimization algorithms."""
    
    def __init__(self, problem: Any, name: str):
        """
        Initialize algorithm.
        
        Args:
            problem: Problem instance to solve
            name: Algorithm identifier (e.g., 'GA', 'SA', 'TS')
        """
        self.problem = problem
        self.name = name
        self.best_solution = None
        self.best_cost = float('inf')
        self.evaluations_used = 0
        
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize algorithm-specific state.
        This is called once at the start or when resetting.
        """
        pass
    
    @abstractmethod
    def step(self, num_evaluations: int) -> Tuple[Any, float]:
        """
        Execute algorithm for a specified number of evaluations.
        
        Args:
            num_evaluations: Number of solution evaluations to perform
            
        Returns:
            Tuple of (best_solution, best_cost) found in this step
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get current algorithm state for context preservation.
        
        Returns:
            Dictionary containing algorithm-specific state
        """
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore algorithm state from context.
        
        Args:
            state: Dictionary containing algorithm state to restore
        """
        pass
        
    @abstractmethod
    def inject_solution(self, solution: Any, cost: float) -> None:
        """
        Inject a solution into the algorithm's state to continue search from there.
        Used when switching algorithms to share the global best.
        
        Args:
            solution: Solution to inject
            cost: Cost of the solution
        """
        pass
    
    def get_best(self) -> Tuple[Any, float]:
        """
        Return the best solution found so far.
        
        Returns:
            Tuple of (best_solution, best_cost)
        """
        return self.best_solution, self.best_cost
    
    def update_best(self, solution: Any, cost: float) -> bool:
        """
        Update best solution if cost is better.
        
        Args:
            solution: Candidate solution
            cost: Cost of the solution
            
        Returns:
            True if best was updated, False otherwise
        """
        if cost < self.best_cost:
            self.best_solution = solution
            self.best_cost = cost
            return True
        return False
