"""
Abstract base class for discrete optimization problems.

This defines the interface that all problems must implement for
the RL-DAS system to work with them in a generalizable way.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
import numpy as np


class BaseProblem(ABC):
    """
    Abstract base class for discrete optimization problems.
    
    All problems must implement this interface to work with the
    RL-DAS system. The key design principle is that all methods
    should return normalized/problem-agnostic values where possible.
    """
    
    def __init__(self):
        """Initialize problem with evaluation counter."""
        self.evaluation_count = 0
        self._cost_cache = {}  # Optional caching for expensive evaluations
    
    @property
    @abstractmethod
    def problem_type(self) -> str:
        """
        Return problem type identifier.
        
        Returns:
            String identifier like 'TSP', 'VRP', 'JSS', 'GCP'
        """
        pass
    
    @property
    @abstractmethod
    def size(self) -> int:
        """
        Return problem size for reference.
        
        For TSP: number of cities
        For VRP: number of customers
        For JSS: jobs × machines
        For GCP: number of nodes
        
        Returns:
            Integer representing problem size
        """
        pass
    
    @abstractmethod
    def evaluate(self, solution: Any) -> float:
        """
        Evaluate solution cost.
        
        IMPORTANT: This method MUST increment self.evaluation_count.
        
        Args:
            solution: Problem-specific solution representation
            
        Returns:
            Cost value (lower is better for minimization problems)
        """
        pass
    
    @abstractmethod
    def generate_random_solution(self) -> Any:
        """
        Generate a random feasible solution.
        
        Returns:
            A random solution in the problem's solution space
        """
        pass
    
    @abstractmethod
    def generate_neighbor(self, solution: Any) -> Any:
        """
        Generate a single neighbor solution using a simple move operator.
        
        This is used for landscape sampling. Should be a lightweight
        operation (e.g., 2-opt move, swap, insert).
        
        Args:
            solution: Current solution
            
        Returns:
            A neighbor solution
        """
        pass
    
    def generate_neighbors(self, solution: Any, k: int) -> List[Any]:
        """
        Generate k diverse neighbor solutions for landscape analysis.
        
        Default implementation calls generate_neighbor k times.
        Override for more sophisticated sampling.
        
        Args:
            solution: Current solution
            k: Number of neighbors to generate
            
        Returns:
            List of k neighbor solutions
        """
        return [self.generate_neighbor(solution) for _ in range(k)]
    
    @abstractmethod
    def solution_distance(self, sol1: Any, sol2: Any) -> float:
        """
        Compute distance/dissimilarity between two solutions.
        
        This is used for:
        - Diversity measurement
        - Algorithm history tracking (shift vectors)
        - Fitness-Distance Correlation
        
        The distance should be normalized to [0, 1] where:
        - 0 means identical solutions
        - 1 means maximally different solutions
        
        Args:
            sol1: First solution
            sol2: Second solution
            
        Returns:
            Normalized distance in [0, 1]
        """
        pass
    
    @abstractmethod
    def get_cost_bounds(self) -> Tuple[float, float]:
        """
        Return (lower_bound, upper_bound) for cost normalization.
        
        Lower bound: Known or estimated minimum cost
        Upper bound: Estimated maximum cost (e.g., worst random solution)
        
        These don't need to be exact but should bound typical costs.
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        pass
    
    def normalize_cost(self, cost: float) -> float:
        """
        Normalize cost to [0, 1] range.
        
        Uses bounds from get_cost_bounds().
        Values outside bounds are clipped.
        
        Args:
            cost: Raw cost value
            
        Returns:
            Normalized cost in [0, 1]
        """
        lb, ub = self.get_cost_bounds()
        if ub <= lb:
            return 0.5  # Fallback if bounds are invalid
        normalized = (cost - lb) / (ub - lb)
        return np.clip(normalized, 0.0, 1.0)
    
    def is_feasible(self, solution: Any) -> bool:
        """
        Check if solution satisfies all constraints.
        
        Default implementation assumes unconstrained problem.
        Override for constrained problems (VRP capacity, JSS precedence, etc.)
        
        Args:
            solution: Solution to check
            
        Returns:
            True if feasible, False otherwise
        """
        return True
    
    def copy_solution(self, solution: Any) -> Any:
        """
        Create a deep copy of a solution.
        
        Default uses numpy copy. Override if needed.
        
        Args:
            solution: Solution to copy
            
        Returns:
            Deep copy of solution
        """
        if isinstance(solution, np.ndarray):
            return solution.copy()
        elif isinstance(solution, list):
            return solution.copy()
        else:
            import copy
            return copy.deepcopy(solution)
    
    def reset_evaluation_count(self):
        """Reset the evaluation counter to zero."""
        self.evaluation_count = 0
        self._cost_cache.clear()
    
    def get_evaluation_count(self) -> int:
        """Get current evaluation count."""
        return self.evaluation_count
