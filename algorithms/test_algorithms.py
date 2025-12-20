"""
Simple algorithm implementations for testing.

These are minimal implementations to verify integration.
Full implementations should be in separate files.
"""

from typing import Any, Dict, List, Tuple
import numpy as np

from .base_algorithm import BaseAlgorithm


class RandomSearchAlgorithm(BaseAlgorithm):
    """
    Random Search algorithm (for testing).
    
    Simply generates random solutions and keeps the best.
    """
    
    def __init__(self, problem: Any):
        super().__init__(problem, 'RS')
        self.solutions: List[Tuple[Any, float]] = []
    
    def initialize(self, **kwargs):
        """Initialize with random solution."""
        sol = self.problem.generate_random_solution()
        cost = self.problem.evaluate(sol)
        self.solutions = [(sol.copy(), cost)]
        self.update_best(sol, cost)
        self.evaluations_used = 1
    
    def step(self, num_evaluations: int) -> Tuple[Any, float]:
        """Generate random solutions."""
        for _ in range(num_evaluations):
            sol = self.problem.generate_random_solution()
            cost = self.problem.evaluate(sol)
            self.evaluations_used += 1
            self.update_best(sol, cost)
            
            # Keep some solutions
            if len(self.solutions) < 10:
                self.solutions.append((sol.copy(), cost))
            elif cost < self.solutions[-1][1]:
                self.solutions[-1] = (sol.copy(), cost)
                self.solutions.sort(key=lambda x: x[1])
        
        return self.best_solution, self.best_cost
    
    def get_state(self) -> Dict[str, Any]:
        """Get algorithm state."""
        return {
            'solutions': [(s.copy(), c) for s, c in self.solutions],
            'best_solution': self.best_solution.copy() if self.best_solution is not None else None,
            'best_cost': self.best_cost,
            'evaluations_used': self.evaluations_used
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore algorithm state."""
        self.solutions = [(s.copy(), c) for s, c in state['solutions']]
        self.best_solution = state['best_solution'].copy() if state['best_solution'] is not None else None
        self.best_cost = state['best_cost']
        self.evaluations_used = state['evaluations_used']
    
    def inject_solution(self, solution: Any, cost: float):
        """Inject a solution."""
        self.update_best(solution, cost)
        self.solutions.insert(0, (solution.copy(), cost))
        self.solutions.sort(key=lambda x: x[1])
        self.solutions = self.solutions[:10]


class LocalSearchAlgorithm(BaseAlgorithm):
    """
    Local Search algorithm (for testing).
    
    Hill climbing with random restarts.
    """
    
    def __init__(self, problem: Any, neighbors_per_step: int = 10):
        super().__init__(problem, 'LS')
        self.neighbors_per_step = neighbors_per_step
        self.current_solution = None
        self.current_cost = float('inf')
        self.restart_count = 0
    
    def initialize(self, **kwargs):
        """Initialize with random solution."""
        self.current_solution = self.problem.generate_random_solution()
        self.current_cost = self.problem.evaluate(self.current_solution)
        self.update_best(self.current_solution, self.current_cost)
        self.evaluations_used = 1
        self.restart_count = 0
    
    def step(self, num_evaluations: int) -> Tuple[Any, float]:
        """Perform local search steps."""
        evals_done = 0
        
        while evals_done < num_evaluations:
            # Generate neighbors
            improved = False
            neighbors = self.problem.generate_neighbors(
                self.current_solution, 
                min(self.neighbors_per_step, num_evaluations - evals_done)
            )
            
            for neighbor in neighbors:
                cost = self.problem.evaluate(neighbor)
                evals_done += 1
                self.evaluations_used += 1
                
                if cost < self.current_cost:
                    self.current_solution = neighbor.copy()
                    self.current_cost = cost
                    self.update_best(neighbor, cost)
                    improved = True
                    break
                
                if evals_done >= num_evaluations:
                    break
            
            # Random restart if stuck
            if not improved and evals_done < num_evaluations:
                self.current_solution = self.problem.generate_random_solution()
                self.current_cost = self.problem.evaluate(self.current_solution)
                evals_done += 1
                self.evaluations_used += 1
                self.update_best(self.current_solution, self.current_cost)
                self.restart_count += 1
        
        return self.best_solution, self.best_cost
    
    def get_state(self) -> Dict[str, Any]:
        """Get algorithm state."""
        return {
            'current_solution': self.current_solution.copy() if self.current_solution is not None else None,
            'current_cost': self.current_cost,
            'best_solution': self.best_solution.copy() if self.best_solution is not None else None,
            'best_cost': self.best_cost,
            'evaluations_used': self.evaluations_used,
            'restart_count': self.restart_count
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore algorithm state."""
        self.current_solution = state['current_solution'].copy() if state['current_solution'] is not None else None
        self.current_cost = state['current_cost']
        self.best_solution = state['best_solution'].copy() if state['best_solution'] is not None else None
        self.best_cost = state['best_cost']
        self.evaluations_used = state['evaluations_used']
        self.restart_count = state['restart_count']
    
    def inject_solution(self, solution: Any, cost: float):
        """Inject a solution."""
        self.current_solution = solution.copy()
        self.current_cost = cost
        self.update_best(solution, cost)


class GreedySearchAlgorithm(BaseAlgorithm):
    """
    Greedy improvement algorithm (for testing).
    
    Always accepts improving neighbors, never accepts worse.
    """
    
    def __init__(self, problem: Any):
        super().__init__(problem, 'GS')
        self.current_solution = None
        self.current_cost = float('inf')
    
    def initialize(self, **kwargs):
        """Initialize with nearest neighbor heuristic if available."""
        if hasattr(self.problem, '_nearest_neighbor_tour'):
            self.current_solution = self.problem._nearest_neighbor_tour()
            self.current_cost = self.problem.evaluate(self.current_solution)
        else:
            self.current_solution = self.problem.generate_random_solution()
            self.current_cost = self.problem.evaluate(self.current_solution)
        
        self.update_best(self.current_solution, self.current_cost)
        self.evaluations_used = 1
    
    def step(self, num_evaluations: int) -> Tuple[Any, float]:
        """Perform greedy search steps."""
        for _ in range(num_evaluations):
            # Generate one neighbor
            neighbor = self.problem.generate_neighbor(self.current_solution)
            cost = self.problem.evaluate(neighbor)
            self.evaluations_used += 1
            
            if cost < self.current_cost:
                self.current_solution = neighbor.copy()
                self.current_cost = cost
                self.update_best(neighbor, cost)
        
        return self.best_solution, self.best_cost
    
    def get_state(self) -> Dict[str, Any]:
        """Get algorithm state."""
        return {
            'current_solution': self.current_solution.copy() if self.current_solution is not None else None,
            'current_cost': self.current_cost,
            'best_solution': self.best_solution.copy() if self.best_solution is not None else None,
            'best_cost': self.best_cost,
            'evaluations_used': self.evaluations_used
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore algorithm state."""
        self.current_solution = state['current_solution'].copy() if state['current_solution'] is not None else None
        self.current_cost = state['current_cost']
        self.best_solution = state['best_solution'].copy() if state['best_solution'] is not None else None
        self.best_cost = state['best_cost']
        self.evaluations_used = state['evaluations_used']
    
    def inject_solution(self, solution: Any, cost: float):
        """Inject a solution."""
        self.current_solution = solution.copy()
        self.current_cost = cost
        self.update_best(solution, cost)
