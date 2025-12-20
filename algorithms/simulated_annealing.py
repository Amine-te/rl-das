"""
Simulated Annealing (SA) for discrete optimization.

A probabilistic local search that accepts worse solutions
with decreasing probability, controlled by a temperature parameter.

Good for: Escaping local optima, robust exploration-exploitation balance
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np

from .base_algorithm import BaseAlgorithm


class SimulatedAnnealing(BaseAlgorithm):
    """
    Simulated Annealing for discrete optimization.
    
    Features:
    - Geometric cooling schedule
    - Adaptive reheating when stuck
    - Multiple neighborhood operators
    - Acceptance tracking for parameter tuning
    """
    
    def __init__(
        self,
        problem: Any,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.995,
        min_temperature: float = 0.01,
        neighbors_per_temp: int = 10,
        reheat_threshold: int = 50,
        reheat_factor: float = 2.0
    ):
        """
        Initialize Simulated Annealing.
        
        Args:
            problem: Problem instance to solve
            initial_temperature: Starting temperature
            cooling_rate: Temperature decay factor (0 < rate < 1)
            min_temperature: Minimum temperature threshold
            neighbors_per_temp: Number of neighbors to try at each temperature
            reheat_threshold: Iterations without improvement before reheating
            reheat_factor: Factor to multiply temperature when reheating
        """
        super().__init__(problem, 'SA')
        
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.neighbors_per_temp = neighbors_per_temp
        self.reheat_threshold = reheat_threshold
        self.reheat_factor = reheat_factor
        
        # Search state
        self.current_solution = None
        self.current_cost = float('inf')
        self.temperature = initial_temperature
        self.iteration = 0
        self.stagnation_counter = 0
        self.reheat_count = 0
        
        # Statistics for adaptive control
        self.accepted_worse = 0
        self.total_moves = 0
        self.acceptance_history = []
    
    def initialize(self, **kwargs) -> None:
        """Initialize with random solution."""
        self.current_solution = self.problem.generate_random_solution()
        self.current_cost = self.problem.evaluate(self.current_solution)
        self.evaluations_used = 1
        
        self.update_best(self.current_solution, self.current_cost)
        
        self.temperature = self.initial_temperature
        self.iteration = 0
        self.stagnation_counter = 0
        self.reheat_count = 0
        self.accepted_worse = 0
        self.total_moves = 0
        self.acceptance_history = []
    
    def step(self, num_evaluations: int) -> Tuple[Any, float]:
        """
        Run SA for specified number of evaluations.
        
        Args:
            num_evaluations: Budget of function evaluations
            
        Returns:
            Tuple of (best_solution, best_cost)
        """
        evals_done = 0
        
        while evals_done < num_evaluations:
            # Try multiple moves at current temperature
            moves_at_temp = min(
                self.neighbors_per_temp,
                num_evaluations - evals_done
            )
            
            improved_this_temp = False
            
            for _ in range(moves_at_temp):
                if evals_done >= num_evaluations:
                    break
                
                # Generate neighbor
                neighbor = self._generate_neighbor()
                neighbor_cost = self.problem.evaluate(neighbor)
                evals_done += 1
                self.evaluations_used += 1
                self.total_moves += 1
                
                # Calculate acceptance probability
                delta = neighbor_cost - self.current_cost
                
                if delta < 0:
                    # Always accept improvements
                    accept = True
                elif self.temperature > 0:
                    # Probabilistically accept worse solutions
                    probability = np.exp(-delta / self.temperature)
                    accept = np.random.random() < probability
                    if accept:
                        self.accepted_worse += 1
                else:
                    accept = False
                
                if accept:
                    self.current_solution = neighbor.copy()
                    self.current_cost = neighbor_cost
                    
                    if self.update_best(neighbor, neighbor_cost):
                        improved_this_temp = True
                        self.stagnation_counter = 0
            
            # Cool down
            self.temperature *= self.cooling_rate
            self.temperature = max(self.temperature, self.min_temperature)
            self.iteration += 1
            
            # Track stagnation
            if not improved_this_temp:
                self.stagnation_counter += 1
            
            # Record acceptance rate
            if self.total_moves > 0:
                rate = self.accepted_worse / self.total_moves
                self.acceptance_history.append(rate)
            
            # Reheat if stuck
            if self.stagnation_counter >= self.reheat_threshold:
                self._reheat()
        
        return self.best_solution, self.best_cost
    
    def _generate_neighbor(self) -> np.ndarray:
        """Generate a neighbor solution using random operator."""
        n = len(self.current_solution)
        neighbor = self.current_solution.copy()
        
        # Choose operator based on temperature (more disruptive at high temp)
        if self.temperature > self.initial_temperature * 0.5:
            operators = ['2opt', 'insert', 'swap', 'or_opt']
        else:
            operators = ['2opt', 'swap']
        
        operator = np.random.choice(operators)
        
        if operator == 'swap':
            i, j = np.random.choice(n, 2, replace=False)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
        elif operator == '2opt':
            i = np.random.randint(0, n - 1)
            j = np.random.randint(i + 1, n)
            neighbor[i:j+1] = neighbor[i:j+1][::-1]
            
        elif operator == 'insert':
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                element = neighbor[i]
                neighbor = np.delete(neighbor, i)
                neighbor = np.insert(neighbor, j if j < i else j, element)
                
        elif operator == 'or_opt':
            # Move a segment of 1-3 elements
            seg_len = np.random.randint(1, min(4, n))
            start = np.random.randint(0, n - seg_len + 1)
            segment = neighbor[start:start + seg_len].copy()
            neighbor = np.delete(neighbor, range(start, start + seg_len))
            insert_pos = np.random.randint(0, len(neighbor) + 1)
            neighbor = np.insert(neighbor, insert_pos, segment)
        
        return neighbor
    
    def _reheat(self) -> None:
        """Reheat temperature to escape local optimum."""
        self.temperature = min(
            self.temperature * self.reheat_factor,
            self.initial_temperature * 0.5
        )
        self.stagnation_counter = 0
        self.reheat_count += 1
        
        # Optionally restart from best solution
        if self.reheat_count % 3 == 0 and self.best_solution is not None:
            self.current_solution = self.best_solution.copy()
            self.current_cost = self.best_cost
    
    def get_state(self) -> Dict[str, Any]:
        """Get algorithm state for context preservation."""
        return {
            'current_solution': self.current_solution.copy() if self.current_solution is not None else None,
            'current_cost': self.current_cost,
            'temperature': self.temperature,
            'iteration': self.iteration,
            'stagnation_counter': self.stagnation_counter,
            'reheat_count': self.reheat_count,
            'accepted_worse': self.accepted_worse,
            'total_moves': self.total_moves,
            'best_solution': self.best_solution.copy() if self.best_solution is not None else None,
            'best_cost': self.best_cost,
            'evaluations_used': self.evaluations_used
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore algorithm state from context."""
        self.current_solution = state['current_solution'].copy() if state['current_solution'] is not None else None
        self.current_cost = state['current_cost']
        self.temperature = state['temperature']
        self.iteration = state['iteration']
        self.stagnation_counter = state['stagnation_counter']
        self.reheat_count = state['reheat_count']
        self.accepted_worse = state['accepted_worse']
        self.total_moves = state['total_moves']
        self.best_solution = state['best_solution'].copy() if state['best_solution'] is not None else None
        self.best_cost = state['best_cost']
        self.evaluations_used = state['evaluations_used']
    
    def inject_solution(self, solution: Any, cost: float) -> None:
        """Inject a solution to continue search from."""
        self.current_solution = solution.copy()
        self.current_cost = cost
        self.update_best(solution, cost)
    
    def get_temperature_ratio(self) -> float:
        """Get current temperature as ratio of initial (for monitoring)."""
        return self.temperature / self.initial_temperature
