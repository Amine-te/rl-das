"""
Tabu Search (TS) for discrete optimization.

A local search method that uses memory structures to avoid
cycling and escape local optima.

Good for: Intensification, systematic neighborhood exploration
"""

from typing import Any, Dict, List, Tuple, Optional, Set
from collections import deque
import numpy as np

from .base_algorithm import BaseAlgorithm


class TabuSearch(BaseAlgorithm):
    """
    Tabu Search for discrete optimization.
    
    Features:
    - Short-term memory (tabu list) to prevent cycling
    - Aspiration criterion (accept tabu move if it's the best ever)
    - Adaptive tabu tenure based on search progress
    - Intensification and diversification mechanisms
    """
    
    def __init__(
        self,
        problem: Any,
        tabu_tenure: int = 10,
        neighborhood_size: int = 50,
        aspiration_enabled: bool = True,
        intensification_threshold: int = 10,
        diversification_threshold: int = 20
    ):
        """
        Initialize Tabu Search.
        
        Args:
            problem: Problem instance to solve
            tabu_tenure: Number of iterations a move stays tabu
            neighborhood_size: Number of neighbors to explore per iteration
            aspiration_enabled: Whether to use aspiration criterion
            intensification_threshold: Iterations without improvement before intensification
            diversification_threshold: Iterations without improvement before diversification
        """
        super().__init__(problem, 'TS')
        
        self.base_tabu_tenure = tabu_tenure
        self.tabu_tenure = tabu_tenure
        self.neighborhood_size = neighborhood_size
        self.aspiration_enabled = aspiration_enabled
        self.intensification_threshold = intensification_threshold
        self.diversification_threshold = diversification_threshold
        
        # Search state
        self.current_solution = None
        self.current_cost = float('inf')
        self.tabu_list: deque = deque(maxlen=tabu_tenure * 2)
        self.iteration = 0
        self.stagnation_counter = 0
        self.last_improvement_iteration = 0
        
        # Elite solutions for intensification
        self.elite_solutions: List[Tuple[Any, float]] = []
        self.elite_size = 5
        
        # Frequency-based diversification
        self.move_frequency: Dict[Tuple, int] = {}
    
    def initialize(self, **kwargs) -> None:
        """Initialize with random solution."""
        self.current_solution = self.problem.generate_random_solution()
        self.current_cost = self.problem.evaluate(self.current_solution)
        self.evaluations_used = 1
        
        self.update_best(self.current_solution, self.current_cost)
        
        self.tabu_list = deque(maxlen=self.tabu_tenure * 2)
        self.iteration = 0
        self.stagnation_counter = 0
        self.last_improvement_iteration = 0
        self.tabu_tenure = self.base_tabu_tenure
        self.elite_solutions = [(self.current_solution.copy(), self.current_cost)]
        self.move_frequency = {}
    
    def step(self, num_evaluations: int) -> Tuple[Any, float]:
        """
        Run Tabu Search for specified number of evaluations.
        
        Args:
            num_evaluations: Budget of function evaluations
            
        Returns:
            Tuple of (best_solution, best_cost)
        """
        evals_done = 0
        
        while evals_done < num_evaluations:
            # Generate neighborhood
            neighbors_to_generate = min(
                self.neighborhood_size, 
                num_evaluations - evals_done
            )
            
            if neighbors_to_generate <= 0:
                break
            
            neighbors = self._generate_neighborhood(neighbors_to_generate)
            
            # Evaluate neighbors
            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_move = None
            
            for neighbor, move in neighbors:
                cost = self.problem.evaluate(neighbor)
                evals_done += 1
                self.evaluations_used += 1
                
                # Check if move is tabu
                is_tabu = self._is_tabu(move)
                
                # Aspiration criterion: accept if best ever
                aspiration = (
                    self.aspiration_enabled and 
                    cost < self.best_cost
                )
                
                # Accept if not tabu or satisfies aspiration
                if (not is_tabu or aspiration) and cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = cost
                    best_move = move
                
                if evals_done >= num_evaluations:
                    break
            
            # Move to best neighbor
            if best_neighbor is not None:
                self.current_solution = best_neighbor.copy()
                self.current_cost = best_neighbor_cost
                
                # Add move to tabu list
                if best_move is not None:
                    self._add_to_tabu(best_move)
                    self._update_frequency(best_move)
                
                # Update best
                if self.update_best(best_neighbor, best_neighbor_cost):
                    self.stagnation_counter = 0
                    self.last_improvement_iteration = self.iteration
                    self._update_elite(best_neighbor, best_neighbor_cost)
                else:
                    self.stagnation_counter += 1
            else:
                self.stagnation_counter += 1
            
            self.iteration += 1
            
            # Adaptive tabu tenure
            self._adapt_tenure()
            
            # Intensification: go to best elite if stagnating
            if self.stagnation_counter == self.intensification_threshold:
                self._intensify()
            
            # Diversification: random restart if very stagnant
            if self.stagnation_counter >= self.diversification_threshold:
                self._diversify()
        
        return self.best_solution, self.best_cost
    
    def _generate_neighborhood(self, size: int) -> List[Tuple[Any, Tuple]]:
        """Generate neighbors with their corresponding moves."""
        neighbors = []
        n = len(self.current_solution)
        
        for _ in range(size):
            # Random move type
            move_type = np.random.choice(['swap', '2opt', 'insert'])
            
            if move_type == 'swap':
                i, j = sorted(np.random.choice(n, 2, replace=False))
                neighbor = self.current_solution.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                move = ('swap', i, j)
                
            elif move_type == '2opt':
                i = np.random.randint(0, n - 1)
                j = np.random.randint(i + 1, n)
                neighbor = self.current_solution.copy()
                neighbor[i:j+1] = neighbor[i:j+1][::-1]
                move = ('2opt', i, j)
                
            else:  # insert
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    neighbor = list(self.current_solution)
                    element = neighbor.pop(i)
                    neighbor.insert(j, element)
                    neighbor = np.array(neighbor)
                    move = ('insert', i, j)
                else:
                    neighbor = self.current_solution.copy()
                    move = ('insert', i, i)
            
            neighbors.append((neighbor, move))
        
        return neighbors
    
    def _is_tabu(self, move: Tuple) -> bool:
        """Check if a move is in the tabu list."""
        # Also check reverse moves
        if move[0] == 'swap':
            return move in self.tabu_list or ('swap', move[2], move[1]) in self.tabu_list
        elif move[0] == '2opt':
            return move in self.tabu_list
        else:
            return move in self.tabu_list
    
    def _add_to_tabu(self, move: Tuple) -> None:
        """Add move to tabu list."""
        self.tabu_list.append(move)
    
    def _update_frequency(self, move: Tuple) -> None:
        """Update move frequency for diversification."""
        key = (move[0], min(move[1], move[2]) if len(move) > 2 else move[1])
        self.move_frequency[key] = self.move_frequency.get(key, 0) + 1
    
    def _adapt_tenure(self) -> None:
        """Adapt tabu tenure based on search progress."""
        if self.stagnation_counter > 10:
            # Increase tenure to escape
            self.tabu_tenure = min(self.base_tabu_tenure * 3, len(self.current_solution))
        elif self.stagnation_counter < 3:
            # Decrease tenure for exploitation
            self.tabu_tenure = max(self.base_tabu_tenure // 2, 5)
        else:
            self.tabu_tenure = self.base_tabu_tenure
        
        self.tabu_list = deque(self.tabu_list, maxlen=self.tabu_tenure * 2)
    
    def _update_elite(self, solution: Any, cost: float) -> None:
        """Update elite solution archive."""
        self.elite_solutions.append((solution.copy(), cost))
        self.elite_solutions.sort(key=lambda x: x[1])
        self.elite_solutions = self.elite_solutions[:self.elite_size]
    
    def _intensify(self) -> None:
        """Intensify search around best solutions."""
        if self.elite_solutions:
            # Go to best elite solution
            best_elite = self.elite_solutions[0]
            self.current_solution = best_elite[0].copy()
            self.current_cost = best_elite[1]
            # Clear tabu list to allow new exploration
            self.tabu_list.clear()
    
    def _diversify(self) -> None:
        """Diversify search to escape stagnation."""
        # Generate new solution biased away from frequent moves
        self.current_solution = self.problem.generate_random_solution()
        self.current_cost = self.problem.evaluate(self.current_solution)
        self.evaluations_used += 1
        
        self.update_best(self.current_solution, self.current_cost)
        
        # Clear tabu list
        self.tabu_list.clear()
        self.stagnation_counter = 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get algorithm state for context preservation."""
        return {
            'current_solution': self.current_solution.copy() if self.current_solution is not None else None,
            'current_cost': self.current_cost,
            'tabu_list': list(self.tabu_list),
            'tabu_tenure': self.tabu_tenure,
            'iteration': self.iteration,
            'stagnation_counter': self.stagnation_counter,
            'last_improvement_iteration': self.last_improvement_iteration,
            'elite_solutions': [(s.copy(), c) for s, c in self.elite_solutions],
            'best_solution': self.best_solution.copy() if self.best_solution is not None else None,
            'best_cost': self.best_cost,
            'evaluations_used': self.evaluations_used
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore algorithm state from context."""
        self.current_solution = state['current_solution'].copy() if state['current_solution'] is not None else None
        self.current_cost = state['current_cost']
        self.tabu_list = deque(state['tabu_list'], maxlen=state['tabu_tenure'] * 2)
        self.tabu_tenure = state['tabu_tenure']
        self.iteration = state['iteration']
        self.stagnation_counter = state['stagnation_counter']
        self.last_improvement_iteration = state['last_improvement_iteration']
        self.elite_solutions = [(s.copy(), c) for s, c in state['elite_solutions']]
        self.best_solution = state['best_solution'].copy() if state['best_solution'] is not None else None
        self.best_cost = state['best_cost']
        self.evaluations_used = state['evaluations_used']
    
    def inject_solution(self, solution: Any, cost: float) -> None:
        """Inject a solution to continue search from."""
        self.current_solution = solution.copy()
        self.current_cost = cost
        self.update_best(solution, cost)
        self._update_elite(solution, cost)
