"""
Variable Neighborhood Search (VNS) for discrete optimization.

A metaheuristic that systematically changes neighborhood structures
during the search process.

Good for: Escaping local optima, systematic exploration of solution space
"""

from typing import Any, Dict, List, Tuple, Callable
import numpy as np

from .base_algorithm import BaseAlgorithm


class VariableNeighborhoodSearch(BaseAlgorithm):
    """
    Variable Neighborhood Search for discrete optimization.
    
    Features:
    - Multiple neighborhood structures (shaking)
    - Local search in current neighborhood  
    - Systematic neighborhood change when stuck
    - Optional reduced VNS mode for efficiency
    """
    
    def __init__(
        self,
        problem: Any,
        k_max: int = 5,
        local_search_iterations: int = 20,
        reduced_vns: bool = False
    ):
        """
        Initialize VNS.
        
        Args:
            problem: Problem instance to solve
            k_max: Maximum neighborhood index
            local_search_iterations: Max iterations for local search
            reduced_vns: If True, skip local search (faster but less optimal)
        """
        super().__init__(problem, 'VNS')
        
        self.k_max = k_max
        self.local_search_iterations = local_search_iterations
        self.reduced_vns = reduced_vns
        
        # Search state
        self.current_solution = None
        self.current_cost = float('inf')
        self.k = 1  # Current neighborhood
        self.iteration = 0
        self.neighborhood_visits = {i: 0 for i in range(1, k_max + 1)}
    
    def initialize(self, **kwargs) -> None:
        """Initialize with random solution."""
        self.current_solution = self.problem.generate_random_solution()
        self.current_cost = self.problem.evaluate(self.current_solution)
        self.evaluations_used = 1
        
        self.update_best(self.current_solution, self.current_cost)
        
        self.k = 1
        self.iteration = 0
        self.neighborhood_visits = {i: 0 for i in range(1, self.k_max + 1)}
    
    def step(self, num_evaluations: int) -> Tuple[Any, float]:
        """
        Run VNS for specified number of evaluations.
        
        Args:
            num_evaluations: Budget of function evaluations
            
        Returns:
            Tuple of (best_solution, best_cost)
        """
        evals_done = 0
        
        while evals_done < num_evaluations:
            # Shaking: generate random solution in k-th neighborhood
            x_prime = self._shake(self.current_solution, self.k)
            x_prime_cost = self.problem.evaluate(x_prime)
            evals_done += 1
            self.evaluations_used += 1
            
            if evals_done >= num_evaluations:
                break
            
            # Local search (unless reduced VNS)
            if not self.reduced_vns:
                evals_for_ls = min(
                    self.local_search_iterations,
                    num_evaluations - evals_done
                )
                x_double_prime, x_double_prime_cost, ls_evals = self._local_search(
                    x_prime, x_prime_cost, evals_for_ls
                )
                evals_done += ls_evals
            else:
                x_double_prime = x_prime
                x_double_prime_cost = x_prime_cost
            
            # Move or not (neighborhood change)
            if x_double_prime_cost < self.current_cost:
                # Improvement found: move and reset to first neighborhood
                self.current_solution = x_double_prime.copy()
                self.current_cost = x_double_prime_cost
                self.update_best(x_double_prime, x_double_prime_cost)
                self.k = 1  # Reset to smallest neighborhood
            else:
                # No improvement: move to next neighborhood
                self.k += 1
                if self.k > self.k_max:
                    self.k = 1  # Cycle back
            
            self.neighborhood_visits[self.k] += 1
            self.iteration += 1
        
        return self.best_solution, self.best_cost
    
    def _shake(self, solution: np.ndarray, k: int) -> np.ndarray:
        """
        Generate random solution in k-th neighborhood.
        
        Larger k = more disruptive perturbation.
        """
        shaken = solution.copy()
        n = len(solution)
        
        # Apply k random moves
        for _ in range(k):
            move_type = np.random.choice(['swap', '2opt', 'insert', 'or_opt', 'shuffle'])
            
            if move_type == 'swap':
                i, j = np.random.choice(n, 2, replace=False)
                shaken[i], shaken[j] = shaken[j], shaken[i]
                
            elif move_type == '2opt':
                i = np.random.randint(0, n - 1)
                j = np.random.randint(i + 1, n)
                shaken[i:j+1] = shaken[i:j+1][::-1]
                
            elif move_type == 'insert':
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    elem = shaken[i]
                    shaken = np.delete(shaken, i)
                    shaken = np.insert(shaken, j if j < i else j, elem)
                    
            elif move_type == 'or_opt':
                seg_len = min(np.random.randint(1, 4), n - 1)
                start = np.random.randint(0, n - seg_len)
                segment = shaken[start:start + seg_len].copy()
                shaken = np.delete(shaken, range(start, start + seg_len))
                insert_pos = np.random.randint(0, len(shaken) + 1)
                shaken = np.insert(shaken, insert_pos, segment)
                
            else:  # shuffle
                # Shuffle a random segment
                seg_len = min(np.random.randint(3, 6), n)
                start = np.random.randint(0, n - seg_len + 1)
                segment = shaken[start:start + seg_len].copy()
                np.random.shuffle(segment)
                shaken[start:start + seg_len] = segment
        
        return shaken
    
    def _local_search(
        self, 
        solution: np.ndarray, 
        cost: float, 
        max_evals: int
    ) -> Tuple[np.ndarray, float, int]:
        """
        Perform local search from given solution.
        
        Uses first-improvement strategy with 2-opt.
        
        Returns:
            Tuple of (improved_solution, improved_cost, evaluations_used)
        """
        current = solution.copy()
        current_cost = cost
        evals_used = 0
        improved = True
        
        while improved and evals_used < max_evals:
            improved = False
            n = len(current)
            
            # Try 2-opt moves
            for i in range(n - 1):
                if evals_used >= max_evals:
                    break
                    
                for j in range(i + 2, n):
                    if evals_used >= max_evals:
                        break
                    
                    # Apply 2-opt
                    neighbor = current.copy()
                    neighbor[i:j+1] = neighbor[i:j+1][::-1]
                    
                    neighbor_cost = self.problem.evaluate(neighbor)
                    evals_used += 1
                    self.evaluations_used += 1
                    
                    if neighbor_cost < current_cost:
                        current = neighbor
                        current_cost = neighbor_cost
                        self.update_best(neighbor, neighbor_cost)
                        improved = True
                        break  # First improvement
                
                if improved:
                    break
        
        return current, current_cost, evals_used
    
    def get_state(self) -> Dict[str, Any]:
        """Get algorithm state for context preservation."""
        return {
            'current_solution': self.current_solution.copy() if self.current_solution is not None else None,
            'current_cost': self.current_cost,
            'k': self.k,
            'iteration': self.iteration,
            'neighborhood_visits': self.neighborhood_visits.copy(),
            'best_solution': self.best_solution.copy() if self.best_solution is not None else None,
            'best_cost': self.best_cost,
            'evaluations_used': self.evaluations_used
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore algorithm state from context."""
        self.current_solution = state['current_solution'].copy() if state['current_solution'] is not None else None
        self.current_cost = state['current_cost']
        self.k = state['k']
        self.iteration = state['iteration']
        self.neighborhood_visits = state['neighborhood_visits'].copy()
        self.best_solution = state['best_solution'].copy() if state['best_solution'] is not None else None
        self.best_cost = state['best_cost']
        self.evaluations_used = state['evaluations_used']
    
    def inject_solution(self, solution: Any, cost: float) -> None:
        """Inject a solution to continue search from."""
        self.current_solution = solution.copy()
        self.current_cost = cost
        self.update_best(solution, cost)
        self.k = 1  # Reset neighborhood
