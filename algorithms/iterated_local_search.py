"""
Iterated Local Search (ILS) for discrete optimization.

Alternates between local search and perturbation to escape
local optima while maintaining solution quality.

Good for: Robust local search intensification, controlled exploration
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np

from .base_algorithm import BaseAlgorithm


class IteratedLocalSearch(BaseAlgorithm):
    """
    Iterated Local Search for discrete optimization.
    
    Features:
    - Perturbation + local search cycle
    - Adaptive perturbation strength
    - Acceptance criterion (better, random walk, simulated annealing)
    - History-based perturbation
    """
    
    def __init__(
        self,
        problem: Any,
        perturbation_strength: int = 3,
        local_search_max_iters: int = 30,
        acceptance_criterion: str = 'better',
        adaptive_perturbation: bool = True
    ):
        """
        Initialize ILS.
        
        Args:
            problem: Problem instance to solve
            perturbation_strength: Number of moves in perturbation (base)
            local_search_max_iters: Maximum local search iterations
            acceptance_criterion: 'better', 'random_walk', or 'sa'
            adaptive_perturbation: Adapt strength based on progress
        """
        super().__init__(problem, 'ILS')
        
        self.base_perturbation_strength = perturbation_strength
        self.perturbation_strength = perturbation_strength
        self.local_search_max_iters = local_search_max_iters
        self.acceptance_criterion = acceptance_criterion
        self.adaptive_perturbation = adaptive_perturbation
        
        # Search state
        self.current_solution = None
        self.current_cost = float('inf')
        self.home_base_solution = None  # Best local optimum found
        self.home_base_cost = float('inf')
        
        self.iteration = 0
        self.stagnation_counter = 0
        self.restarts = 0
        
        # SA parameters (if acceptance_criterion == 'sa')
        self.temperature = 1.0
        self.cooling_rate = 0.99
    
    def initialize(self, **kwargs) -> None:
        """Initialize with random solution and local search."""
        initial = self.problem.generate_random_solution()
        initial_cost = self.problem.evaluate(initial)
        self.evaluations_used = 1
        
        # Apply initial local search
        self.current_solution, self.current_cost, ls_evals = self._local_search(
            initial, initial_cost, self.local_search_max_iters
        )
        self.evaluations_used += ls_evals
        
        self.home_base_solution = self.current_solution.copy()
        self.home_base_cost = self.current_cost
        
        self.update_best(self.current_solution, self.current_cost)
        
        self.iteration = 0
        self.stagnation_counter = 0
        self.restarts = 0
        self.perturbation_strength = self.base_perturbation_strength
        self.temperature = 1.0
    
    def step(self, num_evaluations: int) -> Tuple[Any, float]:
        """
        Run ILS for specified number of evaluations.
        
        Args:
            num_evaluations: Budget of function evaluations
            
        Returns:
            Tuple of (best_solution, best_cost)
        """
        evals_done = 0
        
        while evals_done < num_evaluations:
            # Perturbation
            perturbed = self._perturb(self.current_solution)
            perturbed_cost = self.problem.evaluate(perturbed)
            evals_done += 1
            self.evaluations_used += 1
            
            if evals_done >= num_evaluations:
                break
            
            # Local search
            ls_budget = min(
                self.local_search_max_iters,
                num_evaluations - evals_done
            )
            locally_optimal, locally_optimal_cost, ls_evals = self._local_search(
                perturbed, perturbed_cost, ls_budget
            )
            evals_done += ls_evals
            
            # Acceptance decision
            accept = self._acceptance_decision(locally_optimal_cost, self.current_cost)
            
            if accept:
                self.current_solution = locally_optimal.copy()
                self.current_cost = locally_optimal_cost
                
                # Update home base if improved
                if locally_optimal_cost < self.home_base_cost:
                    self.home_base_solution = locally_optimal.copy()
                    self.home_base_cost = locally_optimal_cost
                    self.stagnation_counter = 0
                else:
                    self.stagnation_counter += 1
            else:
                self.stagnation_counter += 1
            
            # Update global best
            self.update_best(locally_optimal, locally_optimal_cost)
            
            # Adaptive perturbation strength
            if self.adaptive_perturbation:
                self._adapt_perturbation()
            
            # Restart if very stuck
            if self.stagnation_counter > 50:
                self._restart()
                evals_done += 1  # Account for restart evaluation
            
            self.iteration += 1
            
            # Cool down for SA acceptance
            if self.acceptance_criterion == 'sa':
                self.temperature *= self.cooling_rate
                self.temperature = max(self.temperature, 0.001)
        
        return self.best_solution, self.best_cost
    
    def _perturb(self, solution: np.ndarray) -> np.ndarray:
        """Apply perturbation to escape current local optimum."""
        perturbed = solution.copy()
        n = len(solution)
        
        # Apply multiple moves
        for _ in range(self.perturbation_strength):
            move = np.random.choice(['double_bridge', 'segment_shuffle', 'multi_swap'])
            
            if move == 'double_bridge' and n >= 8:
                # Double bridge move (good for TSP)
                # Cut tour into 4 segments and reconnect differently
                positions = sorted(np.random.choice(range(1, n), 4, replace=False))
                p1, p2, p3, p4 = positions
                
                perturbed = np.concatenate([
                    perturbed[:p1],
                    perturbed[p3:p4],
                    perturbed[p2:p3],
                    perturbed[p1:p2],
                    perturbed[p4:]
                ])
                
            elif move == 'segment_shuffle':
                # Shuffle a segment
                seg_len = min(np.random.randint(3, 6), n // 2)
                start = np.random.randint(0, n - seg_len)
                segment = perturbed[start:start + seg_len].copy()
                np.random.shuffle(segment)
                perturbed[start:start + seg_len] = segment
                
            else:  # multi_swap
                # Multiple random swaps
                num_swaps = np.random.randint(2, 4)
                for _ in range(num_swaps):
                    i, j = np.random.choice(n, 2, replace=False)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        
        return perturbed
    
    def _local_search(
        self, 
        solution: np.ndarray, 
        cost: float,
        max_evals: int
    ) -> Tuple[np.ndarray, float, int]:
        """
        First-improvement local search using 2-opt.
        
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
            
            # Random order for diversity
            i_order = np.random.permutation(n - 1)
            
            for i in i_order:
                if evals_used >= max_evals:
                    break
                
                for j in range(i + 2, n):
                    if evals_used >= max_evals:
                        break
                    
                    # 2-opt move
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
                        break
                
                if improved:
                    break
        
        return current, current_cost, evals_used
    
    def _acceptance_decision(self, new_cost: float, current_cost: float) -> bool:
        """Decide whether to accept new solution."""
        if self.acceptance_criterion == 'better':
            # Only accept improvements
            return new_cost < current_cost
        
        elif self.acceptance_criterion == 'random_walk':
            # Always accept
            return True
        
        elif self.acceptance_criterion == 'sa':
            # Simulated annealing acceptance
            if new_cost < current_cost:
                return True
            delta = new_cost - current_cost
            prob = np.exp(-delta / max(self.temperature, 1e-10))
            return np.random.random() < prob
        
        return new_cost <= current_cost
    
    def _adapt_perturbation(self) -> None:
        """Adapt perturbation strength based on search progress."""
        if self.stagnation_counter > 20:
            # Increase perturbation if stuck
            self.perturbation_strength = min(
                self.base_perturbation_strength * 3,
                len(self.current_solution) // 3
            )
        elif self.stagnation_counter < 5:
            # Decrease perturbation if making progress
            self.perturbation_strength = max(
                self.base_perturbation_strength // 2,
                2
            )
        else:
            self.perturbation_strength = self.base_perturbation_strength
    
    def _restart(self) -> None:
        """Restart from best solution or random."""
        if self.restarts % 2 == 0 and self.best_solution is not None:
            # Restart from best
            self.current_solution = self.best_solution.copy()
            self.current_cost = self.best_cost
        else:
            # Random restart
            self.current_solution = self.problem.generate_random_solution()
            self.current_cost = self.problem.evaluate(self.current_solution)
        
        self.home_base_solution = self.current_solution.copy()
        self.home_base_cost = self.current_cost
        self.stagnation_counter = 0
        self.restarts += 1
        self.perturbation_strength = self.base_perturbation_strength
    
    def get_state(self) -> Dict[str, Any]:
        """Get algorithm state for context preservation."""
        return {
            'current_solution': self.current_solution.copy() if self.current_solution is not None else None,
            'current_cost': self.current_cost,
            'home_base_solution': self.home_base_solution.copy() if self.home_base_solution is not None else None,
            'home_base_cost': self.home_base_cost,
            'iteration': self.iteration,
            'stagnation_counter': self.stagnation_counter,
            'restarts': self.restarts,
            'perturbation_strength': self.perturbation_strength,
            'temperature': self.temperature,
            'best_solution': self.best_solution.copy() if self.best_solution is not None else None,
            'best_cost': self.best_cost,
            'evaluations_used': self.evaluations_used
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore algorithm state from context."""
        self.current_solution = state['current_solution'].copy() if state['current_solution'] is not None else None
        self.current_cost = state['current_cost']
        self.home_base_solution = state['home_base_solution'].copy() if state['home_base_solution'] is not None else None
        self.home_base_cost = state['home_base_cost']
        self.iteration = state['iteration']
        self.stagnation_counter = state['stagnation_counter']
        self.restarts = state['restarts']
        self.perturbation_strength = state['perturbation_strength']
        self.temperature = state['temperature']
        self.best_solution = state['best_solution'].copy() if state['best_solution'] is not None else None
        self.best_cost = state['best_cost']
        self.evaluations_used = state['evaluations_used']
    
    def inject_solution(self, solution: Any, cost: float) -> None:
        """Inject a solution to continue search from."""
        self.current_solution = solution.copy()
        self.current_cost = cost
        self.home_base_solution = solution.copy()
        self.home_base_cost = cost
        self.update_best(solution, cost)
