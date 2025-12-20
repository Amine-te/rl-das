"""
Context manager for RL-DAS.

Manages algorithm state preservation for warm-start switching.
This is crucial for effective dynamic algorithm selection.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from copy import deepcopy


class ContextManager:
    """
    Manage algorithm context for warm-start switching.
    
    The context manager stores:
    1. Algorithm-specific state for each algorithm
    2. Common context shared across all algorithms
    
    This enables:
    - Resuming an algorithm from where it left off
    - Sharing elite solutions across algorithms
    - Tracking global best across all algorithms
    """
    
    def __init__(self, num_algorithms: int, elite_size: int = 5):
        """
        Initialize context manager.
        
        Args:
            num_algorithms: Number of algorithms in the pool
            elite_size: Number of elite solutions to maintain
        """
        self.num_algorithms = num_algorithms
        self.elite_size = elite_size
        self.context = self._create_empty_context()
    
    def _create_empty_context(self) -> Dict[str, Any]:
        """Create empty context structure."""
        return {
            'algorithms': {
                i: {
                    'state': None,
                    'initialized': False,
                    'last_used_fes': 0,
                    'best_cost_achieved': float('inf'),
                    'times_selected': 0
                }
                for i in range(self.num_algorithms)
            },
            'common': {
                'global_best_solution': None,
                'global_best_cost': float('inf'),
                'elite_archive': [],  # List of (solution, cost) tuples
                'visited_hashes': set(),  # For avoiding revisits
                'best_per_algorithm': {i: float('inf') for i in range(self.num_algorithms)},
                'total_fes_used': 0,
                'switch_count': 0,
                'last_algorithm': None
            }
        }
    
    def reset(self):
        """Reset context for new episode."""
        self.context = self._create_empty_context()
    
    def save_algorithm_context(self, algo_idx: int, algorithm: Any):
        """
        Save algorithm's internal state to context.
        
        Args:
            algo_idx: Index of the algorithm
            algorithm: Algorithm instance (must have get_state method)
        """
        if algo_idx < 0 or algo_idx >= self.num_algorithms:
            raise ValueError(f"Invalid algorithm index: {algo_idx}")
        
        algo_context = self.context['algorithms'][algo_idx]
        
        # Save algorithm state
        if hasattr(algorithm, 'get_state'):
            algo_context['state'] = deepcopy(algorithm.get_state())
        
        algo_context['initialized'] = True
        algo_context['last_used_fes'] = self.context['common']['total_fes_used']
        
        # Update best cost for this algorithm
        if hasattr(algorithm, 'best_cost'):
            algo_context['best_cost_achieved'] = min(
                algo_context['best_cost_achieved'],
                algorithm.best_cost
            )
            self.context['common']['best_per_algorithm'][algo_idx] = algo_context['best_cost_achieved']
    
    def restore_algorithm_context(self, algo_idx: int, algorithm: Any) -> bool:
        """
        Restore algorithm's state from context.
        
        Args:
            algo_idx: Index of the algorithm
            algorithm: Algorithm instance (must have set_state method)
            
        Returns:
            True if context was restored, False if cold start needed
        """
        if algo_idx < 0 or algo_idx >= self.num_algorithms:
            raise ValueError(f"Invalid algorithm index: {algo_idx}")
        
        algo_context = self.context['algorithms'][algo_idx]
        
        if algo_context['initialized'] and algo_context['state'] is not None:
            # Warm start: restore saved state
            if hasattr(algorithm, 'set_state'):
                algorithm.set_state(deepcopy(algo_context['state']))
            return True
        else:
            # Cold start: inject best known solution
            best_sol, best_cost = self.get_warm_start_solution()
            if best_sol is not None and hasattr(algorithm, 'inject_solution'):
                algorithm.inject_solution(best_sol, best_cost)
            return False
    
    def update_common_context(
        self,
        solution: Any,
        cost: float,
        algo_idx: int,
        fes_used: int
    ):
        """
        Update shared context with new discoveries.
        
        Args:
            solution: New solution found
            cost: Cost of the solution
            algo_idx: Algorithm that found it
            fes_used: Total FEs used so far
        """
        common = self.context['common']
        
        # Update global best
        if cost < common['global_best_cost']:
            common['global_best_solution'] = deepcopy(solution)
            common['global_best_cost'] = cost
        
        # Update elite archive
        self._update_elite_archive(solution, cost)
        
        # Update per-algorithm tracking
        if algo_idx >= 0:
            common['best_per_algorithm'][algo_idx] = min(
                common['best_per_algorithm'][algo_idx],
                cost
            )
        
        common['total_fes_used'] = fes_used
    
    def _update_elite_archive(self, solution: Any, cost: float):
        """
        Update elite archive with new solution if it qualifies.
        
        Maintains diversity in the archive.
        """
        archive = self.context['common']['elite_archive']
        
        # Check if solution is better than worst in archive
        if len(archive) < self.elite_size:
            archive.append((deepcopy(solution), cost))
            archive.sort(key=lambda x: x[1])
        elif cost < archive[-1][1]:
            # Replace worst solution
            archive[-1] = (deepcopy(solution), cost)
            archive.sort(key=lambda x: x[1])
    
    def get_warm_start_solution(self) -> Tuple[Optional[Any], float]:
        """
        Get best solution for new algorithm initialization.
        
        Returns:
            Tuple of (best_solution, best_cost), or (None, inf) if no solution
        """
        common = self.context['common']
        
        if common['global_best_solution'] is not None:
            return deepcopy(common['global_best_solution']), common['global_best_cost']
        
        if common['elite_archive']:
            best = common['elite_archive'][0]
            return deepcopy(best[0]), best[1]
        
        return None, float('inf')
    
    def get_elite_solutions(self) -> List[Tuple[Any, float]]:
        """
        Get all elite solutions for population initialization.
        
        Returns:
            List of (solution, cost) tuples
        """
        return [
            (deepcopy(sol), cost) 
            for sol, cost in self.context['common']['elite_archive']
        ]
    
    def record_switch(self, from_algo: Optional[int], to_algo: int):
        """
        Record algorithm switch for analysis.
        
        Args:
            from_algo: Previous algorithm (None if first selection)
            to_algo: New algorithm being selected
        """
        common = self.context['common']
        
        if from_algo != to_algo:
            common['switch_count'] += 1
        
        common['last_algorithm'] = to_algo
        self.context['algorithms'][to_algo]['times_selected'] += 1
    
    def get_algorithm_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        Get statistics for each algorithm.
        
        Returns:
            Dictionary mapping algorithm index to stats
        """
        stats = {}
        for idx in range(self.num_algorithms):
            algo_ctx = self.context['algorithms'][idx]
            stats[idx] = {
                'times_selected': algo_ctx['times_selected'],
                'best_cost': algo_ctx['best_cost_achieved'],
                'initialized': algo_ctx['initialized']
            }
        return stats
    
    def get_global_best(self) -> Tuple[Optional[Any], float]:
        """Get global best solution and cost."""
        common = self.context['common']
        return common['global_best_solution'], common['global_best_cost']
    
    def get_switch_count(self) -> int:
        """Get number of algorithm switches this episode."""
        return self.context['common']['switch_count']
