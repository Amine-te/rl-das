"""
Ant Colony Optimization (ACO) for discrete optimization.

A swarm intelligence algorithm inspired by ant foraging behavior.
Particularly effective for path-based problems like TSP/VRP.

Good for: TSP/VRP, exploiting problem structure, implicit exploration
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from .base_algorithm import BaseAlgorithm


class AntColonyOptimization(BaseAlgorithm):
    """
    Ant Colony Optimization (MAX-MIN Ant System variant).
    
    Features:
    - Pheromone-guided construction
    - MAX-MIN pheromone bounds (avoids stagnation)
    - Local search integration
    - Pheromone smoothing
    """
    
    def __init__(
        self,
        problem: Any,
        num_ants: int = 20,
        alpha: float = 1.0,
        beta: float = 3.0,
        evaporation_rate: float = 0.1,
        q0: float = 0.9,
        local_search: bool = True,
        local_search_iters: int = 10
    ):
        """
        Initialize ACO.
        
        Args:
            problem: Problem instance (must have distance_matrix for TSP)
            num_ants: Number of ants in colony
            alpha: Pheromone importance factor
            beta: Heuristic importance factor
            evaporation_rate: Pheromone evaporation rate (0-1)
            q0: Exploitation vs exploration parameter
            local_search: Whether to apply local search to ant solutions
            local_search_iters: Max iterations for local search
        """
        super().__init__(problem, 'ACO')
        
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q0 = q0
        self.use_local_search = local_search
        self.local_search_iters = local_search_iters
        
        # Will be initialized based on problem size
        self.pheromone = None
        self.heuristic = None
        self.tau_max = None
        self.tau_min = None
        
        # Search state
        self.iteration = 0
        self.stagnation_counter = 0
        self.iteration_best = None
        self.iteration_best_cost = float('inf')
    
    def initialize(self, **kwargs) -> None:
        """Initialize pheromone matrix and heuristics."""
        n = self.problem.size
        
        # Initialize pheromone matrix
        # Use nearest neighbor heuristic to estimate initial pheromone level
        nn_tour = self._nearest_neighbor_tour()
        nn_cost = self.problem.evaluate(nn_tour)
        self.evaluations_used = 1
        
        self.update_best(nn_tour, nn_cost)
        
        # Initial pheromone level
        tau_0 = 1.0 / (n * nn_cost)
        self.tau_max = 1.0 / (self.evaporation_rate * nn_cost)
        self.tau_min = self.tau_max / (2 * n)
        
        self.pheromone = np.full((n, n), tau_0)
        
        # Heuristic information (inverse distance)
        if hasattr(self.problem, 'distance_matrix'):
            self.heuristic = 1.0 / (self.problem.distance_matrix + 1e-10)
            np.fill_diagonal(self.heuristic, 0)
        else:
            self.heuristic = np.ones((n, n))
        
        self.iteration = 0
        self.stagnation_counter = 0
        self.iteration_best = nn_tour.copy()
        self.iteration_best_cost = nn_cost
    
    def step(self, num_evaluations: int) -> Tuple[Any, float]:
        """
        Run ACO for specified number of evaluations.
        
        Args:
            num_evaluations: Budget of function evaluations
            
        Returns:
            Tuple of (best_solution, best_cost)
        """
        evals_done = 0
        
        while evals_done < num_evaluations:
            # Determine how many ants can run with remaining budget
            ants_to_run = min(
                self.num_ants,
                num_evaluations - evals_done
            )
            
            if ants_to_run <= 0:
                break
            
            # Construct solutions
            ant_solutions = []
            
            for _ in range(ants_to_run):
                tour = self._construct_solution()
                cost = self.problem.evaluate(tour)
                evals_done += 1
                self.evaluations_used += 1
                
                # Optional local search
                if self.use_local_search and evals_done + self.local_search_iters <= num_evaluations:
                    tour, cost, ls_evals = self._local_search(
                        tour, cost, 
                        min(self.local_search_iters, num_evaluations - evals_done)
                    )
                    evals_done += ls_evals
                
                ant_solutions.append((tour, cost))
                
                if evals_done >= num_evaluations:
                    break
            
            # Find best ant this iteration
            if ant_solutions:
                self.iteration_best, self.iteration_best_cost = min(
                    ant_solutions, key=lambda x: x[1]
                )
                
                # Update global best
                if self.update_best(self.iteration_best, self.iteration_best_cost):
                    self.stagnation_counter = 0
                else:
                    self.stagnation_counter += 1
            
            # Update pheromone
            self._update_pheromone()
            
            # Pheromone smoothing if stagnating
            if self.stagnation_counter > 20:
                self._smooth_pheromone()
                self.stagnation_counter = 0
            
            self.iteration += 1
        
        return self.best_solution, self.best_cost
    
    def _nearest_neighbor_tour(self) -> np.ndarray:
        """Generate tour using nearest neighbor heuristic."""
        n = self.problem.size
        start = np.random.randint(n)
        tour = [start]
        visited = {start}
        
        current = start
        for _ in range(n - 1):
            # Find nearest unvisited
            best_next = None
            best_dist = float('inf')
            
            for j in range(n):
                if j not in visited:
                    if hasattr(self.problem, 'distance_matrix'):
                        dist = self.problem.distance_matrix[current, j]
                    else:
                        dist = 1.0
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_next = j
            
            if best_next is not None:
                tour.append(best_next)
                visited.add(best_next)
                current = best_next
        
        return np.array(tour)
    
    def _construct_solution(self) -> np.ndarray:
        """Construct a tour using pheromone and heuristic information."""
        n = self.problem.size
        start = np.random.randint(n)
        tour = [start]
        visited = {start}
        current = start
        
        for _ in range(n - 1):
            # Calculate transition probabilities
            unvisited = [j for j in range(n) if j not in visited]
            
            if not unvisited:
                break
            
            # Exploitation vs exploration decision
            if np.random.random() < self.q0:
                # Exploitation: choose best
                scores = [
                    (self.pheromone[current, j] ** self.alpha) * 
                    (self.heuristic[current, j] ** self.beta)
                    for j in unvisited
                ]
                next_city = unvisited[np.argmax(scores)]
            else:
                # Exploration: probabilistic selection
                scores = np.array([
                    (self.pheromone[current, j] ** self.alpha) * 
                    (self.heuristic[current, j] ** self.beta)
                    for j in unvisited
                ])
                
                total = scores.sum()
                if total > 0:
                    probs = scores / total
                    next_city = np.random.choice(unvisited, p=probs)
                else:
                    next_city = np.random.choice(unvisited)
            
            tour.append(next_city)
            visited.add(next_city)
            current = next_city
        
        return np.array(tour)
    
    def _update_pheromone(self) -> None:
        """Update pheromone matrix (MAX-MIN variant)."""
        n = self.problem.size
        
        # Evaporation
        self.pheromone *= (1 - self.evaporation_rate)
        
        # Deposit pheromone (only best ant)
        if self.best_solution is not None:
            delta = 1.0 / self.best_cost
            tour = self.best_solution
            
            for i in range(n):
                j = (i + 1) % n
                self.pheromone[tour[i], tour[j]] += delta
                self.pheromone[tour[j], tour[i]] += delta
        
        # Apply MAX-MIN bounds
        self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)
    
    def _smooth_pheromone(self) -> None:
        """Smooth pheromone to escape stagnation."""
        # Reset pheromone towards uniform distribution
        avg = self.pheromone.mean()
        self.pheromone = 0.8 * self.pheromone + 0.2 * avg
    
    def _local_search(
        self, 
        solution: np.ndarray, 
        cost: float,
        max_evals: int
    ) -> Tuple[np.ndarray, float, int]:
        """
        Apply 2-opt local search.
        
        Returns:
            Tuple of (improved_solution, improved_cost, evaluations_used)
        """
        current = solution.copy()
        current_cost = cost
        evals_used = 0
        
        n = len(current)
        improved = True
        
        while improved and evals_used < max_evals:
            improved = False
            
            for i in range(n - 1):
                if evals_used >= max_evals:
                    break
                    
                for j in range(i + 2, n):
                    if evals_used >= max_evals:
                        break
                    
                    # 2-opt
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
    
    def get_state(self) -> Dict[str, Any]:
        """Get algorithm state for context preservation."""
        return {
            'pheromone': self.pheromone.copy() if self.pheromone is not None else None,
            'tau_max': self.tau_max,
            'tau_min': self.tau_min,
            'iteration': self.iteration,
            'stagnation_counter': self.stagnation_counter,
            'iteration_best': self.iteration_best.copy() if self.iteration_best is not None else None,
            'iteration_best_cost': self.iteration_best_cost,
            'best_solution': self.best_solution.copy() if self.best_solution is not None else None,
            'best_cost': self.best_cost,
            'evaluations_used': self.evaluations_used
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore algorithm state from context."""
        self.pheromone = state['pheromone'].copy() if state['pheromone'] is not None else None
        self.tau_max = state['tau_max']
        self.tau_min = state['tau_min']
        self.iteration = state['iteration']
        self.stagnation_counter = state['stagnation_counter']
        self.iteration_best = state['iteration_best'].copy() if state['iteration_best'] is not None else None
        self.iteration_best_cost = state['iteration_best_cost']
        self.best_solution = state['best_solution'].copy() if state['best_solution'] is not None else None
        self.best_cost = state['best_cost']
        self.evaluations_used = state['evaluations_used']
    
    def inject_solution(self, solution: Any, cost: float) -> None:
        """Inject a solution (updates pheromone to bias towards this solution)."""
        self.update_best(solution, cost)
        self.iteration_best = solution.copy()
        self.iteration_best_cost = cost
        
        # Bias pheromone towards injected solution
        if self.pheromone is not None:
            n = len(solution)
            boost = self.tau_max * 0.5
            for i in range(n):
                j = (i + 1) % n
                self.pheromone[solution[i], solution[j]] += boost
                self.pheromone[solution[j], solution[i]] += boost
            
            # Re-apply bounds
            self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)
