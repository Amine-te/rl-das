"""
DAS Environment for RL-DAS.

Core orchestration layer that ties together:
- State extraction
- Reward calculation
- Context management
- Algorithm execution
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .state_extractor import StateExtractor
from .reward_calculator import RewardCalculator
from .context_manager import ContextManager


class DASEnvironment:
    """
    Dynamic Algorithm Selection Environment.
    
    This environment manages the optimization process where an RL agent
    selects which algorithm to run at each decision point.
    
    Episode structure:
        [Init] → [Interval 1] → [Interval 2] → ... → [Terminate]
                 ↓ agent selects
                 execute algorithm for interval_fes
    """
    
    def __init__(
        self,
        problem: Any,
        algorithms: List[Any],
        max_fes: int = 100000,
        interval_fes: int = 2000,
        neighbor_sample_size: int = 20,
        population_size: int = 50,
        stagnation_limit: Optional[int] = None
    ):
        """
        Initialize DAS environment.
        
        Args:
            problem: Problem instance (implements BaseProblem)
            algorithms: List of algorithm instances (implement BaseAlgorithm)
            max_fes: Maximum function evaluations per episode
            interval_fes: FEs per decision interval
            neighbor_sample_size: Neighbors to sample for LA features
            population_size: Population size for tracking
            stagnation_limit: Optional limit on intervals without improvement
        """
        self.problem = problem
        self.algorithms = algorithms
        self.num_algorithms = len(algorithms)
        self.max_fes = max_fes
        self.interval_fes = interval_fes
        self.population_size = population_size
        self.stagnation_limit = stagnation_limit
        
        # Initialize components
        self.state_extractor = StateExtractor(num_algorithms=self.num_algorithms)
        self.reward_calculator = RewardCalculator(max_fes=max_fes, num_algorithms=self.num_algorithms)
        self.context_manager = ContextManager(
            num_algorithms=self.num_algorithms,
            elite_size=5
        )
        
        # Episode state
        self.current_fes = 0
        self.current_algo_idx: Optional[int] = None
        self.last_action: Optional[int] = None  # For switching bonus
        self.population: List[Tuple[Any, float]] = []
        self.episode_step = 0
        self.stagnation_counter = 0
        self.last_best_cost = float('inf')
        self.done = False
    
    @property
    def state_dim(self) -> int:
        """Dimension of state vector."""
        return self.state_extractor.state_dim
    
    @property
    def action_dim(self) -> int:
        """Number of actions (algorithms)."""
        return self.num_algorithms
    
    def reset(self, problem_instance: Optional[Any] = None) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Args:
            problem_instance: Optional new problem instance
            
        Returns:
            Initial state vector
        """
        # Reset problem if provided
        if problem_instance is not None:
            self.problem = problem_instance
        
        self.problem.reset_evaluation_count()
        
        # Reset components
        self.state_extractor.reset()
        self.context_manager.reset()
        
        # Reset episode state
        self.current_fes = 0
        self.current_algo_idx = None
        self.last_action = None
        self.episode_step = 0
        self.stagnation_counter = 0
        self.cumulative_reward = 0.0
        self.done = False
        
        # Initialize all algorithms
        for algo in self.algorithms:
            algo.initialize()
        
        # Generate initial population
        self._initialize_population()
        
        # Get best solution
        best_sol, best_cost = self._get_population_best()
        self.last_best_cost = best_cost
        
        # Initialize reward calculator
        self.reward_calculator.reset(best_cost)
        
        # Update context with initial best
        self.context_manager.update_common_context(
            best_sol, best_cost, -1, self.current_fes
        )
        
        # Extract initial state (no current algorithm yet)
        state = self.state_extractor.extract_state(
            self.problem,
            best_sol,
            best_cost,
            self.population,
            self.current_fes,
            self.max_fes,
            current_algo_idx=None
        )
        
        return state
    
    def _initialize_population(self):
        """Generate initial population of solutions."""
        self.population = []
        
        for _ in range(self.population_size):
            sol = self.problem.generate_random_solution()
            cost = self.problem.evaluate(sol)
            self.population.append((sol, cost))
            self.current_fes += 1
        
        # Sort by cost
        self.population.sort(key=lambda x: x[1])
    
    def _get_population_best(self) -> Tuple[Any, float]:
        """Get best solution from population."""
        if not self.population:
            raise ValueError("Population is empty")
        return self.population[0]
    
    def _get_population_worst(self) -> Tuple[Any, float]:
        """Get worst solution from population."""
        if not self.population:
            raise ValueError("Population is empty")
        return self.population[-1]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action (algorithm selection) and return results.
        
        Args:
            action: Index of algorithm to execute
            
        Returns:
            Tuple of (state, reward, terminated, truncated, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")
        
        if action < 0 or action >= self.num_algorithms:
            raise ValueError(f"Invalid action: {action}")
        
        # Record algorithm switch
        self.context_manager.record_switch(self.current_algo_idx, action)
        
        # Save current algorithm's context if switching
        if self.current_algo_idx is not None and self.current_algo_idx != action:
            self.context_manager.save_algorithm_context(
                self.current_algo_idx,
                self.algorithms[self.current_algo_idx]
            )
        
        # Restore/initialize the selected algorithm
        algo = self.algorithms[action]
        was_warm_start = self.context_manager.restore_algorithm_context(action, algo)
        
        # Track solutions before step (for AH features)
        self.best_before_step = self.problem.copy_solution(self.population[0][0])
        self.worst_before_step = self.problem.copy_solution(self.population[-1][0])
        prev_best_cost = self.population[0][1]
        
        # Execute algorithm for interval
        start_fes = self.current_fes
        new_best, new_cost = self._execute_algorithm(action)
        fes_used = self.current_fes - start_fes
        progress = self.current_fes / self.max_fes
        
        # Determine if we switched
        switched = self.last_action is not None and action != self.last_action
        
        # Get credibility of selected algorithm
        credibility = self.state_extractor.credibility_scores[action]
        
        # Compute improvement magnitude for state update
        if self.reward_calculator.initial_cost and self.reward_calculator.initial_cost > 0:
            improvement_magnitude = (prev_best_cost - new_cost) / self.reward_calculator.initial_cost
        else:
            improvement_magnitude = 0.0
        
        # Compute reward with new parameters
        reward = self.reward_calculator.compute_step_reward(
            prev_cost=prev_best_cost,
            curr_cost=new_cost,
            fes_used=fes_used,
            progress=progress,
            switched=switched,
            credibility=credibility,
            algo_idx=action
        )
        self.cumulative_reward += reward
        
        # Update stagnation tracking with improvement magnitude
        improved = new_cost < prev_best_cost
        self.state_extractor.update_stagnation(action, improved, improvement_magnitude)
        
        # Update context
        self.context_manager.update_common_context(
            new_best, new_cost, action, self.current_fes
        )
        
        # Check for improvement
        is_global_best = new_cost < self.last_best_cost
        if is_global_best:
            self.stagnation_counter = 0
            self.last_best_cost = new_cost
        else:
            self.stagnation_counter += 1
        
        # Check termination
        terminated = self._is_terminated()
        truncated = False
        
        if terminated:
            self.done = True
            # Save final context
            self.context_manager.save_algorithm_context(action, algo)
        
        # Extract new state (include current algorithm)
        state = self.state_extractor.extract_state(
            self.problem,
            new_best,
            new_cost,
            self.population,
            self.current_fes,
            self.max_fes,
            current_algo_idx=action
        )
        
        # Update tracking
        self.current_algo_idx = action
        self.last_action = action
        self.episode_step += 1
        
        # Build info dict
        info = {
            'current_fes': self.current_fes,
            'best_cost': new_cost,
            'algorithm_used': action,
            'was_warm_start': was_warm_start,
            'is_global_best': is_global_best,
            'stagnation_counter': self.stagnation_counter,
            'episode_step': self.episode_step
        }
        
        return state, reward, terminated, truncated, info
    
    def _execute_algorithm(self, algo_idx: int) -> Tuple[Any, float]:
        """
        Run selected algorithm for one interval.
        
        Args:
            algo_idx: Index of algorithm to run
            
        Returns:
            Tuple of (best_solution, best_cost) after execution
        """
        algo = self.algorithms[algo_idx]
        
        # Inject current best into algorithm
        best_sol, best_cost = self._get_population_best()
        algo.inject_solution(self.problem.copy_solution(best_sol), best_cost)
        
        # Execute for interval
        fes_to_use = min(self.interval_fes, self.max_fes - self.current_fes)
        new_best, new_cost = algo.step(fes_to_use)
        
        # Update FEs used
        self.current_fes = self.problem.get_evaluation_count()
        
        # Update population with algorithm's discoveries
        if new_cost < self.population[-1][1]:
            # Replace worst solution
            self.population[-1] = (self.problem.copy_solution(new_best), new_cost)
            self.population.sort(key=lambda x: x[1])
        
        return self.population[0][0], self.population[0][1]
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Budget exhausted
        if self.current_fes >= self.max_fes:
            return True
        
        # Stagnation limit reached
        if self.stagnation_limit is not None:
            if self.stagnation_counter >= self.stagnation_limit:
                return True
        
        return False
    
    def get_final_rewards(self) -> List[float]:
        """
        Get final rewards.
        
        In minimal design, we use raw rewards directly, so this is just
        compatibility wrapper or empty. Gym wrapper handles accumulation.
        """
        return []
    
    def get_episode_info(self) -> Dict:
        """Get comprehensive episode information."""
        best_sol, best_cost = self.context_manager.get_global_best()
        
        return {
            'final_cost': best_cost,
            'total_fes': self.current_fes,
            'total_steps': self.episode_step,
            'switch_count': self.context_manager.get_switch_count(),
            'algorithm_stats': self.context_manager.get_algorithm_stats(),
            'reward_sum': self.cumulative_reward
        }
