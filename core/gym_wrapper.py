"""
Gymnasium wrapper for DAS Environment.

Provides compatibility with standard RL libraries like Stable Baselines3.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYMNASIUM = False
    except ImportError:
        raise ImportError("Neither gymnasium nor gym is installed. Please install one.")

from .environment import DASEnvironment


class DASGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for DASEnvironment.
    
    This wrapper allows using standard RL training libraries
    (Stable Baselines3, CleanRL, etc.) with the DAS environment.
    
    Observation space: Box of normalized features [0, 1]
    Action space: Discrete selection of algorithms
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        das_env: Optional[DASEnvironment] = None,
        problem: Optional[Any] = None,
        algorithms: Optional[list] = None,
        max_fes: int = 100000,
        interval_fes: int = 2000,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Gymnasium wrapper.
        
        Can be initialized with either:
        - A pre-configured DASEnvironment
        - Problem and algorithms to create a new DASEnvironment
        
        Args:
            das_env: Pre-configured DAS environment (optional)
            problem: Problem instance (required if das_env not provided)
            algorithms: List of algorithms (required if das_env not provided)
            max_fes: Max function evaluations
            interval_fes: FEs per interval
            render_mode: Rendering mode
            **kwargs: Additional arguments for DASEnvironment
        """
        super().__init__()
        
        if das_env is not None:
            self.das_env = das_env
        elif problem is not None and algorithms is not None:
            self.das_env = DASEnvironment(
                problem=problem,
                algorithms=algorithms,
                max_fes=max_fes,
                interval_fes=interval_fes,
                **kwargs
            )
        else:
            raise ValueError(
                "Must provide either das_env or both problem and algorithms"
            )
        
        self.render_mode = render_mode
        
        # Define observation space
        state_dim = self.das_env.state_dim
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Define action space
        self.action_space = spaces.Discrete(self.das_env.num_algorithms)
        
        # Episode tracking for rendering
        self._episode_rewards = []
        self._episode_actions = []
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed (for reproducibility)
            options: Additional options (can include 'problem_instance')
            
        Returns:
            Tuple of (observation, info)
        """
        # Handle seeding
        if seed is not None:
            np.random.seed(seed)
        
        # Get problem instance from options if provided
        problem_instance = None
        if options is not None and 'problem_instance' in options:
            problem_instance = options['problem_instance']
        
        # Reset the DAS environment
        state = self.das_env.reset(problem_instance)
        
        # Reset episode tracking
        self._episode_rewards = []
        self._episode_actions = []
        
        # Build info
        info = {
            'initial_cost': self.das_env.last_best_cost,
            'max_fes': self.das_env.max_fes
        }
        
        return state.astype(np.float32), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Algorithm index to select
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        state, reward, terminated, truncated, info = self.das_env.step(action)
        
        # Track for rendering
        self._episode_rewards.append(reward)
        self._episode_actions.append(action)
        
        return state.astype(np.float32), reward, terminated, truncated, info
    
    def render(self) -> Optional[str]:
        """
        Render the environment.
        
        Returns:
            String representation if render_mode is 'ansi'
        """
        if self.render_mode == 'human':
            self._render_human()
        elif self.render_mode == 'ansi':
            return self._render_ansi()
        return None
    
    def _render_human(self):
        """Print episode status to console."""
        info = self.das_env.get_episode_info()
        print(f"\n=== Episode Step {info['total_steps']} ===")
        print(f"Best Cost: {info['final_cost']:.4f}")
        print(f"FEs Used: {info['total_fes']} / {self.das_env.max_fes}")
        print(f"Algorithm Switches: {info['switch_count']}")
        print(f"Recent Actions: {self._episode_actions[-5:]}")
        print(f"Recent Rewards: {[f'{r:.4f}' for r in self._episode_rewards[-5:]]}")
    
    def _render_ansi(self) -> str:
        """Return string representation of episode status."""
        info = self.das_env.get_episode_info()
        lines = [
            f"Step: {info['total_steps']}",
            f"Cost: {info['final_cost']:.4f}",
            f"FEs: {info['total_fes']}/{self.das_env.max_fes}",
            f"Switches: {info['switch_count']}"
        ]
        return '\n'.join(lines)
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_episode_summary(self) -> Dict:
        """
        Get comprehensive episode summary.
        
        Returns:
            Dictionary with episode statistics
        """
        info = self.das_env.get_episode_info()
        info['episode_rewards'] = self._episode_rewards
        info['episode_actions'] = self._episode_actions
        info['final_rewards'] = self.das_env.get_final_rewards()
        return info


def make_das_env(
    problem: Any,
    algorithms: list,
    max_fes: int = 100000,
    interval_fes: int = 2000,
    **kwargs
) -> DASGymEnv:
    """
    Factory function to create a DAS Gym environment.
    
    Args:
        problem: Problem instance
        algorithms: List of algorithm instances
        max_fes: Maximum function evaluations
        interval_fes: FEs per decision interval
        **kwargs: Additional arguments
        
    Returns:
        Configured DASGymEnv instance
    """
    return DASGymEnv(
        problem=problem,
        algorithms=algorithms,
        max_fes=max_fes,
        interval_fes=interval_fes,
        **kwargs
    )
