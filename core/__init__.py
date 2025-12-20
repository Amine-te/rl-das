"""
Core module for RL-DAS.

Contains the main components:
- StateExtractor: Extract problem-agnostic features
- RewardCalculator: Compute RL rewards with speed factor
- ContextManager: Manage algorithm warm-start context
- DASEnvironment: Main environment orchestrator
- DASGymEnv: Gymnasium-compatible wrapper
"""

from .state_extractor import StateExtractor
from .reward_calculator import RewardCalculator
from .context_manager import ContextManager
from .environment import DASEnvironment
from .gym_wrapper import DASGymEnv, make_das_env

__all__ = [
    'StateExtractor',
    'RewardCalculator',
    'ContextManager',
    'DASEnvironment',
    'DASGymEnv',
    'make_das_env'
]
