"""
Unified training script for RL-DAS (PPO & DQN).

Trains an RL agent to dynamically select algorithms for TSP optimization.
Supports both PPO (Continuous/Discrete) and DQN (Discrete) agents.
Includes support for synthetic instances and TSPLIB benchmarks with data augmentation.

Usage:
    # Train DQN on synthetic data
    python train.py --model-type dqn --timesteps 100000

    # Train PPO on TSPLIB
    python train.py --model-type ppo --tsplib-dir data/tsplib --num-envs 4
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from problems import TSPProblem
from utils.tsplib_loader import load_all_instances, list_available_instances
from algorithms import GeneticAlgorithm, TabuSearch, SimulatedAnnealing, IteratedLocalSearch
from core import DASGymEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train RL agent (PPO/DQN) for Dynamic Algorithm Selection on TSP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument('--model-type', type=str, default='dqn', choices=['ppo', 'dqn'],
                        help='RL algorithm to use')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this training run')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total training timesteps')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency (in timesteps)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Number of parallel environments (PPO only, DQN uses 1)')
    
    # Problem inputs
    parser.add_argument('--tsplib-dir', type=str, default=None,
                        help='Directory containing TSPLIB .tsp files. If set, uses TSPLIB instead of synthetic.')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Enable data augmentation (rotations/flips) for TSPLIB')
    parser.add_argument('--no-augment', action='store_false', dest='augment',
                        help='Disable data augmentation')
    
    # Synthetic problem generation (if tsplib-dir not set)
    parser.add_argument('--num-cities', type=int, default=None,
                        help='Number of cities (fixed size). If None, uses variable sizes')
    parser.add_argument('--min-cities', type=int, default=30,
                        help='Minimum number of cities for variable-size training')
    parser.add_argument('--max-cities', type=int, default=100,
                        help='Maximum number of cities for variable-size training')
    parser.add_argument('--num-instances', type=int, default=100,
                        help='Number of synthetic training problem instances')
    parser.add_argument('--instance-type', type=str, default='random',
                        choices=['random', 'clustered', 'grid', 'mixed'],
                        help='TSP instance distribution type')
    
    # RL-DAS environment parameters
    parser.add_argument('--max-fes', type=int, default=20000,
                        help='Maximum function evaluations per episode')
    parser.add_argument('--interval-fes', type=int, default=1000,
                        help='FEs per decision interval')
    parser.add_argument('--population-size', type=int, default=30,
                        help='Population size for environment tracking')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize observations and rewards (Recommended)')
    parser.add_argument('--no-normalize', action='store_false', dest='normalize',
                        help='Disable normalization')

    # DQN Hyperparameters
    parser.add_argument('--dqn-lr', type=float, default=1e-4, help='DQN learning rate')
    parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--learning-starts', type=int, default=1000, help='Steps before learning')
    parser.add_argument('--batch-size', type=int, default=64, help='Minibatch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient')
    parser.add_argument('--target-update-interval', type=int, default=1000, help='Target net update freq')
    parser.add_argument('--train-freq', type=int, default=4, help='Update model every N steps')
    parser.add_argument('--gradient-steps', type=int, default=1, help='Gradient steps per update')
    parser.add_argument('--exploration-fraction', type=float, default=0.3, help='Exploration decay fraction')
    parser.add_argument('--initial-eps', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--final-eps', type=float, default=0.05, help='Final exploration rate')
    
    # PPO Hyperparameters
    parser.add_argument('--ppo-lr', type=float, default=3e-4, help='PPO learning rate')
    parser.add_argument('--n-steps', type=int, default=2048, help='Steps per update')
    parser.add_argument('--n-epochs', type=int, default=10, help='PPO epochs per update')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=0.2, help='PPO clip range')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Base directory for checkpoints/logs')
    parser.add_argument('--checkpoint-freq', type=int, default=50000,
                        help='Checkpoint save frequency')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def generate_tsp_instances(
    num_instances: int,
    num_cities: int = None,
    min_cities: int = 30,
    max_cities: int = 75,
    instance_type: str = 'random',
    seed: int = 42
) -> List[TSPProblem]:
    """Generate diverse TSP problem instances."""
    np.random.seed(seed)
    instances = []
    
    if instance_type == 'mixed':
        distributions = ['random', 'clustered', 'grid']
    else:
        distributions = [instance_type]
    
    for i in range(num_instances):
        dist = distributions[i % len(distributions)]
        
        if num_cities is not None:
            size = num_cities
        else:
            size = np.random.randint(min_cities, max_cities + 1)
        
        problem = TSPProblem(
            num_cities=size,
            distribution=dist,
            seed=seed + i
        )
        instances.append(problem)
    
    if num_cities is not None:
        print(f"Generated {len(instances)} TSP instances ({instance_type}, {num_cities} cities)")
    else:
        print(f"Generated {len(instances)} TSP instances ({instance_type}, {min_cities}-{max_cities} cities)")
    
    return instances


def load_tsplib_instances(tsplib_dir: str, max_cities: int = None, augment: bool = True) -> List[TSPProblem]:
    """Load TSPLIB benchmark instances from directory."""
    files = list_available_instances(tsplib_dir)
    if not files:
        print(f"Warning: No .tsp files found in {tsplib_dir}")
        return []
    
    print(f"Loading TSPLIB instances from {tsplib_dir} (augment={augment})...")
    instances = load_all_instances(tsplib_dir, max_cities=max_cities, augment=augment)
    print(f"Loaded {len(instances)} TSPLIB benchmark instances (including variants)")
    return instances


def create_env(problem_instances: List[TSPProblem], args, is_eval: bool = False):
    """Create a single DAS environment."""
    def _make_env():
        problem = np.random.choice(problem_instances)
        
        algorithms = [
            GeneticAlgorithm(problem, population_size=min(50, problem.size), tournament_size=3),
            TabuSearch(problem, tabu_tenure=min(20, problem.size // 2), neighborhood_size=min(50, problem.size * 2), aspiration_enabled=True),
            SimulatedAnnealing(problem, initial_temperature=100.0, cooling_rate=0.995, min_temperature=0.01),
            IteratedLocalSearch(problem, perturbation_strength=max(2, problem.size // 10), local_search_max_iters=30)
        ]
        
        env = DASGymEnv(
            problem=problem,
            algorithms=algorithms,
            max_fes=args.max_fes,
            interval_fes=args.interval_fes,
            population_size=args.population_size
        )
        
        env = Monitor(env)
        return env
    
    return _make_env


def make_vec_env(problem_instances: List[TSPProblem], args, n_envs: int = 1):
    """Create vectorized environment."""
    if n_envs > 1:
        # Use SubprocVecEnv for parallel execution (PPO)
        env_fns = [create_env(problem_instances, args) for _ in range(n_envs)]
        venv = SubprocVecEnv(env_fns)
    else:
        # Use DummyVecEnv for single process (DQN or simple PPO)
        env_fn = create_env(problem_instances, args)
        venv = DummyVecEnv([env_fn])
    
    if args.normalize:
        # Normalize observations and rewards
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=100.0)
    
    return venv


def setup_callbacks(args, eval_env, total_timesteps):
    """Setup training callbacks."""
    callbacks = []
    
    # Checkpoint callback
    checkpoints_path = os.path.join(args.checkpoint_dir, 'checkpoints')
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.num_envs,
        save_path=checkpoints_path,
        name_prefix=f'{args.run_name}_checkpoint',
        save_replay_buffer=(args.model_type == 'dqn'), # Only save buffer for DQN
        save_vecnormalize=args.normalize
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    logs_path = os.path.join(args.checkpoint_dir, 'logs')
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.checkpoint_dir,
        log_path=logs_path,
        eval_freq=args.eval_freq // args.num_envs,
        n_eval_episodes=args.eval_episodes,
        deterministic=True
    )
    callbacks.append(eval_callback)
    
    return CallbackList(callbacks)


def train(args):
    """Main training function."""
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name if args.run_name else 'default_run')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoints_path = os.path.join(args.checkpoint_dir, 'checkpoints')
    logs_path = os.path.join(args.checkpoint_dir, 'logs')
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)
    
    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_name = f'{args.model_type}_tsp_{timestamp}'
    
    print("=" * 60)
    print(f"RL-DAS Training ({args.model_type.upper()}): {args.run_name}")
    print("=" * 60)
    
    # Data Loading
    np.random.seed(args.seed)
    
    if args.tsplib_dir:
        print(f"Source: TSPLIB ({args.tsplib_dir})")
        all_instances = load_tsplib_instances(args.tsplib_dir, max_cities=args.max_cities, augment=args.augment)
        if not all_instances and not args.tsplib_dir:
            print("Fallback to synthetic data") 
        
        if all_instances:
            np.random.shuffle(all_instances)
            split_idx = max(1, int(len(all_instances) * 0.8))
            train_instances = all_instances[:split_idx]
            eval_instances = all_instances[split_idx:] if split_idx < len(all_instances) else all_instances[:2]
    else:
        all_instances = None
        
    if not args.tsplib_dir or not all_instances:
        print(f"Source: Synthetic (Type={args.instance_type})")
        train_instances = generate_tsp_instances(args.num_instances, args.num_cities, args.min_cities, args.max_cities, args.instance_type, args.seed)
        eval_instances = generate_tsp_instances(max(10, args.num_instances//10), args.num_cities, args.min_cities, args.max_cities, args.instance_type, args.seed + 1000)

    print(f"Train set: {len(train_instances)} instances")
    print(f"Eval set:  {len(eval_instances)} instances")
    
    # Environment Setup
    # Override num_envs for DQN (must be 1 for basic DQN)
    if args.model_type == 'dqn':
        if args.num_envs > 1:
            print("Warning: DQN requires num_envs=1. Overriding.")
            args.num_envs = 1
            
    print("\nCreating environments...")
    train_env = make_vec_env(train_instances, args, args.num_envs)
    eval_env = make_vec_env(eval_instances, args, 1) # Eval always single env
    
    callbacks = setup_callbacks(args, eval_env, args.timesteps)
    
    # Model Init
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        if args.model_type == 'dqn':
            model = DQN.load(args.resume, env=train_env, tensorboard_log=logs_path)
        else:
            model = PPO.load(args.resume, env=train_env, tensorboard_log=logs_path)
    else:
        print(f"\nInitializing {args.model_type.upper()} model...")
        if args.model_type == 'dqn':
            model = DQN(
                policy='MlpPolicy',
                env=train_env,
                learning_rate=args.dqn_lr,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                gamma=args.gamma,
                tau=args.tau,
                target_update_interval=args.target_update_interval,
                train_freq=args.train_freq,
                gradient_steps=args.gradient_steps,
                exploration_fraction=args.exploration_fraction,
                exploration_initial_eps=args.initial_eps,
                exploration_final_eps=args.final_eps,
                verbose=1,
                tensorboard_log=logs_path,
                seed=args.seed
            )
        else: # PPO
            def linear_schedule(initial_value):
                def func(progress_remaining):
                    return progress_remaining * initial_value
                return func
                
            model = PPO(
                policy='MlpPolicy',
                env=train_env,
                learning_rate=linear_schedule(args.ppo_lr),
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                verbose=1,
                tensorboard_log=logs_path,
                seed=args.seed
            )
            
    print(f"\nModel Config:")
    print(f"  Normalization: {args.normalize}")
    print(f"  Observation:   {train_env.observation_space.shape}")
    
    print("\nStarting training...")
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name=args.run_name, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        
    final_path = os.path.join(checkpoints_path, f'{args.run_name}_final')
    model.save(final_path)
    if args.normalize:
        train_env.save(os.path.join(checkpoints_path, f'{args.run_name}_vecnormalize.pkl'))
        
    print(f"\nDone! Model saved to {final_path}")
    train_env.close()
    eval_env.close()

if __name__ == '__main__':
    args = parse_args()
    train(args)
