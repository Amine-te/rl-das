"""
Training script for RL-DAS with Stable Baselines3 PPO.

Trains an RL agent to dynamically select algorithms for TSP optimization.

Usage:
    python train.py --timesteps 1000000 --num-cities 50 --num-instances 100
    
Algorithm Selection (4 algorithms for balanced training):
    - GA (Genetic Algorithm): Population-based exploration
    - TS (Tabu Search): Memory-based intensification
    - SA (Simulated Annealing): Probabilistic balance
    - ILS (Iterated Local Search): Perturbation + refinement
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from problems import TSPProblem
from algorithms import GeneticAlgorithm, TabuSearch, SimulatedAnnealing, IteratedLocalSearch
from core import DASGymEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train RL agent for Dynamic Algorithm Selection on TSP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=1000000,
                        help='Total training timesteps')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency (in timesteps)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    
    # Problem parameters
    parser.add_argument('--num-cities', type=int, default=50,
                        help='Number of cities in TSP instances')
    parser.add_argument('--num-instances', type=int, default=100,
                        help='Number of training problem instances')
    parser.add_argument('--instance-type', type=str, default='random',
                        choices=['random', 'clustered', 'grid', 'mixed'],
                        help='TSP instance distribution type')
    
    # RL-DAS environment parameters
    parser.add_argument('--max-fes', type=int, default=10000,
                        help='Maximum function evaluations per episode')
    parser.add_argument('--interval-fes', type=int, default=500,
                        help='FEs per decision interval')
    parser.add_argument('--population-size', type=int, default=30,
                        help='Population size for environment tracking')
    
    # PPO hyperparameters (from paper)
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='PPO learning rate')
    parser.add_argument('--n-steps', type=int, default=2048,
                        help='Steps per update')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Minibatch size')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='PPO epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=0.2,
                        help='PPO clip range')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='Entropy coefficient for exploration')
    
    # Parallelization
    parser.add_argument('--num-envs', type=int, default=4,
                        help='Number of parallel environments')
    
    # Checkpointing and logging
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Base directory for checkpoints, logs, and models (creates subdirs: checkpoints/, logs/)')
    parser.add_argument('--checkpoint-freq', type=int, default=50000,
                        help='Checkpoint save frequency (timesteps)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this training run (auto-generated if not specified)')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Resume training  
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def generate_tsp_instances(
    num_instances: int,
    num_cities: int,
    instance_type: str,
    seed: int = 42
) -> List[TSPProblem]:
    """
    Generate diverse TSP problem instances.
    
    Args:
        num_instances: Number of instances to generate
        num_cities: Number of cities per instance
        instance_type: Distribution type
        seed: Random seed
        
    Returns:
        List of TSPProblem instances
    """
    np.random.seed(seed)
    instances = []
    
    if instance_type == 'mixed':
        # Mixed distribution: 1/3 each type
        distributions = ['random', 'clustered', 'grid']
        for i in range(num_instances):
            dist = distributions[i % 3]
            problem = TSPProblem(
                num_cities=num_cities,
                distribution=dist,
                seed=seed + i
            )
            instances.append(problem)
    else:
        # Single distribution type
        for i in range(num_instances):
            problem = TSPProblem(
                num_cities=num_cities,
                distribution=instance_type,
                seed=seed + i
            )
            instances.append(problem)
    
    print(f"Generated {len(instances)} TSP instances ({instance_type}, {num_cities} cities)")
    return instances


def create_env(problem_instances: List[TSPProblem], args, is_eval: bool = False):
    """
    Create a single DAS environment.
    
    Args:
        problem_instances: List of problem instances to sample from
        args: Command line arguments
        is_eval: Whether this is an evaluation environment
        
    Returns:
        Monitored DASGymEnv instance
    """
    def _make_env():
        # Sample a random problem instance
        problem = np.random.choice(problem_instances)
        
        # Create algorithms with problem-specific tuning
        algorithms = [
            GeneticAlgorithm(
                problem,
                population_size=min(50, problem.size),
                tournament_size=3,
                crossover_rate=0.9,
                mutation_rate=0.1
            ),
            TabuSearch(
                problem,
                tabu_tenure=min(20, problem.size // 2),
                neighborhood_size=min(50, problem.size * 2),
                aspiration_enabled=True
            ),
            SimulatedAnnealing(
                problem,
                initial_temperature=100.0,
                cooling_rate=0.995,
                min_temperature=0.01
            ),
            IteratedLocalSearch(
                problem,
                perturbation_strength=max(2, problem.size // 10),
                local_search_max_iters=30,
                acceptance_criterion='better'
            )
        ]
        
        # Create environment
        env = DASGymEnv(
            problem=problem,
            algorithms=algorithms,
            max_fes=args.max_fes,
            interval_fes=args.interval_fes,
            population_size=args.population_size
        )
        
        # Wrap with Monitor for logging
        env = Monitor(env)
        
        return env
    
    return _make_env


def make_vec_env(problem_instances: List[TSPProblem], args, n_envs: int):
    """
    Create vectorized environments for parallel training.
    
    Args:
        problem_instances: List of problem instances
        args: Command line arguments
        n_envs: Number of parallel environments
        
    Returns:
        Vectorized environment
    """
    env_fns = [create_env(problem_instances, args) for _ in range(n_envs)]
    
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    else:
        # Use SubprocVecEnv for true parallelism
        return SubprocVecEnv(env_fns)


def setup_callbacks(args, eval_env):
    """
    Setup training callbacks for checkpointing and evaluation.
    
    Args:
        args: Command line arguments
        eval_env: Evaluation environment
        
    Returns:
        CallbackList with all callbacks
    """
    callbacks = []
    
    # Checkpoint callback
    checkpoints_path = os.path.join(args.checkpoint_dir, 'checkpoints')
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.num_envs,  # Adjust for parallel envs
        save_path=checkpoints_path,
        name_prefix=f'{args.run_name}_checkpoint',
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback (saves best_model.zip in checkpoint_dir)
    logs_path = os.path.join(args.checkpoint_dir, 'logs')
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.checkpoint_dir,  # best_model.zip in base dir
        log_path=logs_path,
        eval_freq=args.eval_freq // args.num_envs,  # Adjust for parallel envs
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    return CallbackList(callbacks)


def train(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Setup directory structure
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoints_path = os.path.join(args.checkpoint_dir, 'checkpoints')
    logs_path = os.path.join(args.checkpoint_dir, 'logs')
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)
    
    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_name = f'rl_das_tsp{args.num_cities}_{timestamp}'
    
    print("=" * 60)
    print(f"RL-DAS Training: {args.run_name}")
    print("=" * 60)
    print(f"Problem: TSP with {args.num_cities} cities ({args.instance_type})")
    print(f"Algorithms: GA, TS, SA, ILS (4 algorithms)")
    print(f"Max FEs: {args.max_fes}, Interval: {args.interval_fes}")
    print(f"Training timesteps: {args.timesteps:,}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Entropy coefficient: {args.ent_coef}")
    print("=" * 60)
    
    # Set random seeds
    np.random.seed(args.seed)
    
    # Generate problem instances
    train_instances = generate_tsp_instances(
        num_instances=args.num_instances,
        num_cities=args.num_cities,
        instance_type=args.instance_type,
        seed=args.seed
    )
    
    # Generate separate evaluation instances
    eval_instances = generate_tsp_instances(
        num_instances=max(10, args.num_instances // 10),
        num_cities=args.num_cities,
        instance_type=args.instance_type,
        seed=args.seed + 10000  # Different seed
    )
    
    # Create environments
    print("\nCreating environments...")
    train_env = make_vec_env(train_instances, args, args.num_envs)
    eval_env = make_vec_env(eval_instances, args, 1)  # Single eval env
    
    # Setup callbacks
    callbacks = setup_callbacks(args, eval_env)
    
    # Create or load PPO model
    if args.resume:
        print(f"\nResuming training from: {args.resume}")
        logs_path = os.path.join(args.checkpoint_dir, 'logs')
        model = PPO.load(
            args.resume,
            env=train_env,
            tensorboard_log=logs_path
        )
    else:
        print("\nInitializing PPO model...")
        logs_path = os.path.join(args.checkpoint_dir, 'logs')
        model = PPO(
            policy='MlpPolicy',
            env=train_env,
            learning_rate=args.learning_rate,
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
    
    print(f"\nModel configuration:")
    print(f"  Observation space: {train_env.observation_space.shape}")
    print(f"  Action space: {train_env.action_space.n} algorithms")
    print(f"  Policy: MlpPolicy")
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=args.run_name,
            reset_num_timesteps=False if args.resume else True,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    checkpoints_path = os.path.join(args.checkpoint_dir, 'checkpoints')
    final_model_path = os.path.join(checkpoints_path, f'{args.run_name}_final')
    model.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Run name: {args.run_name}")
    print(f"Output directory: {args.checkpoint_dir}")
    print(f"  - Checkpoints: {os.path.join(args.checkpoint_dir, 'checkpoints')}")
    print(f"  - Best model: {os.path.join(args.checkpoint_dir, 'best_model.zip')}")
    print(f"  - TensorBoard: {os.path.join(args.checkpoint_dir, 'logs')}")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir {os.path.join(args.checkpoint_dir, 'logs')}")


def main():
    """Entry point."""
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
