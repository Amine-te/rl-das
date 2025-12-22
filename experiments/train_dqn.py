"""
Training script for RL-DAS with Stable Baselines3 DQN.

Trains a DQN agent to dynamically select algorithms for TSP optimization.
DQN is often better for discrete action spaces like algorithm selection.

Usage:
    python train_dqn.py --timesteps 500000 --num-instances 200 --checkpoint-dir checkpoints/dqn_run1
    
Algorithm Selection (4 algorithms):
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
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from problems import TSPProblem
from algorithms import GeneticAlgorithm, TabuSearch, SimulatedAnnealing, IteratedLocalSearch
from core import DASGymEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train DQN agent for Dynamic Algorithm Selection on TSP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total training timesteps')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency (in timesteps)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    
    # Problem parameters
    parser.add_argument('--num-cities', type=int, default=None,
                        help='Number of cities (fixed size). If None, uses variable sizes')
    parser.add_argument('--min-cities', type=int, default=30,
                        help='Minimum number of cities for variable-size training')
    parser.add_argument('--max-cities', type=int, default=75,
                        help='Maximum number of cities for variable-size training')
    parser.add_argument('--num-instances', type=int, default=100,
                        help='Number of training problem instances')
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
    
    # DQN hyperparameters
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='DQN learning rate')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Replay buffer size')
    parser.add_argument('--learning-starts', type=int, default=1000,
                        help='Number of steps before learning starts')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Minibatch size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient for target network')
    parser.add_argument('--target-update-interval', type=int, default=1000,
                        help='Update target network every N steps')
    parser.add_argument('--train-freq', type=int, default=4,
                        help='Update the model every N steps')
    parser.add_argument('--gradient-steps', type=int, default=1,
                        help='Gradient steps per update')
    
    # Exploration parameters
    parser.add_argument('--exploration-fraction', type=float, default=0.3,
                        help='Fraction of training for epsilon decay')
    parser.add_argument('--exploration-initial-eps', type=float, default=1.0,
                        help='Initial exploration rate')
    parser.add_argument('--exploration-final-eps', type=float, default=0.05,
                        help='Final exploration rate')
    
    # Checkpointing and logging
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Base directory for checkpoints, logs, and models')
    parser.add_argument('--checkpoint-freq', type=int, default=50000,
                        help='Checkpoint save frequency (timesteps)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this training run')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Resume training  
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


def create_env(problem_instances: List[TSPProblem], args, is_eval: bool = False):
    """Create a single DAS environment."""
    def _make_env():
        problem = np.random.choice(problem_instances)
        
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


def make_vec_env(problem_instances: List[TSPProblem], args):
    """Create vectorized environment for training."""
    # DQN only supports single environment
    env_fn = create_env(problem_instances, args)
    return DummyVecEnv([env_fn])


def setup_callbacks(args, eval_env):
    """Setup training callbacks for checkpointing and evaluation."""
    callbacks = []
    
    checkpoints_path = os.path.join(args.checkpoint_dir, 'checkpoints')
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=checkpoints_path,
        name_prefix=f'{args.run_name}_checkpoint',
        save_replay_buffer=True,
        save_vecnormalize=False
    )
    callbacks.append(checkpoint_callback)
    
    logs_path = os.path.join(args.checkpoint_dir, 'logs')
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.checkpoint_dir,
        log_path=logs_path,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    return CallbackList(callbacks)


def train(args):
    """Main training function."""
    # Setup directory structure
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoints_path = os.path.join(args.checkpoint_dir, 'checkpoints')
    logs_path = os.path.join(args.checkpoint_dir, 'logs')
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)
    
    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_name = f'dqn_das_tsp{args.num_cities}_{timestamp}'
    
    print("=" * 60)
    print(f"RL-DAS DQN Training: {args.run_name}")
    print("=" * 60)
    if args.num_cities is not None:
        print(f"Problem: TSP with {args.num_cities} cities ({args.instance_type})")
    else:
        print(f"Problem: TSP with {args.min_cities}-{args.max_cities} cities (variable, {args.instance_type})")
    print(f"Algorithms: GA, TS, SA, ILS (4 algorithms)")
    print(f"Max FEs: {args.max_fes}, Interval: {args.interval_fes}")
    print(f"Training timesteps: {args.timesteps:,}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Exploration: {args.exploration_initial_eps} -> {args.exploration_final_eps} over {args.exploration_fraction*100:.0f}%")
    print("=" * 60)
    
    # Set random seeds
    np.random.seed(args.seed)
    
    # Generate problem instances
    train_instances = generate_tsp_instances(
        num_instances=args.num_instances,
        num_cities=args.num_cities,
        min_cities=args.min_cities,
        max_cities=args.max_cities,
        instance_type=args.instance_type,
        seed=args.seed
    )
    
    eval_instances = generate_tsp_instances(
        num_instances=max(10, args.num_instances // 10),
        num_cities=args.num_cities,
        min_cities=args.min_cities,
        max_cities=args.max_cities,
        instance_type=args.instance_type,
        seed=args.seed + 10000
    )
    
    # Create environments
    print("\nCreating environments...")
    train_env = make_vec_env(train_instances, args)
    eval_env = make_vec_env(eval_instances, args)
    
    # Setup callbacks
    callbacks = setup_callbacks(args, eval_env)
    
    # Create or load DQN model
    if args.resume:
        print(f"\nResuming training from: {args.resume}")
        model = DQN.load(
            args.resume,
            env=train_env,
            tensorboard_log=logs_path
        )
    else:
        print("\nInitializing DQN model...")
        model = DQN(
            policy='MlpPolicy',
            env=train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            target_update_interval=args.target_update_interval,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
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
    print(f"  - Checkpoints: {checkpoints_path}")
    print(f"  - Best model: {os.path.join(args.checkpoint_dir, 'best_model.zip')}")
    print(f"  - TensorBoard: {logs_path}")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir {logs_path}")


def main():
    """Entry point."""
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
