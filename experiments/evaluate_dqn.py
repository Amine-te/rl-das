"""
Evaluation script for RL-DAS trained DQN models.

Evaluates a trained DQN agent against baseline algorithms on TSP instances.
Generates detailed logs showing:
- Algorithm selections at each interval
- Q-values for each action
- Performance comparisons vs single-algorithm baselines

Usage:
    python evaluate_dqn.py --model checkpoints/dqn_run1/best_model.zip --num-test-instances 20
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import time

import numpy as np
import torch
from stable_baselines3 import DQN

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from problems import TSPProblem
from algorithms import GeneticAlgorithm, TabuSearch, SimulatedAnnealing, IteratedLocalSearch
from core import DASGymEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained DQN model on TSP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and results
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this evaluation run')
    
    # Test problem parameters
    parser.add_argument('--num-test-instances', type=int, default=20,
                        help='Number of test instances to evaluate')
    parser.add_argument('--num-cities', type=int, default=50,
                        help='Number of cities in TSP instances')
    parser.add_argument('--instance-type', type=str, default='mixed',
                        choices=['random', 'clustered', 'grid', 'mixed'],
                        help='TSP instance distribution type')
    parser.add_argument('--seed', type=int, default=999,
                        help='Random seed for test instances')
    
    # Evaluation parameters
    parser.add_argument('--max-fes', type=int, default=30000,
                        help='Maximum function evaluations per episode')
    parser.add_argument('--interval-fes', type=int, default=1000,
                        help='FEs per decision interval')
    parser.add_argument('--population-size', type=int, default=30,
                        help='Population size for environment tracking')
    
    # Baseline comparisons
    parser.add_argument('--run-baselines', action='store_true',
                        help='Run single-algorithm baselines for comparison')
    parser.add_argument('--baseline-fes', type=int, default=None,
                        help='FEs for baseline (defaults to --max-fes)')
    
    # Evaluation settings
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic policy (greedy action selection)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress during evaluation')
    
    return parser.parse_args()


def generate_test_instances(
    num_instances: int,
    num_cities: int,
    instance_type: str,
    seed: int
) -> List[TSPProblem]:
    """Generate test problem instances."""
    np.random.seed(seed)
    instances = []
    
    if instance_type == 'mixed':
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
        for i in range(num_instances):
            problem = TSPProblem(
                num_cities=num_cities,
                distribution=instance_type,
                seed=seed + i
            )
            instances.append(problem)
    
    return instances


def create_algorithms(problem: TSPProblem) -> List:
    """Create algorithm instances for a problem."""
    return [
        GeneticAlgorithm(
            problem,
            population_size=min(50, problem.size),
            tournament_size=3
        ),
        TabuSearch(
            problem,
            tabu_tenure=min(20, problem.size // 2),
            neighborhood_size=min(50, problem.size * 2)
        ),
        SimulatedAnnealing(
            problem,
            initial_temperature=100.0,
            cooling_rate=0.995
        ),
        IteratedLocalSearch(
            problem,
            perturbation_strength=max(2, problem.size // 10),
            local_search_max_iters=30
        )
    ]


def evaluate_dqn_agent(
    model: DQN,
    problem: TSPProblem,
    args,
    instance_id: int
) -> Dict:
    """
    Evaluate DQN agent on a single instance.
    
    Returns detailed episode information including action sequence and Q-values.
    """
    # Create environment
    algorithms = create_algorithms(problem)
    env = DASGymEnv(
        problem=problem,
        algorithms=algorithms,
        max_fes=args.max_fes,
        interval_fes=args.interval_fes,
        population_size=args.population_size
    )
    
    # Run episode
    obs, info = env.reset()
    done = False
    
    episode_data = {
        'instance_id': instance_id,
        'initial_cost': info['initial_cost'],
        'steps': [],
        'actions': [],
        'q_values': [],
        'rewards': [],
        'costs': []
    }
    
    step = 0
    while not done:
        # Get action and Q-values
        if args.deterministic:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=False)
        
        # Get Q-values from the model
        obs_tensor = torch.as_tensor(obs.reshape(1, -1), device=model.device)
        with torch.no_grad():
            q_values = model.q_net(obs_tensor).cpu().numpy()[0]
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        
        # Record step data
        episode_data['steps'].append(step)
        episode_data['actions'].append(int(action))
        episode_data['q_values'].append(q_values.tolist())
        episode_data['rewards'].append(float(reward))
        episode_data['costs'].append(float(info['best_cost']))
        
        step += 1
    
    # Final statistics
    episode_summary = env.get_episode_summary()
    episode_data['final_cost'] = episode_summary['final_cost']
    episode_data['total_steps'] = episode_summary['total_steps']
    episode_data['total_fes'] = episode_summary['total_fes']
    episode_data['switch_count'] = episode_summary['switch_count']
    
    env.close()
    
    return episode_data


def run_baseline(
    algorithm_class,
    algorithm_name: str,
    problem: TSPProblem,
    max_fes: int,
    instance_id: int
) -> Dict:
    """Run a single algorithm as baseline."""
    algo = algorithm_class(problem)
    algo.initialize()
    
    remaining_fes = max_fes - algo.evaluations_used
    if remaining_fes > 0:
        algo.step(remaining_fes)
    
    return {
        'instance_id': instance_id,
        'algorithm': algorithm_name,
        'final_cost': algo.best_cost,
        'evaluations_used': algo.evaluations_used
    }


def format_episode_report(episode_data: Dict, algo_names: List[str]) -> str:
    """Format detailed episode report with Q-values."""
    lines = []
    lines.append("=" * 100)
    lines.append(f"INSTANCE {episode_data['instance_id']}")
    lines.append("=" * 100)
    lines.append(f"Initial Cost: {episode_data['initial_cost']:.4f}")
    lines.append(f"Final Cost:   {episode_data['final_cost']:.4f}")
    lines.append(f"Improvement:  {episode_data['initial_cost'] - episode_data['final_cost']:.4f} "
                 f"({100 * (episode_data['initial_cost'] - episode_data['final_cost']) / episode_data['initial_cost']:.2f}%)")
    lines.append(f"Total Steps:  {episode_data['total_steps']}")
    lines.append(f"Total FEs:    {episode_data['total_fes']}")
    lines.append(f"Switches:     {episode_data['switch_count']}")
    lines.append("")
    lines.append("-" * 100)
    lines.append("STEP-BY-STEP ALGORITHM SELECTION (DQN Q-VALUES)")
    lines.append("-" * 100)
    lines.append(f"{'Step':<6} {'Action':<8} {'Algorithm':<12} {'Cost':<12} {'Reward':<10} {'Q-Values'}")
    lines.append("-" * 100)
    
    for i in range(len(episode_data['steps'])):
        step = episode_data['steps'][i]
        action = episode_data['actions'][i]
        algo_name = algo_names[action]
        cost = episode_data['costs'][i]
        reward = episode_data['rewards'][i]
        q_vals = episode_data['q_values'][i]
        
        # Format Q-values
        q_str = " ".join([f"{algo_names[j]}:{q_vals[j]:.3f}" for j in range(len(q_vals))])
        
        lines.append(f"{step:<6} {action:<8} {algo_name:<12} {cost:<12.4f} {reward:<10.4f} {q_str}")
    
    lines.append("=" * 100)
    lines.append("")
    
    return "\n".join(lines)


def evaluate(args):
    """Main evaluation function."""
    # Setup
    os.makedirs(args.results_dir, exist_ok=True)
    
    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_name = f'dqn_eval_tsp{args.num_cities}_{timestamp}'
    
    if args.baseline_fes is None:
        args.baseline_fes = args.max_fes
    
    print("=" * 80)
    print(f"RL-DAS DQN Evaluation: {args.run_name}")
    print("=" * 80)
    print(f"Model:          {args.model}")
    print(f"Problem:        TSP with {args.num_cities} cities ({args.instance_type})")
    print(f"Test instances: {args.num_test_instances}")
    print(f"Max FEs:        {args.max_fes}")
    print(f"Interval FEs:   {args.interval_fes}")
    print(f"Deterministic:  {args.deterministic}")
    print(f"Run baselines:  {args.run_baselines}")
    print("=" * 80)
    
    # Load model
    print("\nLoading trained DQN model...")
    model = DQN.load(args.model)
    print(f"✓ Model loaded from {args.model}")
    
    # Generate test instances
    print(f"\nGenerating {args.num_test_instances} test instances...")
    test_instances = generate_test_instances(
        args.num_test_instances,
        args.num_cities,
        args.instance_type,
        args.seed
    )
    print(f"✓ Generated {len(test_instances)} test instances")
    
    # Algorithm names
    algo_names = ['GA', 'TS', 'SA', 'ILS']
    
    # Evaluate DQN agent
    print("\n" + "=" * 80)
    print("EVALUATING DQN AGENT")
    print("=" * 80)
    
    dqn_results = []
    detailed_logs = []
    
    start_time = time.time()
    
    for i, problem in enumerate(test_instances):
        if args.verbose:
            print(f"\nInstance {i+1}/{len(test_instances)}...")
        
        episode_data = evaluate_dqn_agent(model, problem, args, i)
        dqn_results.append(episode_data)
        
        # Format detailed report
        report = format_episode_report(episode_data, algo_names)
        detailed_logs.append(report)
        
        if args.verbose:
            print(f"  Final cost: {episode_data['final_cost']:.4f}")
    
    dqn_time = time.time() - start_time
    
    # Compute DQN statistics
    dqn_costs = [r['final_cost'] for r in dqn_results]
    dqn_mean = np.mean(dqn_costs)
    dqn_std = np.std(dqn_costs)
    dqn_best = np.min(dqn_costs)
    dqn_worst = np.max(dqn_costs)
    
    print(f"\n✓ DQN Agent evaluation complete ({dqn_time:.1f}s)")
    print(f"  Mean cost: {dqn_mean:.4f} ± {dqn_std:.4f}")
    print(f"  Best:      {dqn_best:.4f}")
    print(f"  Worst:     {dqn_worst:.4f}")
    
    # Baseline evaluation
    baseline_results = {}
    
    if args.run_baselines:
        print("\n" + "=" * 80)
        print("EVALUATING BASELINES")
        print("=" * 80)
        
        baseline_algos = [
            (GeneticAlgorithm, 'GA'),
            (TabuSearch, 'TS'),
            (SimulatedAnnealing, 'SA'),
            (IteratedLocalSearch, 'ILS')
        ]
        
        for algo_class, algo_name in baseline_algos:
            print(f"\n{algo_name}...")
            start_time = time.time()
            
            results = []
            for i, problem in enumerate(test_instances):
                if args.verbose:
                    print(f"  Instance {i+1}/{len(test_instances)}...", end='')
                
                result = run_baseline(algo_class, algo_name, problem, args.baseline_fes, i)
                results.append(result)
                
                if args.verbose:
                    print(f" {result['final_cost']:.4f}")
            
            baseline_time = time.time() - start_time
            
            costs = [r['final_cost'] for r in results]
            baseline_results[algo_name] = {
                'costs': costs,
                'mean': np.mean(costs),
                'std': np.std(costs),
                'best': np.min(costs),
                'worst': np.max(costs),
                'time': baseline_time
            }
            
            print(f"  ✓ Complete ({baseline_time:.1f}s)")
            print(f"    Mean: {baseline_results[algo_name]['mean']:.4f} ± {baseline_results[algo_name]['std']:.4f}")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    result_file = os.path.join(args.results_dir, f'{args.run_name}.txt')
    
    with open(result_file, 'w') as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write(f"RL-DAS DQN EVALUATION REPORT\n")
        f.write("=" * 100 + "\n")
        f.write(f"Run Name:       {args.run_name}\n")
        f.write(f"Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model:          {args.model}\n")
        f.write(f"Problem:        TSP {args.num_cities} cities ({args.instance_type})\n")
        f.write(f"Test Instances: {args.num_test_instances}\n")
        f.write(f"Max FEs:        {args.max_fes}\n")
        f.write(f"Interval FEs:   {args.interval_fes}\n")
        f.write(f"Deterministic:  {args.deterministic}\n")
        f.write("\n\n")
        
        # DQN Agent Summary
        f.write("=" * 100 + "\n")
        f.write("DQN AGENT PERFORMANCE SUMMARY\n")
        f.write("=" * 100 + "\n")
        f.write(f"Mean Cost:   {dqn_mean:.6f} ± {dqn_std:.6f}\n")
        f.write(f"Best Cost:   {dqn_best:.6f}\n")
        f.write(f"Worst Cost:  {dqn_worst:.6f}\n")
        f.write(f"Total Time:  {dqn_time:.2f}s\n")
        f.write("\n\n")
        
        # Baseline comparison
        if args.run_baselines:
            f.write("=" * 100 + "\n")
            f.write("BASELINE COMPARISON\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'Algorithm':<12} {'Mean Cost':<15} {'Std':<12} {'Best':<12} {'Worst':<12} {'Time (s)'}\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'DQN-DAS':<12} {dqn_mean:<15.6f} {dqn_std:<12.6f} {dqn_best:<12.6f} {dqn_worst:<12.6f} {dqn_time:<10.2f}\n")
            
            for algo_name in ['GA', 'TS', 'SA', 'ILS']:
                stats = baseline_results[algo_name]
                f.write(f"{algo_name:<12} {stats['mean']:<15.6f} {stats['std']:<12.6f} "
                       f"{stats['best']:<12.6f} {stats['worst']:<12.6f} {stats['time']:<10.2f}\n")
            
            f.write("\n")
            f.write("Improvement over baselines:\n")
            for algo_name in ['GA', 'TS', 'SA', 'ILS']:
                improvement = ((baseline_results[algo_name]['mean'] - dqn_mean) / baseline_results[algo_name]['mean']) * 100
                f.write(f"  vs {algo_name}: {improvement:+.2f}%\n")
            
            f.write("\n\n")
        
        # Detailed episode logs
        f.write("=" * 100 + "\n")
        f.write("DETAILED EPISODE LOGS (WITH Q-VALUES)\n")
        f.write("=" * 100 + "\n\n")
        
        for log in detailed_logs:
            f.write(log)
            f.write("\n")
    
    print(f"✓ Results saved to: {result_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nDQN-DAS Mean Cost: {dqn_mean:.6f} ± {dqn_std:.6f}")
    
    if args.run_baselines:
        print("\nComparison vs baselines:")
        for algo_name in ['GA', 'TS', 'SA', 'ILS']:
            improvement = ((baseline_results[algo_name]['mean'] - dqn_mean) / baseline_results[algo_name]['mean']) * 100
            print(f"  vs {algo_name}: {improvement:+.2f}%")
    
    print(f"\nDetailed results: {result_file}")
    print("=" * 80)


def main():
    """Entry point."""
    args = parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
