"""
Unified evaluation script for RL-DAS (PPO & DQN).

Evaluates a trained RL agent against baseline algorithms on TSP instances.
Supports both PPO and DQN models.
Automatically detects and loads VecNormalize statistics if available.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from problems import TSPProblem
from utils.tsp_loader import load_all_instances
from algorithms import GeneticAlgorithm, TabuSearch, SimulatedAnnealing, IteratedLocalSearch
from core import DASGymEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained RL-DAS model (PPO/DQN) on TSP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and results
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint (.zip)')
    parser.add_argument('--model-type', type=str, default='auto', choices=['auto', 'ppo', 'dqn'],
                        help='Model type (auto tries to infer from content)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this evaluation run')
    
    # Test problem parameters
    parser.add_argument('--tsplib-dir', type=str, default=None,
                        help='Directory containing TSPLIB .tsp files (optional)')
    
    parser.add_argument('--num-test-instances', type=int, default=20,
                        help='Number of test instances (if synthetic)')
    parser.add_argument('--num-cities', type=int, default=50,
                        help='Number of cities (if synthetic)')
    parser.add_argument('--instance-type', type=str, default='mixed',
                        help='Distribution type (if synthetic)')
    parser.add_argument('--seed', type=int, default=999,
                        help='Random seed for test instances')
    
    # Evaluation settings
    parser.add_argument('--max-fes', type=int, default=30000,
                        help='Maximum function evaluations per episode')
    parser.add_argument('--interval-fes', type=int, default=1000,
                        help='FEs per decision interval')
    parser.add_argument('--population-size', type=int, default=30,
                        help='Population size')
    
    parser.add_argument('--run-baselines', action='store_true',
                        help='Run single-algorithm baselines for comparison')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic policy')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    
    return parser.parse_args()


def load_model(model_path: str, model_type: str):
    """Load model and optionally VecNormalize stats."""
    # Infer type if auto
    if model_type == 'auto':
        if 'dqn' in model_path.lower():
            model_type = 'dqn'
        elif 'ppo' in model_path.lower():
            model_type = 'ppo'
        else:
            # Try loading as DQN first (arbitrary choice)
            try:
                model = DQN.load(model_path)
                print("Inferred model type: DQN")
                return model, 'dqn'
            except:
                print("Inferred model type: PPO")
                return PPO.load(model_path), 'ppo'
    
    if model_type == 'dqn':
        return DQN.load(model_path), 'dqn'
    else:
        return PPO.load(model_path), 'ppo'


def create_algorithms(problem: TSPProblem) -> List:
    """Create algorithm instances for a problem."""
    return [
        GeneticAlgorithm(problem, population_size=min(50, problem.size), tournament_size=3),
        TabuSearch(problem, tabu_tenure=min(20, problem.size // 2), neighborhood_size=min(50, problem.size * 2), aspiration_enabled=True),
        SimulatedAnnealing(problem, initial_temperature=100.0, cooling_rate=0.995, min_temperature=0.01),
        IteratedLocalSearch(problem, perturbation_strength=max(2, problem.size // 10), local_search_max_iters=30)
    ]


def evaluate_rl_agent(model, model_type, problem, args, instance_id, vec_norm=None) -> Dict:
    """Evaluate RL agent on a single instance."""
    start_time = time.time()
    
    # Create env
    def make_env():
        algorithms = create_algorithms(problem)
        return DASGymEnv(
            problem=problem, 
            algorithms=algorithms, 
            max_fes=args.max_fes, 
            interval_fes=args.interval_fes, 
            population_size=args.population_size
        )
        
    env = DummyVecEnv([make_env])
    
    # Apply normalization if stats provided
    if vec_norm is not None:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        # Load stats
        env.obs_rms = vec_norm.obs_rms
        env.ret_rms = vec_norm.ret_rms
        env.training = False # Don't update stats during eval
        
    obs = env.reset()
    done = False
    
    # Get initial info (from unwrapped env)
    unwrapped = env.envs[0]
    # Handle access to inner DASEnvironment
    if hasattr(unwrapped, 'das_env'):
         inner_env = unwrapped.das_env
    else:
         # Fallback if unwrapped is actually DASEnvironment (unlikely with Gym wrapper)
         inner_env = unwrapped
         
    initial_cost = inner_env.last_best_cost
    
    episode_data = {
        'instance_id': instance_id,
        'initial_cost': initial_cost,
        'steps': [],
        'actions': [],
        'step_details': [],
        'rewards': [],
        'costs': [],
        'problem_name': getattr(problem, 'name', f"Instance_{instance_id}")
    }
    
    step_count = 0
    while not done:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        
        # Introspection (Q-values or Probs)
        detail_str = ""
        if model_type == 'dqn':
            # Get Q-values
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs).to(model.device)
                q_values = model.q_net(obs_tensor)
                q_vals_np = q_values.cpu().numpy()[0]
                detail_str = " ".join([f"A{i}:{q:.2f}" for i, q in enumerate(q_vals_np)])
        else:
            # Get probabilities
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs).to(model.device)
                dist = model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0]
                detail_str = probs  # Store raw probabilities for formatting
        
        obs, reward, done, info = env.step(action)
        
        # Record
        episode_data['steps'].append(step_count)
        episode_data['actions'].append(int(action))
        episode_data['step_details'].append(detail_str)
        episode_data['rewards'].append(float(reward))
        episode_data['costs'].append(float(info[0]['best_cost'])) # VecEnv returns list of infos
        
        step_count += 1
        
    # Final stats
    episode_data['final_cost'] = float(info[0]['best_cost'])
    episode_data['total_steps'] = step_count
    episode_data['total_fes'] = inner_env.current_fes
    episode_data['switch_count'] = inner_env.context_manager.get_switch_count()
    episode_data['time'] = time.time() - start_time
    
    return episode_data


def run_baselines_comparison(problem, args) -> Dict[str, Dict]:
    """Run all baselines on the problem and return detailed results."""
    results = {}
    algorithms = create_algorithms(problem)
    algo_names = ['GA', 'TS', 'SA', 'ILS']
    
    for name, algo in zip(algo_names, algorithms):
        start_time = time.time()
        
        # Reset/Initialize
        algo.initialize()
        
        # Run for max_fes
        current_fes = 0
        best_cost = float('inf')
        
        # Initial solution
        sol = problem.generate_random_solution()
        cost = problem.evaluate(sol)
        algo.inject_solution(sol, cost)
        
        while current_fes < args.max_fes:
            fes_allowed = min(args.interval_fes, args.max_fes - current_fes)
            _, cost = algo.step(fes_allowed)
            current_fes += fes_allowed
            if cost < best_cost:
                best_cost = cost
                
        results[name] = {
            'cost': best_cost,
            'time': time.time() - start_time
        }
        
    return results


def format_report(args, model_type, all_results, all_baselines, total_time, timestamp):
    """Format the complete evaluation report."""
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("RL-DAS EVALUATION REPORT")
    lines.append("=" * 80)
    
    run_name = args.run_name or f"eval_{model_type}_{timestamp}"
    lines.append(f"Run Name:       {run_name}")
    lines.append(f"Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model:          {args.model}")
    lines.append(f"Problem:        TSP {args.num_cities} cities ({args.instance_type})")
    lines.append(f"Test Instances: {len(all_results)}")
    lines.append(f"Max FEs:        {args.max_fes}")
    lines.append(f"Interval FEs:   {args.interval_fes}")
    lines.append(f"Deterministic:  {args.deterministic}")
    lines.append("")
    lines.append("")
    
    # RL Agent Performance Summary
    lines.append("=" * 80)
    lines.append("RL AGENT PERFORMANCE SUMMARY")
    lines.append("=" * 80)
    
    costs = [r['final_cost'] for r in all_results]
    times = [r['time'] for r in all_results]
    
    lines.append(f"Mean Cost:   {np.mean(costs):.6f} ± {np.std(costs):.6f}")
    lines.append(f"Best Cost:   {np.min(costs):.6f}")
    lines.append(f"Worst Cost:  {np.max(costs):.6f}")
    lines.append(f"Total Time:  {total_time:.2f}s")
    lines.append("")
    lines.append("")
    
    # Baseline Comparison (if available)
    if args.run_baselines and all_baselines:
        lines.append("=" * 80)
        lines.append("BASELINE COMPARISON")
        lines.append("=" * 80)
        
        # Aggregate baseline results - each instance separately
        baseline_stats = {}
        for algo_name in ['GA', 'TS', 'SA', 'ILS']:
            algo_costs = [instance_baselines[algo_name]['cost'] for instance_baselines in all_baselines]
            algo_times = [instance_baselines[algo_name]['time'] for instance_baselines in all_baselines]
            baseline_stats[algo_name] = {
                'mean': np.mean(algo_costs),
                'std': np.std(algo_costs),
                'best': np.min(algo_costs),
                'worst': np.max(algo_costs),
                'time': np.sum(algo_times)
            }
        
        # Table header
        lines.append(f"{'Algorithm':<12} {'Mean Cost':<15} {'Std':<12} {'Best':<12} {'Worst':<12} {'Time (s)':<10}")
        lines.append("-" * 80)
        
        # RL-DAS row
        lines.append(f"{'RL-DAS':<12} {np.mean(costs):<15.6f} {np.std(costs):<12.6f} "
                    f"{np.min(costs):<12.6f} {np.max(costs):<12.6f} {total_time:<10.2f}")
        
        # Baseline rows
        for algo in ['GA', 'TS', 'SA', 'ILS']:
            stats = baseline_stats[algo]
            lines.append(f"{algo:<12} {stats['mean']:<15.6f} {stats['std']:<12.6f} "
                        f"{stats['best']:<12.6f} {stats['worst']:<12.6f} {stats['time']:<10.2f}")
        
        lines.append("")
        lines.append("Improvement over baselines:")
        for algo in ['GA', 'TS', 'SA', 'ILS']:
            improvement = ((np.mean(costs) - baseline_stats[algo]['mean']) / baseline_stats[algo]['mean']) * 100
            lines.append(f"  vs {algo}: {improvement:+.2f}%")
        
        lines.append("")
        lines.append("")
    
    # Detailed Episode Logs
    lines.append("=" * 80)
    lines.append("DETAILED EPISODE LOGS")
    lines.append("=" * 80)
    lines.append("")
    
    algo_names = ['GA', 'TS', 'SA', 'ILS']
    
    for idx, episode_data in enumerate(all_results):
        lines.append("=" * 80)
        lines.append(f"INSTANCE {idx}")
        lines.append("=" * 80)
        
        initial_cost = episode_data['initial_cost']
        final_cost = episode_data['final_cost']
        improvement = initial_cost - final_cost
        improvement_pct = (improvement / initial_cost) * 100 if initial_cost > 0 else 0
        
        lines.append(f"Initial Cost: {initial_cost:.4f}")
        lines.append(f"Final Cost:   {final_cost:.4f}")
        lines.append(f"Improvement:  {improvement:.4f} ({improvement_pct:.2f}%)")
        lines.append(f"Total Steps:  {episode_data['total_steps']}")
        lines.append(f"Total FEs:    {episode_data['total_fes']}")
        lines.append(f"Switches:     {episode_data['switch_count']}")
        lines.append("")
        
        # Step-by-step table
        lines.append("-" * 80)
        lines.append("STEP-BY-STEP ALGORITHM SELECTION")
        lines.append("-" * 80)
        lines.append(f"{'Step':<6} {'Action':<8} {'Algorithm':<12} {'Cost':<12} {'Reward':<10} {'Probabilities'}")
        lines.append("-" * 80)
        
        for i in range(len(episode_data['steps'])):
            step = episode_data['steps'][i]
            action = episode_data['actions'][i]
            algo = algo_names[action]
            cost = episode_data['costs'][i]
            reward = episode_data['rewards'][i]
            
            # Format probabilities
            if isinstance(episode_data['step_details'][i], np.ndarray):
                probs = episode_data['step_details'][i]
                prob_str = " ".join([f"{name}:{prob:.3f}" for name, prob in zip(algo_names, probs)])
            else:
                prob_str = episode_data['step_details'][i]
            
            lines.append(f"{step:<6} {action:<8} {algo:<12} {cost:<12.4f} {reward:<10.4f} {prob_str}")
        
        lines.append("=" * 80)
        lines.append("")
    
    return "\n".join(lines)


def evaluate(args):
    """Main evaluation loop."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    start_time = time.time()
    
    # 1. Load Model
    print(f"Loading model: {args.model}")
    model, model_type = load_model(args.model, args.model_type)
    print(f"Type: {model_type.upper()}")
    
    # Check for Normalization stats
    vec_norm = None
    norm_path = os.path.join(os.path.dirname(args.model), f"{Path(args.model).stem}_vecnormalize.pkl")
    if not os.path.exists(norm_path):
        norm_path = os.path.join(os.path.dirname(args.model), "vecnormalize.pkl")
        
    if os.path.exists(norm_path):
        print(f"Loading normalization stats from: {norm_path}")
        vec_norm = VecNormalize.load(norm_path, DummyVecEnv([lambda: None]))
    else:
        print("No normalization stats found. Assuming raw observations.")

    # 2. Prepare Instances
    if args.tsplib_dir:
        print(f"Loading TSPLIB from {args.tsplib_dir}")
        instances = load_all_instances(args.tsplib_dir, augment=False)
        if args.num_test_instances < len(instances):
             instances = instances[:args.num_test_instances]
    else:
        print("Generating synthetic instances...")
        np.random.seed(args.seed)
        instances = []
        for i in range(args.num_test_instances):
             instances.append(TSPProblem(num_cities=args.num_cities, distribution=args.instance_type, seed=args.seed+i))
             
    print(f"Evaluating on {len(instances)} instances.")
    
    # 3. Evaluation Loop
    all_results = []
    all_baselines = [] if args.run_baselines else None
    
    for i, problem in enumerate(instances):
        if args.verbose:
            print(f"Eval {i+1}/{len(instances)}: {getattr(problem, 'name', 'Synthetic')}")
            
        # RL Eval
        data = evaluate_rl_agent(model, model_type, problem, args, i, vec_norm)
        all_results.append(data)
        
        # Baselines Eval
        if args.run_baselines:
            print(f"  Running baselines for instance {i}...")
            b_results = run_baselines_comparison(problem, args)
            all_baselines.append(b_results)
            
            best_b = min([b['cost'] for b in b_results.values()])
            rl_cost = data['final_cost']
            gap = ((rl_cost - best_b) / best_b) * 100
            
            print(f"  RL: {rl_cost:.2f} | Best Baseline: {best_b:.2f} | Gap: {gap:+.2f}%")
    
    total_time = time.time() - start_time
    
    # 4. Generate and Save Report
    report = format_report(args, model_type, all_results, all_baselines, total_time, timestamp)
    
    # Print summary
    costs = [r['final_cost'] for r in all_results]
    print("\n" + "="*80)
    print(f"EVALUATION COMPLETE ({model_type.upper()})")
    print("="*80)
    print(f"Mean Cost: {np.mean(costs):.6f} ± {np.std(costs):.6f}")
    print(f"Best:      {np.min(costs):.6f}")
    print(f"Worst:     {np.max(costs):.6f}")
    print(f"Time:      {total_time:.2f}s")
    
    # Save to file
    os.makedirs(args.results_dir, exist_ok=True)
    run_name = args.run_name or f"eval_{model_type}_{timestamp}"
    out_path = os.path.join(args.results_dir, f"{run_name}.txt")
    
    with open(out_path, 'w') as f:
        f.write(report)
        
    print(f"\nDetailed report saved to: {out_path}")


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)