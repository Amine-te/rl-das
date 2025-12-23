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
from utils.tsplib_loader import load_all_instances
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
                detail_str = " ".join([f"A{i}:{p:.2f}" for i, p in enumerate(probs)])
        
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
    
    return episode_data


def run_baselines_comparison(problem, args) -> Dict[str, float]:
    """Run all baselines on the problem."""
    results = {}
    algorithms = create_algorithms(problem)
    algo_names = ['GA', 'TS', 'SA', 'ILS']
    
    for name, algo in zip(algo_names, algorithms):
        # Reset/Initialize
        algo.initialize()
        
        # Run for max_fes
        # Most algos here are iterative. We can just step them until max_fes.
        # But their step() uses interval.
        # We can just use a loop.
        current_fes = 0
        best_cost = float('inf')
        
        # Initial solution
        sol = problem.generate_random_solution()
        cost = problem.evaluate(sol)
        algo.inject_solution(sol, cost)
        
        while current_fes < args.max_fes:
            fes_allowed = min(args.interval_fes, args.max_fes - current_fes)
            _, cost = algo.step(fes_allowed)
            current_fes += fes_allowed # approx
            if cost < best_cost:
                best_cost = cost
                
        results[name] = best_cost
        
    return results


def format_episode_report(episode_data: Dict, algo_names: List[str], baseline_results: Optional[Dict] = None) -> str:
    """Format detailed report."""
    lines = []
    lines.append(f"INSTANCE: {episode_data['problem_name']}")
    lines.append(f"Initial: {episode_data['initial_cost']:.2f} -> Final: {episode_data['final_cost']:.2f}")
    
    if baseline_results:
        best_baseline = min(baseline_results.values())
        rl_cost = episode_data['final_cost']
        gap = ((rl_cost - best_baseline) / best_baseline) * 100
        
        lines.append("Baselines:")
        for name, cost in baseline_results.items():
            lines.append(f"  {name}: {cost:.2f}")
        lines.append(f"  Best Baseline: {best_baseline:.2f}")
        lines.append(f"  RL Gap: {gap:+.2f}% (Negative is better)")
        
    lines.append("-" * 60)
    lines.append(f"{'Step':<5} {'Act':<5} {'Algo':<5} {'Cost':<10} {'Rew':<6} {'Introspection'}")
    
    for i in range(len(episode_data['steps'])):
        a_idx = episode_data['actions'][i]
        lines.append(f"{episode_data['steps'][i]:<5} {a_idx:<5} {algo_names[a_idx]:<5} "
                     f"{episode_data['costs'][i]:<10.2f} {episode_data['rewards'][i]:<6.2f} "
                     f"{episode_data['step_details'][i]}")
    lines.append("")
    return "\n".join(lines)


def evaluate(args):
    """Main evaluation loop."""
    # 1. Load Model
    print(f"Loading model: {args.model}")
    model, model_type = load_model(args.model, args.model_type)
    print(f"Type: {model_type.upper()}")
    
    # Check for Normalization stats
    vec_norm = None
    # Check same dir as model
    norm_path = os.path.join(os.path.dirname(args.model), f"{Path(args.model).stem}_vecnormalize.pkl")
    # Also check without prefix if standard name used
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
        instances = load_all_instances(args.tsplib_dir, augment=False) # No augment for eval typically
        if args.num_test_instances < len(instances):
             instances = instances[:args.num_test_instances]
    else:
        print("Generating synthetic instances...")
        np.random.seed(args.seed)
        instances = []
        for i in range(args.num_test_instances):
             instances.append(TSPProblem(args.num_cities, args.instance_type, seed=args.seed+i))
             
    print(f"Evaluating on {len(instances)} instances.")
    
    # 3. Validation Loop
    results = []
    logs = []
    algo_names = ['GA', 'TS', 'SA', 'ILS']
    baseline_agg = {'r_wins': 0, 'r_gap': []}
    
    for i, problem in enumerate(instances):
        if args.verbose:
            print(f"Eval {i+1}/{len(instances)}: {getattr(problem, 'name', 'Synthetic')}")
            
        # RL Eval
        data = evaluate_rl_agent(model, model_type, problem, args, i, vec_norm)
        results.append(data)
        
        # Baselines Eval
        b_results = None
        if args.run_baselines:
            print(f"  Running baselines for {getattr(problem, 'name', 'Instance')}...")
            b_results = run_baselines_comparison(problem, args)
            best_b = min(b_results.values())
            rl_cost = data['final_cost']
            
            # Stats
            if rl_cost < best_b:
                baseline_agg['r_wins'] += 1
            gap = ((rl_cost - best_b) / best_b) * 100
            baseline_agg['r_gap'].append(gap)
            
            print(f"  RL: {rl_cost:.2f} | Best Baseline: {best_b:.2f} | Gap: {gap:+.2f}%")
        
        logs.append(format_episode_report(data, algo_names, b_results))
        
    # 4. Report
    costs = [r['final_cost'] for r in results]
    print("\n" + "="*60)
    print(f"RESULTS ({model_type.upper()})")
    print("="*60)
    print(f"Mean Cost: {np.mean(costs):.2f} +/- {np.std(costs):.2f}")
    print(f"Best:      {np.min(costs):.2f}")
    print(f"Worst:     {np.max(costs):.2f}")
    
    if args.run_baselines and baseline_agg['r_gap']:
        win_rate = (baseline_agg['r_wins'] / len(instances)) * 100
        mean_gap = np.mean(baseline_agg['r_gap'])
        print("-" * 60)
        print(f"BASELINE COMPARISON:")
        print(f"RL Win Rate: {win_rate:.1f}%")
        print(f"Mean Gap:    {mean_gap:+.2f}% (Negative = RL Better)")
    
    # Save to file
    os.makedirs(args.results_dir, exist_ok=True)
    if args.run_name:
        fname = args.run_name
    else:
        fname = f"eval_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    out_path = os.path.join(args.results_dir, f"{fname}.txt")
    with open(out_path, 'w') as f:
        f.write(f"EVALUATION REPORT: {fname}\n")
        f.write(f"Mean Cost: {np.mean(costs):.2f}\n")
        if args.run_baselines and baseline_agg['r_gap']:
             f.write(f"RL Win Rate: {win_rate:.1f}%\n")
             f.write(f"Mean Gap:    {mean_gap:+.2f}%\n")
        f.write("\n" + "="*60 + "\n\n")
        f.write("\n".join(logs))
        f.write(f"\nSUMMARY:\nMean: {np.mean(costs)}\nStd: {np.std(costs)}\n")
        
    print(f"Detailed logs saved to {out_path}")


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
