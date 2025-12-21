"""
Validate Dynamic Algorithm Selection Hypothesis.

This script tests whether DAS actually provides benefits over:
1. Running each algorithm individually for full budget
2. Oracle selection (best algorithm per instance, known in hindsight)
3. Random algorithm selection

Key question: Is the RL agent learning meaningful switching patterns,
or is "run TS for full budget" actually optimal?
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from problems import TSPProblem
from algorithms import GeneticAlgorithm, TabuSearch, SimulatedAnnealing, IteratedLocalSearch


def run_single_algorithm(problem: TSPProblem, algo_class, max_fes: int) -> Dict:
    """Run a single algorithm for full budget."""
    algo = algo_class(problem)
    algo.initialize()
    
    remaining_fes = max_fes - algo.evaluations_used
    if remaining_fes > 0:
        algo.step(remaining_fes)
    
    return {
        'final_cost': algo.best_cost,
        'evaluations_used': algo.evaluations_used
    }


def run_oracle_das(problem: TSPProblem, max_fes: int, interval_fes: int) -> Dict:
    """
    Oracle DAS: At each interval, select the algorithm that would give best result.
    
    This is an UPPER BOUND on what any DAS method could achieve.
    """
    algorithms = [
        GeneticAlgorithm(problem, population_size=min(50, problem.size)),
        TabuSearch(problem, tabu_tenure=min(20, problem.size // 2)),
        SimulatedAnnealing(problem, initial_temperature=100.0),
        IteratedLocalSearch(problem, perturbation_strength=max(2, problem.size // 10))
    ]
    
    # Initialize all
    for algo in algorithms:
        algo.initialize()
    
    current_fes = sum(algo.evaluations_used for algo in algorithms)
    best_cost = min(algo.best_cost for algo in algorithms)
    
    decisions = []
    
    while current_fes < max_fes:
        # Try each algorithm for interval_fes and see which gives best result
        interval_budget = min(interval_fes, max_fes - current_fes)
        
        best_algo_idx = 0
        best_result_cost = float('inf')
        
        # Save states
        saved_states = []
        for algo in algorithms:
            saved_states.append({
                'best_cost': algo.best_cost,
                'best_solution': algo.best_solution.copy() if hasattr(algo.best_solution, 'copy') else algo.best_solution[:]
            })
        
        # Try each algorithm
        for i, algo in enumerate(algorithms):
            # Restore state
            algo.best_cost = saved_states[i]['best_cost']
            algo.best_solution = saved_states[i]['best_solution']
            
            # Run for interval
            _, result_cost = algo.step(interval_budget)
            
            if result_cost < best_result_cost:
                best_result_cost = result_cost
                best_algo_idx = i
        
        # Actually execute best algorithm
        best_algo = algorithms[best_algo_idx]
        best_algo.best_cost = saved_states[best_algo_idx]['best_cost']
        best_algo.best_solution = saved_states[best_algo_idx]['best_solution']
        best_algo.step(interval_budget)
        
        decisions.append(best_algo_idx)
        current_fes += interval_budget
        best_cost = min(algo.best_cost for algo in algorithms)
    
    return {
        'final_cost': best_cost,
        'decisions': decisions,
        'algorithm_distribution': np.bincount(decisions, minlength=4).tolist()
    }


def run_random_das(problem: TSPProblem, max_fes: int, interval_fes: int, seed: int = 42) -> Dict:
    """Random DAS: Select algorithms randomly at each interval."""
    np.random.seed(seed)
    
    algorithms = [
        GeneticAlgorithm(problem, population_size=min(50, problem.size)),
        TabuSearch(problem, tabu_tenure=min(20, problem.size // 2)),
        SimulatedAnnealing(problem, initial_temperature=100.0),
        IteratedLocalSearch(problem, perturbation_strength=max(2, problem.size // 10))
    ]
    
    for algo in algorithms:
        algo.initialize()
    
    current_fes = sum(algo.evaluations_used for algo in algorithms)
    decisions = []
    
    while current_fes < max_fes:
        interval_budget = min(interval_fes, max_fes - current_fes)
        
        # Random selection
        algo_idx = np.random.randint(0, 4)
        algorithms[algo_idx].step(interval_budget)
        
        decisions.append(algo_idx)
        current_fes += interval_budget
    
    best_cost = min(algo.best_cost for algo in algorithms)
    
    return {
        'final_cost': best_cost,
        'decisions': decisions,
        'algorithm_distribution': np.bincount(decisions, minlength=4).tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Validate DAS Hypothesis')
    parser.add_argument('--num-instances', type=int, default=20)
    parser.add_argument('--num-cities', type=int, default=50)
    parser.add_argument('--max-fes', type=int, default=10000)
    parser.add_argument('--interval-fes', type=int, default=500)
    parser.add_argument('--seed', type=int, default=999)
    args = parser.parse_args()
    
    print("=" * 80)
    print("VALIDATING DYNAMIC ALGORITHM SELECTION HYPOTHESIS")
    print("=" * 80)
    print(f"Testing on {args.num_instances} TSP instances ({args.num_cities} cities)")
    print(f"Budget: {args.max_fes} FEs, Interval: {args.interval_fes} FEs")
    print()
    
    # Generate test instances
    np.random.seed(args.seed)
    instances = [
        TSPProblem(num_cities=args.num_cities, distribution='random', seed=args.seed + i)
        for i in range(args.num_instances)
    ]
    
    # Test each approach
    results = {
        'GA': [],
        'TS': [],
        'SA': [],
        'ILS': [],
        'Oracle-DAS': [],
        'Random-DAS': []
    }
    
    algo_classes = [GeneticAlgorithm, TabuSearch, SimulatedAnnealing, IteratedLocalSearch]
    algo_names = ['GA', 'TS', 'SA', 'ILS']
    
    print("Running experiments...")
    for i, problem in enumerate(instances):
        print(f"\rInstance {i+1}/{args.num_instances}...", end='', flush=True)
        
        # Single algorithms
        for algo_class, name in zip(algo_classes, algo_names):
            result = run_single_algorithm(problem, algo_class, args.max_fes)
            results[name].append(result['final_cost'])
        
        # Oracle DAS
        oracle_result = run_oracle_das(problem, args.max_fes, args.interval_fes)
        results['Oracle-DAS'].append(oracle_result['final_cost'])
        
        # Random DAS
        random_result = run_random_das(problem, args.max_fes, args.interval_fes, args.seed + i)
        results['Random-DAS'].append(random_result['final_cost'])
    
    print("\n")
    
    # Analyze results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"{'Method':<15} {'Mean Cost':<15} {'Std':<12} {'Best':<12} {'Worst':<12}")
    print("-" * 80)
    
    stats = {}
    for method in ['GA', 'TS', 'SA', 'ILS', 'Random-DAS', 'Oracle-DAS']:
        costs = results[method]
        stats[method] = {
            'mean': np.mean(costs),
            'std': np.std(costs),
            'best': np.min(costs),
            'worst': np.max(costs)
        }
        print(f"{method:<15} {stats[method]['mean']:<15.6f} {stats[method]['std']:<12.6f} "
              f"{stats[method]['best']:<12.6f} {stats[method]['worst']:<12.6f}")
    
    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Find best single algorithm
    best_single = min(algo_names, key=lambda x: stats[x]['mean'])
    best_single_cost = stats[best_single]['mean']
    
    print(f"Best single algorithm: {best_single} (mean cost: {best_single_cost:.6f})")
    print()
    
    # Compare Oracle-DAS vs best single
    oracle_cost = stats['Oracle-DAS']['mean']
    oracle_improvement = ((best_single_cost - oracle_cost) / best_single_cost) * 100
    
    print(f"Oracle-DAS mean cost: {oracle_cost:.6f}")
    print(f"Improvement over best single: {oracle_improvement:+.2f}%")
    print()
    
    if oracle_improvement > 1.0:
        print("✓ Oracle-DAS shows >1% improvement over best single algorithm")
        print("  → DAS has theoretical potential (if agent can learn optimal switching)")
    else:
        print("✗ Oracle-DAS shows <1% improvement over best single algorithm")
        print("  → DAS may not be worthwhile for this problem/configuration")
    
    print()
    
    # Compare Random-DAS vs best single
    random_cost = stats['Random-DAS']['mean']
    random_vs_best = ((best_single_cost - random_cost) / best_single_cost) * 100
    
    print(f"Random-DAS mean cost: {random_cost:.6f}")
    print(f"Difference vs best single: {random_vs_best:+.2f}%")
    print()
    
    if random_cost < best_single_cost:
        print("✓ Even random switching beats best single algorithm")
        print("  → Problem benefits from algorithm diversity")
    else:
        print("✗ Random switching worse than best single algorithm")
        print("  → Switching has overhead or best algorithm dominates")
    
    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if oracle_improvement > 5.0:
        print("Strong evidence that DAS can provide significant benefits.")
        print("The RL agent should be able to learn meaningful switching patterns.")
    elif oracle_improvement > 1.0:
        print("Moderate evidence that DAS can provide benefits.")
        print("Gains are modest but may be worth pursuing with good agent design.")
    else:
        print("Weak evidence for DAS benefits on this problem.")
        print("Consider: (1) increasing interval size, (2) different algorithms,")
        print("         (3) different problem sizes, or (4) accepting single-algo is best")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
