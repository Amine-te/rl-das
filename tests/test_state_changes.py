"""
Test state differentiation to verify states change between intervals.

This test ensures that the state representation changes meaningfully
when different algorithms are executed, which is critical for the
RL agent to learn switching patterns.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from problems import TSPProblem
from algorithms import GeneticAlgorithm, TabuSearch
from core import DASGymEnv


def test_state_changes_between_intervals():
    """Test that state vectors change between intervals."""
    print("=" * 80)
    print("TEST: State Differentiation")
    print("=" * 80)
    
    # Create problem and environment
    problem = TSPProblem(num_cities=30, distribution='random', seed=42)
    algorithms = [
        GeneticAlgorithm(problem, population_size=30),
        TabuSearch(problem, tabu_tenure=15)
    ]
    
    env = DASGymEnv(
        problem=problem,
        algorithms=algorithms,
        max_fes=5000,
        interval_fes=500,
        population_size=20
    )
    
    # Reset and collect states
    obs, info = env.reset()
    states = [obs.copy()]
    actions = []
    
    print(f"\nInitial state shape: {obs.shape}")
    print(f"Expected state dim: {env.das_env.state_extractor.state_dim}")
    assert obs.shape[0] == env.das_env.state_extractor.state_dim, "State dimension mismatch!"
    
    # Run 5 intervals
    for i in range(5):
        # Alternate between algorithms
        action = i % 2
        actions.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(obs.copy())
        
        if terminated or truncated:
            break
    
    print(f"\nCollected {len(states)} states")
    
    # Analyze state changes
    print("\n" + "-" * 80)
    print("State Change Analysis")
    print("-" * 80)
    
    state_diffs = []
    for i in range(1, len(states)):
        diff = np.linalg.norm(states[i] - states[i-1])
        state_diffs.append(diff)
        print(f"State {i-1} -> {i}: L2 norm difference = {diff:.6f}")
        print(f"  Action: {actions[i-1]} ({'GA' if actions[i-1] == 0 else 'TS'})")
        
        # Check specific features
        # 13 features: 5 LA, 4 Algo, 4 Stagnation
        la_features_prev = states[i-1][:5]
        la_features_curr = states[i][:5]
        algo_prev = states[i-1][5:9]
        algo_curr = states[i][5:9]
        stag_prev = states[i-1][9:]
        stag_curr = states[i][9:]
        
        la_diff = np.linalg.norm(la_features_curr - la_features_prev)
        algo_diff = np.linalg.norm(algo_curr - algo_prev)
        stag_diff = np.linalg.norm(stag_curr - stag_prev)
        
        print(f"  LA features diff: {la_diff:.6f}")
        print(f"  Algo features diff: {algo_diff:.6f}")
        print(f"  Stagnation diff: {stag_diff:.6f}")
        
        # Verify current algorithm one-hot encoding
        current_algo = np.argmax(algo_curr) if np.max(algo_curr) > 0 else None
        print(f"  Current algo state: {current_algo} (Should be {actions[i-1]})")
        
        # Verify stagnation updates
        print(f"  Stagnation counters: {stag_curr}")
    
    # Verify states are changing
    avg_diff = np.mean(state_diffs)
    min_diff = np.min(state_diffs)
    max_diff = np.max(state_diffs)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Average state change: {avg_diff:.6f}")
    print(f"Min state change:     {min_diff:.6f}")
    print(f"Max state change:     {max_diff:.6f}")
    
    # Success criteria
    threshold = 0.01
    if min_diff > threshold:
        print(f"\n✓ SUCCESS: All states differ by > {threshold}")
        print("  States are changing meaningfully between intervals")
        return True
    else:
        print(f"\n✗ FAILURE: Some states differ by < {threshold}")
        print("  States are too similar - agent may not learn to switch")
        return False


if __name__ == '__main__':
    success = test_state_changes_between_intervals()
    sys.exit(0 if success else 1)
