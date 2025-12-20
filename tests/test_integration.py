"""
Integration tests for RL-DAS core components.

These tests verify that all core components work together correctly:
- State extraction produces valid vectors
- Reward calculation works properly
- Context management preserves algorithm states
- Environment orchestrates everything correctly
- Gymnasium wrapper is compatible

Run with: pytest tests/test_integration.py -v
"""

import sys
import os
import numpy as np
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems import TSPProblem, BaseProblem
from algorithms import RandomSearchAlgorithm, LocalSearchAlgorithm, GreedySearchAlgorithm
from core import (
    StateExtractor, 
    RewardCalculator, 
    ContextManager, 
    DASEnvironment,
    DASGymEnv,
    make_das_env
)


class TestBaseProblem:
    """Tests for TSP problem implementation."""
    
    def test_tsp_creation(self):
        """Test TSP problem can be created."""
        problem = TSPProblem(num_cities=20, seed=42)
        
        assert problem.problem_type == 'TSP'
        assert problem.size == 20
        assert problem.cities.shape == (20, 2)
        assert problem.distance_matrix.shape == (20, 20)
    
    def test_tsp_evaluation(self):
        """Test TSP evaluation works."""
        problem = TSPProblem(num_cities=10, seed=42)
        
        solution = problem.generate_random_solution()
        cost = problem.evaluate(solution)
        
        assert cost > 0
        assert problem.get_evaluation_count() == 1
    
    def test_tsp_neighbors(self):
        """Test neighbor generation."""
        problem = TSPProblem(num_cities=20, seed=42)
        solution = problem.generate_random_solution()
        
        neighbors = problem.generate_neighbors(solution, 5)
        
        assert len(neighbors) == 5
        for neighbor in neighbors:
            assert len(neighbor) == 20
            assert problem.is_feasible(neighbor)
    
    def test_tsp_distance(self):
        """Test solution distance calculation."""
        problem = TSPProblem(num_cities=20, seed=42)
        
        sol1 = problem.generate_random_solution()
        sol2 = problem.generate_random_solution()
        
        dist = problem.solution_distance(sol1, sol2)
        
        assert 0 <= dist <= 1
        assert problem.solution_distance(sol1, sol1) == 0
    
    def test_tsp_normalization(self):
        """Test cost normalization."""
        problem = TSPProblem(num_cities=20, seed=42)
        solution = problem.generate_random_solution()
        cost = problem.evaluate(solution)
        
        normalized = problem.normalize_cost(cost)
        
        assert 0 <= normalized <= 1


class TestStateExtractor:
    """Tests for state extraction."""
    
    def test_state_extraction(self):
        """Test full state extraction."""
        problem = TSPProblem(num_cities=20, seed=42)
        extractor = StateExtractor(num_algorithms=3, neighbor_sample_size=10)
        
        # Create a population
        population = []
        for _ in range(10):
            sol = problem.generate_random_solution()
            cost = problem.evaluate(sol)
            population.append((sol, cost))
        
        best_sol, best_cost = min(population, key=lambda x: x[1])
        
        state = extractor.extract_state(
            problem, best_sol, best_cost, population,
            current_fes=10, max_fes=1000
        )
        
        # Check state dimensions
        assert state.shape == (extractor.state_dim,)
        assert state.shape == (9 + 2 * 3,)  # 9 LA + 6 AH
        
        # Check all values are in [0, 1]
        assert np.all(state >= 0)
        assert np.all(state <= 1)
    
    def test_la_features(self):
        """Test LA features extraction."""
        problem = TSPProblem(num_cities=20, seed=42)
        extractor = StateExtractor(num_algorithms=3, neighbor_sample_size=10)
        
        population = []
        for _ in range(10):
            sol = problem.generate_random_solution()
            cost = problem.evaluate(sol)
            population.append((sol, cost))
        
        best_sol, best_cost = min(population, key=lambda x: x[1])
        
        la = extractor.extract_la_features(
            problem, best_sol, best_cost, population,
            current_fes=50, max_fes=100
        )
        
        assert la.shape == (9,)
        assert np.all(la >= 0)
        assert np.all(la <= 1)
        
        # Budget consumed should be 0.5
        assert abs(la[8] - 0.5) < 0.1
    
    def test_ah_features_update(self):
        """Test algorithm history update."""
        problem = TSPProblem(num_cities=20, seed=42)
        extractor = StateExtractor(num_algorithms=3)
        
        sol1 = problem.generate_random_solution()
        sol2 = problem.generate_random_solution()
        sol3 = problem.generate_random_solution()
        sol4 = problem.generate_random_solution()
        
        # Initial AH should be zeros
        ah = extractor.extract_ah_features()
        assert np.all(ah == 0)
        
        # Update history for algorithm 0
        extractor.update_algorithm_history(0, sol1, sol2, sol3, sol4, problem)
        
        ah = extractor.extract_ah_features()
        # Algorithm 0's features should be non-zero now
        assert ah[0] > 0 or ah[1] > 0  # At least one should change


class TestRewardCalculator:
    """Tests for reward calculation."""
    
    def test_basic_reward(self):
        """Test basic reward calculation."""
        calc = RewardCalculator(max_fes=1000)
        calc.reset(initial_cost=100.0)
        
        # Improvement of 10%
        reward = calc.compute_step_reward(
            prev_cost=100.0,
            curr_cost=90.0,
            fes_used=100
        )
        
        assert reward > 0  # Improvement should give positive reward
    
    def test_no_improvement_reward(self):
        """Test reward when no improvement."""
        calc = RewardCalculator(max_fes=1000)
        calc.reset(initial_cost=100.0)
        
        reward = calc.compute_step_reward(
            prev_cost=100.0,
            curr_cost=100.0,
            fes_used=100
        )
        
        assert reward == 0  # No improvement = no reward
    
    def test_speed_factor(self):
        """Test speed factor application."""
        calc = RewardCalculator(max_fes=1000)
        calc.reset(initial_cost=100.0)
        
        # Record two improvements
        calc.compute_step_reward(100.0, 90.0, 100)
        calc.compute_step_reward(90.0, 80.0, 200)
        
        # Finalize with fast termination
        fast_rewards = calc.finalize_episode(final_fes=200)
        
        # Reset and do same with slow termination
        calc.reset(initial_cost=100.0)
        calc.compute_step_reward(100.0, 90.0, 100)
        calc.compute_step_reward(90.0, 80.0, 200)
        slow_rewards = calc.finalize_episode(final_fes=800)
        
        # Fast should have higher multiplier
        assert sum(fast_rewards) > sum(slow_rewards)
    
    def test_milestone_bonus(self):
        """Test milestone bonus application."""
        calc = RewardCalculator(max_fes=1000, milestone_bonus=1.5)
        calc.reset(initial_cost=100.0)
        
        # Regular improvement
        reward1 = calc.compute_step_reward(
            prev_cost=100.0,
            curr_cost=90.0,
            fes_used=100,
            is_global_best=False
        )
        
        calc.reset(initial_cost=100.0)
        
        # Same improvement but with milestone
        reward2 = calc.compute_step_reward(
            prev_cost=100.0,
            curr_cost=90.0,
            fes_used=100,
            is_global_best=True
        )
        
        assert reward2 > reward1


class TestContextManager:
    """Tests for context management."""
    
    def test_empty_context(self):
        """Test initial context is empty."""
        cm = ContextManager(num_algorithms=3)
        
        for i in range(3):
            assert cm.context['algorithms'][i]['initialized'] == False
        
        assert cm.context['common']['global_best_cost'] == float('inf')
    
    def test_save_restore_context(self):
        """Test saving and restoring algorithm context."""
        problem = TSPProblem(num_cities=10, seed=42)
        algo = LocalSearchAlgorithm(problem)
        algo.initialize()
        
        cm = ContextManager(num_algorithms=3)
        
        # Run algorithm a bit
        algo.step(100)
        original_best = algo.best_cost
        original_solution = algo.best_solution.copy()
        restart_count = algo.restart_count
        
        # Save context
        cm.save_algorithm_context(0, algo)
        
        # Verify context was saved
        assert cm.context['algorithms'][0]['initialized'] == True
        assert cm.context['algorithms'][0]['state'] is not None
        
        # Modify algorithm substantially
        algo.best_cost = float('inf')
        algo.best_solution = None
        algo.restart_count = 999
        
        # Restore context
        cm.restore_algorithm_context(0, algo)
        
        # Verify state was restored
        assert algo.best_cost == original_best
        assert np.array_equal(algo.best_solution, original_solution)
        assert algo.restart_count == restart_count
    
    def test_common_context_update(self):
        """Test common context updates."""
        problem = TSPProblem(num_cities=10, seed=42)
        cm = ContextManager(num_algorithms=3, elite_size=3)
        
        sol1 = problem.generate_random_solution()
        cost1 = problem.evaluate(sol1)
        
        cm.update_common_context(sol1, cost1, 0, 100)
        
        assert cm.context['common']['global_best_cost'] == cost1
        assert len(cm.context['common']['elite_archive']) == 1
    
    def test_elite_archive(self):
        """Test elite archive maintenance."""
        problem = TSPProblem(num_cities=10, seed=42)
        cm = ContextManager(num_algorithms=3, elite_size=3)
        
        # Add several solutions
        costs = []
        for i in range(5):
            sol = problem.generate_random_solution()
            cost = problem.evaluate(sol)
            cm.update_common_context(sol, cost, 0, i * 10)
            costs.append(cost)
        
        # Should keep only 3 best
        assert len(cm.context['common']['elite_archive']) == 3
        
        # Archive should be sorted
        archive_costs = [c for _, c in cm.context['common']['elite_archive']]
        assert archive_costs == sorted(archive_costs)


class TestDASEnvironment:
    """Tests for DAS environment."""
    
    def test_environment_creation(self):
        """Test environment can be created."""
        problem = TSPProblem(num_cities=20, seed=42)
        algorithms = [
            RandomSearchAlgorithm(problem),
            LocalSearchAlgorithm(problem),
            GreedySearchAlgorithm(problem)
        ]
        
        env = DASEnvironment(
            problem=problem,
            algorithms=algorithms,
            max_fes=10000,
            interval_fes=500
        )
        
        assert env.state_dim == 9 + 2 * 3  # 15
        assert env.action_dim == 3
    
    def test_environment_reset(self):
        """Test environment reset."""
        problem = TSPProblem(num_cities=20, seed=42)
        algorithms = [
            RandomSearchAlgorithm(problem),
            LocalSearchAlgorithm(problem),
            GreedySearchAlgorithm(problem)
        ]
        
        env = DASEnvironment(
            problem=problem,
            algorithms=algorithms,
            max_fes=10000,
            interval_fes=500
        )
        
        state = env.reset()
        
        assert state.shape == (env.state_dim,)
        assert np.all(state >= 0)
        assert np.all(state <= 1)
    
    def test_environment_step(self):
        """Test environment step."""
        problem = TSPProblem(num_cities=20, seed=42)
        algorithms = [
            RandomSearchAlgorithm(problem),
            LocalSearchAlgorithm(problem),
            GreedySearchAlgorithm(problem)
        ]
        
        env = DASEnvironment(
            problem=problem,
            algorithms=algorithms,
            max_fes=5000,
            interval_fes=500,
            population_size=20
        )
        
        state = env.reset()
        
        # Take a step
        next_state, reward, terminated, truncated, info = env.step(action=0)
        
        assert next_state.shape == state.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert 'current_fes' in info
        assert 'best_cost' in info
    
    def test_full_episode(self):
        """Test running a full episode."""
        problem = TSPProblem(num_cities=20, seed=42)
        algorithms = [
            RandomSearchAlgorithm(problem),
            LocalSearchAlgorithm(problem),
            GreedySearchAlgorithm(problem)
        ]
        
        env = DASEnvironment(
            problem=problem,
            algorithms=algorithms,
            max_fes=3000,
            interval_fes=500,
            population_size=10
        )
        
        state = env.reset()
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = np.random.randint(0, 3)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Safety limit
            if steps > 100:
                break
        
        assert done
        assert steps > 0
        
        # Get episode info
        episode_info = env.get_episode_info()
        assert 'final_cost' in episode_info
        assert 'total_steps' in episode_info


class TestDASGymEnv:
    """Tests for Gymnasium wrapper."""
    
    def test_gym_env_creation(self):
        """Test Gym env can be created."""
        problem = TSPProblem(num_cities=20, seed=42)
        algorithms = [
            RandomSearchAlgorithm(problem),
            LocalSearchAlgorithm(problem),
            GreedySearchAlgorithm(problem)
        ]
        
        env = make_das_env(
            problem=problem,
            algorithms=algorithms,
            max_fes=5000,
            interval_fes=500
        )
        
        assert env.observation_space.shape[0] == 15
        assert env.action_space.n == 3
    
    def test_gym_env_spaces(self):
        """Test observation and action spaces."""
        problem = TSPProblem(num_cities=20, seed=42)
        algorithms = [
            RandomSearchAlgorithm(problem),
            LocalSearchAlgorithm(problem)
        ]
        
        env = DASGymEnv(
            problem=problem,
            algorithms=algorithms,
            max_fes=5000
        )
        
        state, info = env.reset()
        
        assert env.observation_space.contains(state)
        
        for action in range(env.action_space.n):
            assert env.action_space.contains(action)
    
    def test_gym_env_episode(self):
        """Test running episode through Gym interface."""
        problem = TSPProblem(num_cities=20, seed=42)
        algorithms = [
            RandomSearchAlgorithm(problem),
            LocalSearchAlgorithm(problem),
            GreedySearchAlgorithm(problem)
        ]
        
        env = make_das_env(
            problem=problem,
            algorithms=algorithms,
            max_fes=3000,
            interval_fes=500,
            population_size=10
        )
        
        state, info = env.reset(seed=42)
        
        done = False
        steps = 0
        
        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            if steps > 100:
                break
        
        assert done
        
        # Get summary
        summary = env.get_episode_summary()
        assert 'final_cost' in summary
        assert 'episode_actions' in summary


class TestGeneralizability:
    """Tests for generalizability across problem sizes."""
    
    def test_state_consistency_across_sizes(self):
        """Test that state dimensions are consistent across problem sizes."""
        extractor = StateExtractor(num_algorithms=3, neighbor_sample_size=10)
        
        for num_cities in [10, 20, 50, 100]:
            problem = TSPProblem(num_cities=num_cities, seed=42)
            
            population = []
            for _ in range(10):
                sol = problem.generate_random_solution()
                cost = problem.evaluate(sol)
                population.append((sol, cost))
            
            best_sol, best_cost = min(population, key=lambda x: x[1])
            
            state = extractor.extract_state(
                problem, best_sol, best_cost, population,
                current_fes=10, max_fes=1000
            )
            
            # State dimension should be the same regardless of problem size
            assert state.shape == (15,), f"Failed for {num_cities} cities"
            
            # All values should be normalized
            assert np.all(state >= 0), f"Negative values for {num_cities} cities"
            assert np.all(state <= 1), f"Values > 1 for {num_cities} cities"
    
    def test_different_distributions(self):
        """Test with different city distributions."""
        extractor = StateExtractor(num_algorithms=3, neighbor_sample_size=10)
        
        for distribution in ['random', 'clustered', 'grid']:
            problem = TSPProblem(num_cities=30, distribution=distribution, seed=42)
            
            population = []
            for _ in range(10):
                sol = problem.generate_random_solution()
                cost = problem.evaluate(sol)
                population.append((sol, cost))
            
            best_sol, best_cost = min(population, key=lambda x: x[1])
            
            state = extractor.extract_state(
                problem, best_sol, best_cost, population,
                current_fes=10, max_fes=1000
            )
            
            assert state.shape == (15,)
            assert np.all(state >= 0)
            assert np.all(state <= 1)


def run_all_tests():
    """Run all tests and print results."""
    print("=" * 60)
    print("RL-DAS Integration Test Suite")
    print("=" * 60)
    
    test_classes = [
        TestBaseProblem,
        TestStateExtractor,
        TestRewardCalculator,
        TestContextManager,
        TestDASEnvironment,
        TestDASGymEnv,
        TestGeneralizability
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    instance = test_class()
                    method = getattr(instance, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ✗ {method_name}")
                    print(f"    Error: {e}")
                    failed_tests.append((test_class.__name__, method_name, str(e)))
    
    print("\n" + "=" * 60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    
    if failed_tests:
        print("\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
    else:
        print("\n✓ All tests passed!")
    
    print("=" * 60)
    
    return len(failed_tests) == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
