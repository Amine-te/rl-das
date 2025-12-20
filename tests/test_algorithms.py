"""
Tests for optimization algorithm implementations.

Verifies that each algorithm:
- Initializes correctly
- Can execute steps
- Maintains proper state
- Supports context save/restore
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems import TSPProblem
from algorithms import (
    GeneticAlgorithm, TabuSearch, SimulatedAnnealing,
    VariableNeighborhoodSearch, IteratedLocalSearch, AntColonyOptimization
)


class TestGeneticAlgorithm:
    """Tests for Genetic Algorithm."""
    
    def test_initialization(self):
        problem = TSPProblem(num_cities=20, seed=42)
        ga = GeneticAlgorithm(problem, population_size=20)
        ga.initialize()
        
        assert len(ga.population) == 20
        assert ga.best_solution is not None
        assert ga.best_cost < float('inf')
    
    def test_step_execution(self):
        problem = TSPProblem(num_cities=20, seed=42)
        ga = GeneticAlgorithm(problem, population_size=20)
        ga.initialize()
        
        initial_cost = ga.best_cost
        ga.step(100)
        
        assert ga.evaluations_used > 0
        assert ga.best_cost <= initial_cost  # Should not get worse
    
    def test_state_preservation(self):
        problem = TSPProblem(num_cities=20, seed=42)
        ga = GeneticAlgorithm(problem, population_size=20)
        ga.initialize()
        ga.step(100)
        
        state = ga.get_state()
        original_best = ga.best_cost
        
        ga.best_cost = float('inf')
        ga.set_state(state)
        
        assert ga.best_cost == original_best


class TestTabuSearch:
    """Tests for Tabu Search."""
    
    def test_initialization(self):
        problem = TSPProblem(num_cities=20, seed=42)
        ts = TabuSearch(problem, neighborhood_size=20)
        ts.initialize()
        
        assert ts.current_solution is not None
        assert ts.best_cost < float('inf')
        assert len(ts.tabu_list) == 0
    
    def test_step_execution(self):
        problem = TSPProblem(num_cities=20, seed=42)
        ts = TabuSearch(problem, neighborhood_size=20)
        ts.initialize()
        
        initial_cost = ts.best_cost
        ts.step(100)
        
        assert ts.evaluations_used > 0
        assert ts.iteration > 0
        assert ts.best_cost <= initial_cost
    
    def test_tabu_list_population(self):
        problem = TSPProblem(num_cities=20, seed=42)
        ts = TabuSearch(problem, neighborhood_size=20, tabu_tenure=5)
        ts.initialize()
        ts.step(200)
        
        # Tabu list should have some entries after running
        assert len(ts.tabu_list) > 0


class TestSimulatedAnnealing:
    """Tests for Simulated Annealing."""
    
    def test_initialization(self):
        problem = TSPProblem(num_cities=20, seed=42)
        sa = SimulatedAnnealing(problem)
        sa.initialize()
        
        assert sa.current_solution is not None
        assert sa.temperature == sa.initial_temperature
        assert sa.best_cost < float('inf')
    
    def test_temperature_cooling(self):
        problem = TSPProblem(num_cities=20, seed=42)
        sa = SimulatedAnnealing(problem)
        sa.initialize()
        
        initial_temp = sa.temperature
        sa.step(100)
        
        assert sa.temperature < initial_temp  # Should have cooled
    
    def test_step_execution(self):
        problem = TSPProblem(num_cities=20, seed=42)
        sa = SimulatedAnnealing(problem)
        sa.initialize()
        
        initial_cost = sa.best_cost
        sa.step(200)
        
        assert sa.evaluations_used > 0
        assert sa.best_cost <= initial_cost


class TestVariableNeighborhoodSearch:
    """Tests for VNS."""
    
    def test_initialization(self):
        problem = TSPProblem(num_cities=20, seed=42)
        vns = VariableNeighborhoodSearch(problem, k_max=3)
        vns.initialize()
        
        assert vns.current_solution is not None
        assert vns.k == 1
    
    def test_neighborhood_cycling(self):
        problem = TSPProblem(num_cities=20, seed=42)
        vns = VariableNeighborhoodSearch(problem, k_max=3, reduced_vns=True)
        vns.initialize()
        vns.step(50)
        
        # Should have visited some neighborhoods
        total_visits = sum(vns.neighborhood_visits.values())
        assert total_visits > 0


class TestIteratedLocalSearch:
    """Tests for ILS."""
    
    def test_initialization(self):
        problem = TSPProblem(num_cities=20, seed=42)
        ils = IteratedLocalSearch(problem)
        ils.initialize()
        
        assert ils.current_solution is not None
        assert ils.home_base_solution is not None
    
    def test_step_execution(self):
        problem = TSPProblem(num_cities=20, seed=42)
        ils = IteratedLocalSearch(problem, local_search_max_iters=10)
        ils.initialize()
        
        initial_cost = ils.best_cost
        ils.step(200)
        
        assert ils.evaluations_used > 0
        assert ils.best_cost <= initial_cost


class TestAntColonyOptimization:
    """Tests for ACO."""
    
    def test_initialization(self):
        problem = TSPProblem(num_cities=20, seed=42)
        aco = AntColonyOptimization(problem, num_ants=10, local_search=False)
        aco.initialize()
        
        assert aco.pheromone is not None
        assert aco.pheromone.shape == (20, 20)
        assert aco.best_solution is not None
    
    def test_step_execution(self):
        problem = TSPProblem(num_cities=20, seed=42)
        aco = AntColonyOptimization(problem, num_ants=10, local_search=False)
        aco.initialize()
        
        initial_cost = aco.best_cost
        aco.step(50)
        
        assert aco.evaluations_used > 0
        assert aco.iteration > 0
        assert aco.best_cost <= initial_cost


class TestAlgorithmComparison:
    """Tests comparing algorithms on same problem."""
    
    def test_all_algorithms_improve(self):
        """Each algorithm should improve initial solution."""
        problem = TSPProblem(num_cities=30, seed=42)
        
        algorithms = [
            ('GA', GeneticAlgorithm(problem, population_size=20)),
            ('TS', TabuSearch(problem, neighborhood_size=30)),
            ('SA', SimulatedAnnealing(problem)),
            ('VNS', VariableNeighborhoodSearch(problem, reduced_vns=True)),
            ('ILS', IteratedLocalSearch(problem, local_search_max_iters=10)),
            ('ACO', AntColonyOptimization(problem, num_ants=10, local_search=False))
        ]
        
        for name, algo in algorithms:
            algo.initialize()
            initial = algo.best_cost
            algo.step(500)
            final = algo.best_cost
            
            # Each should maintain or improve
            assert final <= initial, f"{name} got worse!"
    
    def test_algorithm_inject_solution(self):
        """Test that algorithms can receive injected solutions."""
        problem = TSPProblem(num_cities=20, seed=42)
        
        # Generate a good solution
        good_solution = problem.generate_random_solution()
        good_cost = problem.evaluate(good_solution)
        
        algorithms = [
            GeneticAlgorithm(problem),
            TabuSearch(problem),
            SimulatedAnnealing(problem),
            VariableNeighborhoodSearch(problem),
            IteratedLocalSearch(problem),
            AntColonyOptimization(problem, local_search=False)
        ]
        
        for algo in algorithms:
            algo.initialize()
            algo.inject_solution(good_solution, good_cost)
            
            # Best should be at least as good as injected
            assert algo.best_cost <= good_cost, f"{algo.name} failed injection!"


def run_all_tests():
    """Run all algorithm tests."""
    print("=" * 60)
    print("Algorithm Implementation Tests")
    print("=" * 60)
    
    test_classes = [
        TestGeneticAlgorithm,
        TestTabuSearch,
        TestSimulatedAnnealing,
        TestVariableNeighborhoodSearch,
        TestIteratedLocalSearch,
        TestAntColonyOptimization,
        TestAlgorithmComparison
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
