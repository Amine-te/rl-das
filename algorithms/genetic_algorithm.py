"""
Genetic Algorithm (GA) for discrete optimization.

A population-based evolutionary algorithm that uses:
- Selection (tournament)
- Crossover (order crossover for permutations)
- Mutation (swap, insert, invert)

Good for: Exploration, escaping local optima, diverse search
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from .base_algorithm import BaseAlgorithm


class GeneticAlgorithm(BaseAlgorithm):
    """
    Genetic Algorithm for discrete optimization.
    
    Uses order-based crossover suitable for permutation problems (TSP, VRP).
    Implements adaptive mutation rates based on population diversity.
    """
    
    def __init__(
        self,
        problem: Any,
        population_size: int = 50,
        tournament_size: int = 3,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        elite_size: int = 2
    ):
        """
        Initialize Genetic Algorithm.
        
        Args:
            problem: Problem instance to solve
            population_size: Number of individuals in population
            tournament_size: Tournament selection size
            crossover_rate: Probability of crossover
            mutation_rate: Base probability of mutation
            elite_size: Number of elite individuals to preserve
        """
        super().__init__(problem, 'GA')
        
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        # Population state
        self.population: List[Tuple[Any, float]] = []
        self.generation = 0
        self.stagnation_counter = 0
        self.last_best_cost = float('inf')
        
        # Adaptive parameters
        self.adaptive_mutation_rate = mutation_rate
    
    def initialize(self, **kwargs) -> None:
        """Initialize population with random solutions."""
        self.population = []
        self.generation = 0
        self.stagnation_counter = 0
        self.best_solution = None
        self.best_cost = float('inf')
        self.evaluations_used = 0
        self.adaptive_mutation_rate = self.mutation_rate
        
        # Generate initial population
        for _ in range(self.population_size):
            solution = self.problem.generate_random_solution()
            cost = self.problem.evaluate(solution)
            self.evaluations_used += 1
            self.population.append((solution.copy(), cost))
            self.update_best(solution, cost)
        
        # Sort population by fitness
        self.population.sort(key=lambda x: x[1])
        self.last_best_cost = self.best_cost
    
    def step(self, num_evaluations: int) -> Tuple[Any, float]:
        """
        Run GA for specified number of evaluations.
        
        Args:
            num_evaluations: Budget of function evaluations
            
        Returns:
            Tuple of (best_solution, best_cost)
        """
        evals_done = 0
        
        while evals_done < num_evaluations:
            # Check if we can do a full generation
            offspring_needed = self.population_size - self.elite_size
            evals_for_generation = offspring_needed
            
            if evals_done + evals_for_generation > num_evaluations:
                # Do partial generation with remaining budget
                offspring_needed = min(offspring_needed, num_evaluations - evals_done)
                if offspring_needed <= 0:
                    break
            
            # Create offspring
            offspring = []
            while len(offspring) < offspring_needed:
                # Selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child = self._order_crossover(parent1[0], parent2[0])
                else:
                    child = parent1[0].copy()
                
                # Mutation
                if np.random.random() < self.adaptive_mutation_rate:
                    child = self._mutate(child)
                
                # Evaluate
                cost = self.problem.evaluate(child)
                evals_done += 1
                self.evaluations_used += 1
                
                offspring.append((child, cost))
                self.update_best(child, cost)
                
                if evals_done >= num_evaluations:
                    break
            
            # Elitism: keep best individuals
            self.population.sort(key=lambda x: x[1])
            elite = self.population[:self.elite_size]
            
            # Combine elite with offspring
            self.population = elite + offspring
            self.population.sort(key=lambda x: x[1])
            self.population = self.population[:self.population_size]
            
            self.generation += 1
            
            # Track stagnation for adaptive mutation
            if self.best_cost < self.last_best_cost:
                self.stagnation_counter = 0
                self.last_best_cost = self.best_cost
            else:
                self.stagnation_counter += 1
            
            # Adaptive mutation: increase if stagnating
            if self.stagnation_counter > 5:
                self.adaptive_mutation_rate = min(0.5, self.mutation_rate * 2)
            else:
                self.adaptive_mutation_rate = self.mutation_rate
        
        return self.best_solution, self.best_cost
    
    def _tournament_selection(self) -> Tuple[Any, float]:
        """Select individual using tournament selection."""
        tournament = np.random.choice(
            len(self.population), 
            size=min(self.tournament_size, len(self.population)),
            replace=False
        )
        best_idx = min(tournament, key=lambda i: self.population[i][1])
        return self.population[best_idx]
    
    def _order_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Order Crossover (OX) for permutation representations.
        
        Preserves relative order of elements from both parents.
        """
        size = len(parent1)
        
        # Select crossover points
        start, end = sorted(np.random.choice(size, 2, replace=False))
        
        # Initialize child with segment from parent1
        child = np.full(size, -1, dtype=parent1.dtype)
        child[start:end+1] = parent1[start:end+1]
        
        # Fill remaining positions with elements from parent2 in order
        segment_elements = set(child[start:end+1])
        remaining = [x for x in parent2 if x not in segment_elements]
        
        pos = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = remaining[pos]
                pos += 1
        
        return child
    
    def _mutate(self, solution: np.ndarray) -> np.ndarray:
        """Apply random mutation operator."""
        mutant = solution.copy()
        
        mutation_type = np.random.choice(['swap', 'insert', 'invert', '2opt'])
        
        if mutation_type == 'swap':
            # Swap two random positions
            i, j = np.random.choice(len(mutant), 2, replace=False)
            mutant[i], mutant[j] = mutant[j], mutant[i]
            
        elif mutation_type == 'insert':
            # Remove element and insert elsewhere
            i = np.random.randint(len(mutant))
            j = np.random.randint(len(mutant))
            element = mutant[i]
            mutant = np.delete(mutant, i)
            mutant = np.insert(mutant, j if j < i else j, element)
            
        elif mutation_type == 'invert':
            # Invert a segment
            i, j = sorted(np.random.choice(len(mutant), 2, replace=False))
            mutant[i:j+1] = mutant[i:j+1][::-1]
            
        else:  # 2opt
            # 2-opt move
            i = np.random.randint(0, len(mutant) - 1)
            j = np.random.randint(i + 1, len(mutant))
            mutant[i:j+1] = mutant[i:j+1][::-1]
        
        return mutant
    
    def get_state(self) -> Dict[str, Any]:
        """Get algorithm state for context preservation."""
        return {
            'population': [(s.copy(), c) for s, c in self.population],
            'generation': self.generation,
            'stagnation_counter': self.stagnation_counter,
            'last_best_cost': self.last_best_cost,
            'adaptive_mutation_rate': self.adaptive_mutation_rate,
            'best_solution': self.best_solution.copy() if self.best_solution is not None else None,
            'best_cost': self.best_cost,
            'evaluations_used': self.evaluations_used
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore algorithm state from context."""
        self.population = [(s.copy(), c) for s, c in state['population']]
        self.generation = state['generation']
        self.stagnation_counter = state['stagnation_counter']
        self.last_best_cost = state['last_best_cost']
        self.adaptive_mutation_rate = state['adaptive_mutation_rate']
        self.best_solution = state['best_solution'].copy() if state['best_solution'] is not None else None
        self.best_cost = state['best_cost']
        self.evaluations_used = state['evaluations_used']
    
    def inject_solution(self, solution: Any, cost: float) -> None:
        """Inject a solution into the population."""
        self.update_best(solution, cost)
        
        # Replace worst individual if solution is better
        if self.population and cost < self.population[-1][1]:
            self.population[-1] = (solution.copy(), cost)
            self.population.sort(key=lambda x: x[1])
    
    def get_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        distances = []
        sample_size = min(10, len(self.population))
        indices = np.random.choice(len(self.population), sample_size, replace=False)
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                dist = self.problem.solution_distance(
                    self.population[indices[i]][0],
                    self.population[indices[j]][0]
                )
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
