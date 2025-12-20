"""
Algorithms module for RL-DAS.

Contains optimization algorithm implementations:
- BaseAlgorithm: Abstract base class for all algorithms
- GeneticAlgorithm (GA): Population-based evolutionary algorithm
- TabuSearch (TS): Memory-based local search
- SimulatedAnnealing (SA): Probabilistic local search
- VariableNeighborhoodSearch (VNS): Systematic neighborhood changes
- IteratedLocalSearch (ILS): Perturbation + local search
- AntColonyOptimization (ACO): Swarm intelligence for path problems

Recommended combinations for training:
- 3 algorithms: GA, TS, SA (exploration, exploitation, balance)
- 4 algorithms: GA, TS, SA, ILS (adds controlled perturbation)
- 6 algorithms: All (comprehensive but computationally expensive)
"""

from .base_algorithm import BaseAlgorithm
from .genetic_algorithm import GeneticAlgorithm
from .tabu_search import TabuSearch
from .simulated_annealing import SimulatedAnnealing
from .variable_neighborhood_search import VariableNeighborhoodSearch
from .iterated_local_search import IteratedLocalSearch
from .ant_colony import AntColonyOptimization

__all__ = [
    'BaseAlgorithm',
    'GeneticAlgorithm',
    'TabuSearch',
    'SimulatedAnnealing',
    'VariableNeighborhoodSearch',
    'IteratedLocalSearch',
    'AntColonyOptimization',
    # Aliases
    'GA',
    'TS',
    'SA',
    'VNS',
    'ILS',
    'ACO'
]

# Convenient aliases
GA = GeneticAlgorithm
TS = TabuSearch
SA = SimulatedAnnealing
VNS = VariableNeighborhoodSearch
ILS = IteratedLocalSearch
ACO = AntColonyOptimization
