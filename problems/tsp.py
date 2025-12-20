"""
TSP (Traveling Salesman Problem) implementation.

Provides a concrete implementation of BaseProblem for TSP.
"""

from typing import Any, List, Tuple, Optional
import numpy as np

from .base_problem import BaseProblem


class TSPProblem(BaseProblem):
    """
    Traveling Salesman Problem implementation.
    
    Solution representation: Permutation of city indices [0, 1, 2, ..., n-1]
    Objective: Minimize total tour length
    """
    
    def __init__(
        self,
        cities: Optional[np.ndarray] = None,
        num_cities: int = 50,
        distribution: str = 'random',
        seed: Optional[int] = None
    ):
        """
        Initialize TSP problem.
        
        Args:
            cities: Optional array of city coordinates (n, 2)
            num_cities: Number of cities (used if cities not provided)
            distribution: City distribution type ('random', 'clustered', 'grid')
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        if seed is not None:
            np.random.seed(seed)
        
        if cities is not None:
            self.cities = np.array(cities, dtype=np.float64)
        else:
            self.cities = self._generate_cities(num_cities, distribution)
        
        self.num_cities = len(self.cities)
        self.distribution = distribution
        
        # Precompute distance matrix
        self.distance_matrix = self._compute_distance_matrix()
        
        # Compute cost bounds
        self._lower_bound = self._estimate_lower_bound()
        self._upper_bound = self._estimate_upper_bound()
    
    def _generate_cities(self, n: int, distribution: str) -> np.ndarray:
        """Generate city coordinates based on distribution type."""
        if distribution == 'random':
            return np.random.rand(n, 2)
        
        elif distribution == 'clustered':
            # Generate clustered cities
            n_clusters = max(3, n // 10)
            centers = np.random.rand(n_clusters, 2)
            cities = []
            
            for i in range(n):
                center = centers[i % n_clusters]
                # Add noise around cluster center
                city = center + np.random.randn(2) * 0.05
                city = np.clip(city, 0, 1)
                cities.append(city)
            
            return np.array(cities)
        
        elif distribution == 'grid':
            # Generate cities on a grid with noise
            side = int(np.ceil(np.sqrt(n)))
            cities = []
            
            for i in range(n):
                x = (i % side) / (side - 1) if side > 1 else 0.5
                y = (i // side) / (side - 1) if side > 1 else 0.5
                # Add small noise
                x += np.random.randn() * 0.02
                y += np.random.randn() * 0.02
                cities.append([np.clip(x, 0, 1), np.clip(y, 0, 1)])
            
            return np.array(cities)
        
        else:
            return np.random.rand(n, 2)
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute pairwise distance matrix."""
        n = len(self.cities)
        dist = np.zeros((n, n), dtype=np.float64)
        
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.cities[i] - self.cities[j])
                dist[i, j] = d
                dist[j, i] = d
        
        return dist
    
    def _estimate_lower_bound(self) -> float:
        """Estimate lower bound using minimum spanning tree heuristic."""
        # Simple MST-based lower bound
        from scipy.sparse.csgraph import minimum_spanning_tree
        
        mst = minimum_spanning_tree(self.distance_matrix)
        mst_weight = mst.sum()
        
        # Add two smallest edges to form a tour bound
        # This is a simple Held-Karp style bound
        return mst_weight
    
    def _estimate_upper_bound(self) -> float:
        """Estimate upper bound using nearest neighbor heuristic."""
        tour = self._nearest_neighbor_tour()
        return self.evaluate_tour(tour)
    
    def _nearest_neighbor_tour(self, start: int = 0) -> np.ndarray:
        """Generate a tour using nearest neighbor heuristic."""
        n = self.num_cities
        visited = [False] * n
        tour = [start]
        visited[start] = True
        
        for _ in range(n - 1):
            current = tour[-1]
            best_next = -1
            best_dist = float('inf')
            
            for j in range(n):
                if not visited[j] and self.distance_matrix[current, j] < best_dist:
                    best_dist = self.distance_matrix[current, j]
                    best_next = j
            
            tour.append(best_next)
            visited[best_next] = True
        
        return np.array(tour)
    
    @property
    def problem_type(self) -> str:
        return 'TSP'
    
    @property
    def size(self) -> int:
        return self.num_cities
    
    def evaluate(self, solution: Any) -> float:
        """Evaluate tour length."""
        self.evaluation_count += 1
        return self.evaluate_tour(solution)
    
    def evaluate_tour(self, tour: np.ndarray) -> float:
        """Evaluate tour length without incrementing counter."""
        total = 0.0
        n = len(tour)
        
        for i in range(n):
            total += self.distance_matrix[tour[i], tour[(i + 1) % n]]
        
        return total
    
    def generate_random_solution(self) -> np.ndarray:
        """Generate a random tour."""
        tour = np.arange(self.num_cities)
        np.random.shuffle(tour)
        return tour
    
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """Generate a neighbor using 2-opt move."""
        n = len(solution)
        neighbor = solution.copy()
        
        # Random 2-opt: reverse a segment
        i = np.random.randint(0, n - 1)
        j = np.random.randint(i + 1, n)
        
        neighbor[i:j+1] = neighbor[i:j+1][::-1]
        
        return neighbor
    
    def generate_neighbors(self, solution: np.ndarray, k: int) -> List[np.ndarray]:
        """Generate k diverse neighbors using different operators."""
        neighbors = []
        moves = ['2opt', 'swap', 'insert', 'or_opt']
        
        for i in range(k):
            move_type = moves[i % len(moves)]
            
            if move_type == '2opt':
                neighbors.append(self._2opt_move(solution))
            elif move_type == 'swap':
                neighbors.append(self._swap_move(solution))
            elif move_type == 'insert':
                neighbors.append(self._insert_move(solution))
            else:  # or_opt
                neighbors.append(self._or_opt_move(solution))
        
        return neighbors
    
    def _2opt_move(self, tour: np.ndarray) -> np.ndarray:
        """2-opt: reverse a segment."""
        n = len(tour)
        neighbor = tour.copy()
        i = np.random.randint(0, n - 1)
        j = np.random.randint(i + 1, n)
        neighbor[i:j+1] = neighbor[i:j+1][::-1]
        return neighbor
    
    def _swap_move(self, tour: np.ndarray) -> np.ndarray:
        """Swap two cities."""
        n = len(tour)
        neighbor = tour.copy()
        i, j = np.random.choice(n, 2, replace=False)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor
    
    def _insert_move(self, tour: np.ndarray) -> np.ndarray:
        """Remove a city and insert it elsewhere."""
        n = len(tour)
        neighbor = list(tour)
        i = np.random.randint(0, n)
        city = neighbor.pop(i)
        j = np.random.randint(0, n - 1)
        neighbor.insert(j, city)
        return np.array(neighbor)
    
    def _or_opt_move(self, tour: np.ndarray) -> np.ndarray:
        """Or-opt: move a segment of 1-3 cities."""
        n = len(tour)
        neighbor = list(tour)
        segment_len = np.random.randint(1, min(4, n))
        start = np.random.randint(0, n - segment_len + 1)
        
        # Extract segment
        segment = neighbor[start:start + segment_len]
        del neighbor[start:start + segment_len]
        
        # Insert elsewhere
        insert_pos = np.random.randint(0, len(neighbor) + 1)
        neighbor[insert_pos:insert_pos] = segment
        
        return np.array(neighbor)
    
    def solution_distance(self, sol1: np.ndarray, sol2: np.ndarray) -> float:
        """
        Compute normalized edge-based distance between two tours.
        
        Distance = 1 - (common edges / total edges)
        """
        n = len(sol1)
        
        # Get edges in each tour
        edges1 = set()
        edges2 = set()
        
        for i in range(n):
            # Use sorted tuple so (a,b) == (b,a)
            e1 = tuple(sorted([sol1[i], sol1[(i + 1) % n]]))
            e2 = tuple(sorted([sol2[i], sol2[(i + 1) % n]]))
            edges1.add(e1)
            edges2.add(e2)
        
        # Count common edges
        common = len(edges1 & edges2)
        
        # Normalized distance
        return 1.0 - (common / n)
    
    def get_cost_bounds(self) -> Tuple[float, float]:
        """Return estimated lower and upper bounds."""
        return self._lower_bound, self._upper_bound
    
    def is_feasible(self, solution: np.ndarray) -> bool:
        """Check if solution is a valid permutation."""
        if len(solution) != self.num_cities:
            return False
        return set(solution) == set(range(self.num_cities))
    
    def copy_solution(self, solution: np.ndarray) -> np.ndarray:
        """Copy a tour."""
        return solution.copy()
