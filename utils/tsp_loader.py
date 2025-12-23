"""
TSPLIB instance loader utility.

Loads TSP instances from TSPLIB format files and converts them to TSPProblem objects.
Supports various TSP formats and optional data augmentation.
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import sys

# Import TSPProblem - adjust path as needed
sys.path.insert(0, str(Path(__file__).parent.parent))
from problems import TSPProblem


def parse_tsplib_file(filepath: str) -> Tuple[str, int, np.ndarray, Optional[float]]:
    """
    Parse a TSPLIB format file.
    
    Args:
        filepath: Path to the .tsp file
        
    Returns:
        Tuple of (name, dimension, coordinates, optimal_cost)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    name = None
    dimension = None
    edge_weight_type = None
    optimal_cost = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('NAME'):
            name = line.split(':')[1].strip()
        elif line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
        elif line.startswith('EDGE_WEIGHT_TYPE'):
            edge_weight_type = line.split(':')[1].strip()
        elif line.startswith('COMMENT') and 'Optimal' in line:
            # Try to extract optimal cost from comment
            match = re.search(r'(\d+\.?\d*)', line)
            if match:
                optimal_cost = float(match.group(1))
        elif line.startswith('NODE_COORD_SECTION'):
            break
        
        i += 1
    
    # Parse coordinates
    if dimension is None:
        raise ValueError(f"Could not find DIMENSION in {filepath}")
    
    coordinates = np.zeros((dimension, 2))
    i += 1  # Move past NODE_COORD_SECTION line
    
    coord_idx = 0
    while i < len(lines) and coord_idx < dimension:
        line = lines[i].strip()
        if line == 'EOF' or line.startswith('EOF'):
            break
        if line and not line.startswith('DISPLAY_DATA_SECTION'):
            parts = line.split()
            if len(parts) >= 3:
                # Format: node_id x y
                coordinates[coord_idx, 0] = float(parts[1])
                coordinates[coord_idx, 1] = float(parts[2])
                coord_idx += 1
        i += 1
    
    if coord_idx != dimension:
        raise ValueError(f"Expected {dimension} coordinates but found {coord_idx} in {filepath}")
    
    return name, dimension, coordinates, optimal_cost


def load_tsp_instance(filepath: str, normalize: bool = True) -> TSPProblem:
    """
    Load a single TSPLIB instance.
    
    Args:
        filepath: Path to .tsp file
        normalize: Whether to normalize coordinates to [0, 1]
        
    Returns:
        TSPProblem instance
    """
    name, dimension, coordinates, optimal_cost = parse_tsplib_file(filepath)
    
    # Normalize coordinates if requested
    if normalize:
        min_vals = coordinates.min(axis=0)
        max_vals = coordinates.max(axis=0)
        ranges = max_vals - min_vals
        # Avoid division by zero
        ranges[ranges == 0] = 1.0
        coordinates = (coordinates - min_vals) / ranges
    
    # Create TSPProblem with explicit coordinates
    problem = TSPProblem(num_cities=dimension, distribution='custom')
    problem.coordinates = coordinates
    problem.name = name
    
    # Store optimal cost if available
    if optimal_cost is not None:
        problem.optimal_cost = optimal_cost
    
    return problem


def list_available_instances(directory: str, pattern: str = '*.tsp') -> List[str]:
    """
    List all TSPLIB instance files in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        
    Returns:
        List of file paths
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    tsp_files = sorted(directory.glob(pattern))
    return [str(f) for f in tsp_files]


def load_all_instances(directory: str, 
                       pattern: str = '*.tsp',
                       normalize: bool = True,
                       augment: bool = False,
                       augment_count: int = 5,
                       max_cities: int = None) -> List[TSPProblem]:
    """
    Load all TSPLIB instances from a directory.
    
    Args:
        directory: Directory containing .tsp files
        pattern: File pattern to match (default: '*.tsp')
        normalize: Whether to normalize coordinates
        augment: Whether to create augmented versions (rotations, reflections)
        augment_count: Number of augmented versions per instance if augment=True
        max_cities: Maximum number of cities to include (filters out larger instances)
        
    Returns:
        List of TSPProblem instances
    """
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory not found: {directory}")
    
    # Find all matching files
    tsp_files = sorted(directory.glob(pattern))
    
    if not tsp_files:
        raise ValueError(f"No files matching '{pattern}' found in {directory}")
    
    instances = []
    
    for filepath in tsp_files:
        try:
            problem = load_tsp_instance(str(filepath), normalize=normalize)
            
            # Filter by size if specified
            if max_cities is not None and problem.size > max_cities:
                continue
            
            instances.append(problem)
            
            # Create augmented versions if requested
            if augment:
                for _ in range(augment_count):
                    augmented = augment_instance(problem)
                    instances.append(augmented)
                    
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
            continue
    
    if instances:
        print(f"Loaded {len(instances)} instances from {directory}")
    
    return instances


def augment_instance(problem: TSPProblem) -> TSPProblem:
    """
    Create an augmented version of a TSP instance through random transformations.
    
    Applies random rotation, reflection, and translation to create a new instance
    that is equivalent in structure but visually different.
    
    Args:
        problem: Original TSPProblem instance
        
    Returns:
        New TSPProblem with augmented coordinates
    """
    # Create new problem with same structure
    aug_problem = TSPProblem(num_cities=problem.size, distribution='custom')
    aug_problem.name = f"{problem.name}_aug_{np.random.randint(10000)}"
    
    # Copy coordinates
    coords = problem.coordinates.copy()
    
    # Random rotation
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    coords = coords @ rotation_matrix.T
    
    # Random reflection (50% chance)
    if np.random.rand() < 0.5:
        coords[:, 0] *= -1
    if np.random.rand() < 0.5:
        coords[:, 1] *= -1
    
    # Random translation and scaling to fit [0, 1]
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    coords = (coords - min_vals) / ranges
    
    # Add small noise (optional, keeps structure similar)
    noise_level = 0.01
    coords += np.random.normal(0, noise_level, coords.shape)
    coords = np.clip(coords, 0, 1)
    
    aug_problem.coordinates = coords
    
    # Copy optimal cost if available
    if hasattr(problem, 'optimal_cost'):
        aug_problem.optimal_cost = problem.optimal_cost
    
    return aug_problem


def get_tsplib_optimal(name: str) -> Optional[float]:
    """
    Get known optimal tour length for common TSPLIB instances.
    
    Args:
        name: Instance name (e.g., 'eil51', 'berlin52')
        
    Returns:
        Optimal tour length if known, None otherwise
    """
    # Dictionary of known optimal solutions for common TSPLIB instances
    optimal_values = {
        'eil51': 426,
        'berlin52': 7542,
        'st70': 675,
        'eil76': 538,
        'pr76': 108159,
        'rat99': 1211,
        'kroA100': 21282,
        'kroB100': 22141,
        'kroC100': 20749,
        'kroD100': 21294,
        'kroE100': 22068,
        'rd100': 7910,
        'eil101': 629,
        'lin105': 14379,
        'pr107': 44303,
        'pr124': 59030,
        'bier127': 118282,
        'ch130': 6110,
        'pr136': 96772,
        'pr144': 58537,
        'ch150': 6528,
        'kroA150': 26524,
        'kroB150': 26130,
        'pr152': 73682,
        'u159': 42080,
        'rat195': 2323,
        'kroA200': 29368,
        'kroB200': 29437,
        'ts225': 126643,
        'tsp225': 3916,
        'pr226': 80369,
        'gil262': 2378,
        'pr264': 49135,
        'a280': 2579,
        'pr299': 48191,
        'lin318': 42029,
        'rd400': 15281,
        'fl417': 11861,
        'pr439': 107217,
        'pcb442': 50778,
        'd493': 35002,
        'u574': 36905,
        'rat575': 6773,
        'p654': 34643,
        'd657': 48912,
        'u724': 41910,
        'rat783': 8806,
        'pr1002': 259045,
        'u1060': 224094,
        'vm1084': 239297,
        'pcb1173': 56892,
        'd1291': 50801,
        'rl1304': 252948,
        'rl1323': 270199,
        'nrw1379': 56638,
        'fl1400': 20127,
        'u1432': 152970,
        'fl1577': 22249,
        'd1655': 62128,
        'vm1748': 336556,
        'u1817': 57201,
        'rl1889': 316536,
        'pr2392': 378032,
        'pcb3038': 137694,
        'fl3795': 28772,
        'fnl4461': 182566,
        'rl5915': 565530,
        'rl5934': 556045,
    }
    
    # Try exact match first
    if name in optimal_values:
        return optimal_values[name]
    
    # Try without extensions
    base_name = name.split('.')[0].lower()
    if base_name in optimal_values:
        return optimal_values[base_name]
    
    return None


def calculate_tour_length(problem: TSPProblem, tour: List[int]) -> float:
    """
    Calculate the total length of a tour.
    
    Args:
        problem: TSPProblem instance
        tour: List of city indices representing the tour
        
    Returns:
        Total tour length
    """
    return problem.evaluate(tour)


# Utility function to create a sample TSPLIB format file for testing
def create_sample_tsp_file(filepath: str, num_cities: int = 20):
    """
    Create a sample TSPLIB format file for testing.
    
    Args:
        filepath: Output file path
        num_cities: Number of cities
    """
    np.random.seed(42)
    coords = np.random.rand(num_cities, 2) * 100
    
    with open(filepath, 'w') as f:
        f.write(f"NAME : sample_{num_cities}\n")
        f.write("COMMENT : Sample TSP instance\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {num_cities}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        
        for i, (x, y) in enumerate(coords, 1):
            f.write(f"{i} {x:.6f} {y:.6f}\n")
        
        f.write("EOF\n")
    
    print(f"Created sample TSP file: {filepath}")


if __name__ == '__main__':
    # Test the loader
    import sys
    
    if len(sys.argv) > 1:
        # Load from directory
        directory = sys.argv[1]
        instances = load_all_instances(directory, normalize=True, augment=False)
        
        print(f"\nLoaded {len(instances)} instances:")
        for problem in instances:
            print(f"  {problem.name}: {problem.size} cities")
            if hasattr(problem, 'optimal_cost'):
                print(f"    Optimal cost: {problem.optimal_cost}")
    else:
        # Create a sample file for testing
        create_sample_tsp_file('sample.tsp', num_cities=20)
        problem = load_tsp_instance('sample.tsp')
        print(f"Loaded sample instance: {problem.name} with {problem.size} cities")