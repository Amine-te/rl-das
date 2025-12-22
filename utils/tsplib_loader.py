"""
TSPLIB file parser, loader, and downloader.

Parses standard TSPLIB format files (.tsp) to create TSPProblem instances.
Supports both EUC_2D (Euclidean 2D) and explicit distance matrices.
Includes data augmentation logic (rotations/flips).

Usage:
    # As a library
    from utils.tsplib_loader import load_tsplib
    problem = load_tsplib("data/tsplib/berlin52.tsp")

    # As a script (Download data)
    python utils/tsplib_loader.py
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import argparse

# Add project root to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from problems.tsp import TSPProblem


def parse_tsplib_file(filepath: str) -> Dict:
    """
    Parse a TSPLIB format file.
    
    Returns a dictionary with:
        - name: Instance name
        - dimension: Number of cities
        - edge_weight_type: Type of distance calculation
        - node_coords: Array of (x, y) coordinates (if available)
        - edge_weight_section: Explicit distance matrix (if available)
    """
    result = {
        'name': None,
        'dimension': 0,
        'edge_weight_type': 'EUC_2D',
        'node_coords': None,
        'edge_weight_section': None,
        'comment': None,
        'optimal': None
    }
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Parse header fields
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().upper()
            value = value.strip()
            
            if key == 'NAME':
                result['name'] = value
            elif key == 'DIMENSION':
                result['dimension'] = int(value)
            elif key == 'EDGE_WEIGHT_TYPE':
                result['edge_weight_type'] = value.upper()
            elif key == 'COMMENT':
                result['comment'] = value
                # Try to extract optimal value from comment
                opt_match = re.search(r'[Oo]ptimal[:\s]+(\d+\.?\d*)', value)
                if opt_match:
                    result['optimal'] = float(opt_match.group(1))
        
        # Parse node coordinates section
        elif line.upper() == 'NODE_COORD_SECTION':
            i += 1
            coords = []
            while i < len(lines):
                coord_line = lines[i].strip()
                if coord_line.upper() in ['EOF', 'EDGE_WEIGHT_SECTION', 'DISPLAY_DATA_SECTION']:
                    break
                if not coord_line:
                    i += 1
                    continue
                    
                parts = coord_line.split()
                if len(parts) >= 3:
                    # Format: node_id x y
                    x, y = float(parts[1]), float(parts[2])
                    coords.append([x, y])
                i += 1
            
            result['node_coords'] = np.array(coords)
            continue
        
        # Parse edge weight section (explicit distance matrix)
        elif line.upper() == 'EDGE_WEIGHT_SECTION':
            i += 1
            weights = []
            while i < len(lines):
                weight_line = lines[i].strip()
                if weight_line.upper() in ['EOF', 'DISPLAY_DATA_SECTION']:
                    break
                if not weight_line:
                    i += 1
                    continue
                
                parts = weight_line.split()
                for p in parts:
                    try:
                        weights.append(float(p))
                    except ValueError:
                        break
                i += 1
            
            result['edge_weight_section'] = weights
            continue
        
        elif line.upper() == 'EOF':
            break
        
        i += 1
    
    return result


def load_tsplib(filepath: str) -> TSPProblem:
    """Load a TSPLIB file and create a TSPProblem instance."""
    data = parse_tsplib_file(filepath)
    
    if data['node_coords'] is not None:
        # Normalize coordinates to [0, 1] range for consistency
        coords = data['node_coords']
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        range_coords = max_coords - min_coords
        
        # Avoid division by zero
        range_coords[range_coords == 0] = 1
        
        normalized = (coords - min_coords) / range_coords
        
        problem = TSPProblem(cities=normalized)
        problem.name = data['name'] or Path(filepath).stem
        problem.original_coords = coords  # Keep original for reference
        problem.optimal = data['optimal']
        problem.tsplib_file = filepath
        
        return problem
    
    elif data['edge_weight_section'] is not None:
        # Create problem from explicit distance matrix
        n = data['dimension']
        weights = data['edge_weight_section']
        
        # Build distance matrix based on format
        # Common formats: LOWER_DIAG_ROW, UPPER_DIAG_ROW, FULL_MATRIX
        dist_matrix = np.zeros((n, n))
        
        # Try to infer format - most common is LOWER_DIAG_ROW
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                if idx < len(weights):
                    dist_matrix[i, j] = weights[idx]
                    dist_matrix[j, i] = weights[idx]
                    idx += 1
        
        # Generate fake coordinates for visualization (using MDS-like approach)
        # For simplicity, just use random positions
        np.random.seed(42)
        fake_coords = np.random.rand(n, 2)
        
        problem = TSPProblem(cities=fake_coords)
        problem.distance_matrix = dist_matrix  # Override with real distances
        problem.name = data['name'] or Path(filepath).stem
        problem.optimal = data['optimal']
        problem.tsplib_file = filepath
        
        return problem
    
    else:
        raise ValueError(f"Could not parse TSPLIB file: {filepath}")


def list_available_instances(data_dir: str) -> List[str]:
    """List all available .tsp files in a directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    
    return sorted([str(p) for p in data_path.glob("*.tsp")])


def augment_problem(problem: TSPProblem) -> List[TSPProblem]:
    """
    Generate augmented versions of a TSP problem using symmetries (D4 group).
    
    Generates 8 variations:
    - Original
    - 90 deg rotation
    - 180 deg rotation
    - 270 deg rotation
    - Horizontal flip
    - Vertical flip
    - Diagonal flip 1 (transpose)
    - Diagonal flip 2 (anti-transpose)
    """
    if not hasattr(problem, 'cities') or problem.cities is None:
        return [problem]
    
    variants = []
    coords = problem.cities.copy()
    
    # Define transformations
    # 1. Identity
    variants.append(problem)
    
    # 2. Rotate 90 deg: (x, y) -> (-y, x) -> normalize
    # 3. Rotate 180 deg: (x, y) -> (-x, -y)
    # 4. Rotate 270 deg: (x, y) -> (y, -x)
    # 5. Flip X: (x, y) -> (-x, y)
    # 6. Flip Y: (x, y) -> (x, -y)
    # 7. Transpose: (x, y) -> (y, x)
    # 8. Anti-transpose: (x, y) -> (-y, -x)
    
    # Helper to create normalized variant
    def create_variant(transformed_coords, suffix):
        # Re-normalize to [0, 1]
        min_c = transformed_coords.min(axis=0)
        max_c = transformed_coords.max(axis=0)
        rng = max_c - min_c
        rng[rng == 0] = 1.0
        norm_coords = (transformed_coords - min_c) / rng
        
        new_prob = TSPProblem(cities=norm_coords)
        new_prob.name = f"{problem.name}_{suffix}"
        if hasattr(problem, 'optimal') and problem.optimal:
            new_prob.optimal = problem.optimal
        return new_prob

    # Generate variants
    # Rotate 90
    r90 = np.column_stack((-coords[:, 1], coords[:, 0]))
    variants.append(create_variant(r90, "r90"))
    
    # Rotate 180
    r180 = np.column_stack((-coords[:, 0], -coords[:, 1]))
    variants.append(create_variant(r180, "r180"))
    
    # Rotate 270
    r270 = np.column_stack((coords[:, 1], -coords[:, 0]))
    variants.append(create_variant(r270, "r270"))
    
    # Flip X
    fx = np.column_stack((-coords[:, 0], coords[:, 1]))
    variants.append(create_variant(fx, "fx"))
    
    # Flip Y
    fy = np.column_stack((coords[:, 0], -coords[:, 1]))
    variants.append(create_variant(fy, "fy"))
    
    # Transpose
    ft = np.column_stack((coords[:, 1], coords[:, 0]))
    variants.append(create_variant(ft, "ft"))
    
    # Anti-transpose
    fat = np.column_stack((-coords[:, 1], -coords[:, 0]))
    variants.append(create_variant(fat, "fat"))
    
    return variants


def load_all_instances(data_dir: str, max_cities: Optional[int] = None, augment: bool = True) -> List[TSPProblem]:
    """Load all TSPLIB instances from a directory."""
    instances = []
    
    for filepath in list_available_instances(data_dir):
        try:
            problem = load_tsplib(filepath)
            if max_cities is None or problem.num_cities <= max_cities:
                if augment and problem.cities is not None:
                    # Augment data 8x
                    variants = augment_problem(problem)
                    instances.extend(variants)
                    print(f"  Loaded: {problem.name} ({problem.num_cities} cities) + {len(variants)-1} variants")
                else:
                    instances.append(problem)
                    print(f"  Loaded: {problem.name} ({problem.num_cities} cities)")
        except Exception as e:
            print(f"  Warning: Could not load {filepath}: {e}")
    
    return instances


def download_tsplib_instances(output_dir: str = "data/tsplib") -> None:
    """
    Download common TSPLIB instances.
    """
    import urllib.request
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Common instances
    instances = [
        ('eil51.tsp', 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/eil51.tsp.gz'),
        ('berlin52.tsp', 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/berlin52.tsp.gz'),
        ('st70.tsp', 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/st70.tsp.gz'),
        ('eil76.tsp', 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/eil76.tsp.gz'),
        ('pr76.tsp', 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/pr76.tsp.gz'),
        ('kroA100.tsp', 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/kroA100.tsp.gz'),
        ('kroB100.tsp', 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/kroB100.tsp.gz'),
        ('kroC100.tsp', 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/kroC100.tsp.gz'),
        ('kroD100.tsp', 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/kroD100.tsp.gz'),
        ('kroE100.tsp', 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/kroE100.tsp.gz'),
        ('rd100.tsp', 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/rd100.tsp.gz'),
        ('ch130.tsp', 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ch130.tsp.gz'),
        ('ch150.tsp', 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ch150.tsp.gz'),
    ]
    
    print(f"Downloading {len(instances)} TSPLIB instances to {output_dir}...")
    
    for name, url in instances:
        output_path = os.path.join(output_dir, name)
        if os.path.exists(output_path):
            print(f"  ✓ {name} (already exists)")
            continue
        
        try:
            print(f"  Downloading {name}...")
            # Download gzipped file
            gz_path = output_path + '.gz'
            urllib.request.urlretrieve(url, gz_path)
            
            # Decompress
            import gzip
            with gzip.open(gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            os.remove(gz_path)
            print(f"    ✓ Downloaded {name}")
            
        except Exception as e:
            print(f"    ✗ Failed to download {name}: {e}")
            try:
                if os.path.exists(gz_path): os.remove(gz_path)
                if os.path.exists(output_path): os.remove(output_path)
            except: pass
    
    print("Download complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download TSPLIB instances')
    parser.add_argument('--output-dir', type=str, default='data/tsplib', help='Output directory')
    args = parser.parse_args()
    
    download_tsplib_instances(args.output_dir)
