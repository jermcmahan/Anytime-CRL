"""
Curriculum Learning Logic.

Adapts the finite benchmark logic from src.instances into infinite
random streams for Reinforcement Learning training.

Includes:
1. Stream Generators (Infinite random sampling of Easy/Hard/Correlated)
2. RoundRobinGenerator (Mixes different streams)
"""

from typing import Callable, Tuple, List
from pathlib import Path
import glob

import numpy as np

# Type alias for a generator function expected by KnapsackEnv
# Args: rng (Random Generator)
# Returns: (values, weights, budget)
EnvGeneratorFn = Callable[[np.random.Generator], Tuple[np.ndarray, np.ndarray, int]]


# ==========================================
# 1. Infinite Stream Adapters
# ==========================================

def make_uncorrelated_stream(min_n: int, max_n: int, max_value: int = 1000) -> EnvGeneratorFn:
    """
    Generates Uncorrelated instances with random N and random Budget.
    """
    def _gen(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, int]:
        n = rng.integers(min_n, max_n + 1)
        
        # 1. Generate Arrays (Same math as instances.py)
        values = rng.integers(1, max_value, size=n)
        weights = rng.integers(1, max_value, size=n)
        
        # 2. Random Budget (Continuous sampling instead of fixed steps)
        # Sample a capacity ratio between 0.1 and 0.9
        capacity_ratio = rng.uniform(0.1, 0.9)
        budget = int(np.sum(weights) * capacity_ratio)
        
        return values, weights, max(1, budget)
    return _gen


def make_correlated_stream(min_n: int, max_n: int, max_value: int = 1000) -> EnvGeneratorFn:
    """
    Generates Strongly Correlated instances with random N and random Budget.
    """
    def _gen(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, int]:
        n = rng.integers(min_n, max_n + 1)
        
        # 1. Generate Weights
        weights = rng.integers(1, max_value, size=n)
        
        # 2. Strong Correlation Logic (V = W + Noise)
        # Matches 'strong' logic from instances.py
        values = (weights + max_value / 10.0).astype(np.int64)
        
        # 3. Random Budget
        capacity_ratio = rng.uniform(0.1, 0.9)
        budget = int(np.sum(weights) * capacity_ratio)
        
        return values, weights, max(1, budget)
    return _gen


def make_hard_stream(min_n: int, max_n: int) -> EnvGeneratorFn:
    """
    Generates Subset Sum (Powers of Two) instances.
    Randomly selects one of the valid 'powers of two' budgets.
    """
    def _gen(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, int]:
        n = rng.integers(min_n, max_n + 1)
        
        # 1. Powers of Two Logic (Same as instances.py)
        n_powers = min(n, 50)
        exponents = np.arange(n_powers, dtype=np.int64)
        powers = 2 ** exponents
        
        # 2. Fillers
        n_remaining = n - n_powers
        
        # 3. Randomly select ONE valid budget from the powers
        # (instances.py iterates all of them; here we pick one)
        # We ignore the very small powers as they are trivial
        valid_indices = np.arange(max(0, n_powers - 20), n_powers)
        if len(valid_indices) == 0: valid_indices = np.arange(n_powers)
        
        budget_idx = rng.choice(valid_indices)
        budget = powers[budget_idx]
        
        if n_remaining > 0:
            filler = rng.integers(1, budget, size=n_remaining, dtype=np.int64)
            values = np.concatenate((powers, filler))
            weights = np.concatenate((powers, filler))
        else:
            values = powers
            weights = powers
            
        return values, weights, int(budget)
    return _gen


# ==========================================
# 2. Round Robin Controller
# ==========================================

def make_round_robin_generator(generators: List[EnvGeneratorFn]) -> EnvGeneratorFn:
    """
    Returns a function that cycles through the provided generators.
    
    Example:
        [GenA, GenB] -> yields A, then B, then A...
    """
    cur_idx = 0
    num_gens = len(generators)
    
    def _gen(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, int]:
        nonlocal cur_idx
        
        # 1. Select Strategy
        gen_fn = generators[cur_idx]
        
        # 2. Advance (Round Robin)
        cur_idx = (cur_idx + 1) % num_gens
        
        # 3. Execute
        return gen_fn(rng)
        
    return _gen

def make_fixed_dataset_generator(data_dir: str) -> EnvGeneratorFn:
    """
    Loads a fixed set of .npz files from disk (Recursively).
    Returns raw Integers to match the behavior of random stream generators.
    """
    search_path = Path(data_dir)
    
    # 1. Recursive Loading (Matches your hierarchical folder structure)
    # Using sorted() ensures the order is deterministic (Reproducibility)
    files = sorted(list(search_path.rglob("*.npz")))
    
    if not files:
        raise ValueError(f"No .npz files found in {data_dir}")
    
    print(f"Fixed Generator loaded {len(files)} instances from {data_dir}")
    
    # State to track iteration
    cur_idx = 0
    
    def _gen(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, int]:
        nonlocal cur_idx
        
        # 2. Sequential Access
        # We loop through files 0, 1, 2... and wrap around if needed.
        file_path = files[cur_idx % len(files)]
        cur_idx += 1
        
        # 3. Load Data
        # allow_pickle=True is required for some numpy versions/configs
        try:
            data = np.load(file_path, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}: {e}")
        
        # 4. Return Raw Integers (Consistency Fix)
        # We assume the .npz was saved as integers (which generate_datasets.py does).
        # We explicitly cast budget to python int, as np.int64 scalar can behave oddly in some loops.
        return (
            data['values'], 
            data['weights'], 
            int(data['budget'])
        )
        
    return _gen
