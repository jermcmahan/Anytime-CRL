"""
Data generation logic for the Knapsack Problem.

Some of these datasets are inspired by those introduced in the paper 
"Where are the hard knapsack problems?", which are often used for benchmarks.

This module handles the creation of synthetic datasets, including:
- Easy All-Ones instances - to get best-case performance from DP algorithms
- Hard Subset Sum instances - to get worst-case performance from DP algorithms
- Easy Uncorrelated Random instances - to get optimistic average-case performance
- Hard Strongly Correlated Random instances - to get pessimistic average-case performance
"""

from pathlib import Path
from typing import Protocol, Generator, Tuple

import numpy as np

# Standard Output Type
InstanceData = Tuple[np.ndarray, np.ndarray, int]

# ----------------------------------------------------------------------------
# 1. Instance Generator Protocol
# ----------------------------------------------------------------------------

class GeneratorFunc(Protocol):
    """
    Contract: Any function that claims to be a generator 
    should have this exact signature.
    """
    def __call__(
        self, 
        n_items: int, 
        rng: np.random.Generator, 
        n_samples: int, 
        max_value: int
    ) -> Generator[InstanceData, None, None]:
        ...

# ----------------------------------------------------------------------------
# 2. Instance Generators
# ----------------------------------------------------------------------------

def all_ones_instances(
    n_items: int,
    rng: np.random.Generator,
    n_samples: int,
    max_value: int
) -> Generator[InstanceData, None, None]:
    """
    Generates easy instances to get best-case performance from DP algorithms.

    Designed to induce very few cumulative weights.

    Logic:
        Values  = [1, 1, ... 1]
        Weights = [1, 1, ... 1]
        Budgets are the range [n - n_samples, n]

    Args:
        n_items: Number of items (n)
        rng: Ignored
        n_samples: Number of instances
        max_value: Ignored

    Yields:
        Tuple of (values, weights, budget).
    """
    values = np.ones(n_items, dtype=np.int64)
    weights = np.ones(n_items, dtype=np.int64)
    start_b = max(1, n_items - n_samples + 1)

    for b in range(start_b, n_items + 1):
        yield values, weights, b


def subset_sum_instances(
    n_items: int,
    rng: np.random.Generator,
    n_samples: int,
    max_value: int
) -> Generator[InstanceData, None, None]:
    """
    Generates hard instances to get worst-case performance from DP algorithms.
    
    These are a modification of Subset Sum instances designed to maximize
    the number of possible cumulative weights.

    Logic:
        Values  = [1, 2, 4, ... 2^(n-1)]
        Weights = [1, 2, 4, ... 2^(n-1)]
        Budgets are the largest (n_items - max_instances) powers of two.
        If n > 50, we fill rest of array with random numbers valued < B. 

    Args:
        n_items: Number of items (n)
        rng: Ignored
        n_samples: Number of instances
        max_value: Ignored

    Yields:
        Tuple of (values, weights, budget).
    """
    # 1. Generate Powers of Two (Capped at 50)
    n_powers = min(n_items, 50)
    exponents = np.arange(n_powers, dtype=np.int64)
    powers = 2 ** exponents

    # 3. We want the 'n_samples' largest valid budgets
    gap = max(0, n_powers - n_samples)
    n_missing = n_items - n_powers
    
    for b in powers[gap:]:
        filler = rng.integers(1, max(2,b), size=n_missing, dtype=np.int64)
        values = np.concatenate((powers, filler))
        weights = np.concatenate((powers, filler))
        
        yield values, weights, int(b)


def uncorrelated_instances(
    n_items: int,
    rng: np.random.Generator,
    n_samples: int,
    max_value: int
) -> Generator[InstanceData, None, None]:
    """
    Generates easy uncorrelated random instances.
    (From: "Where are the hard knapsack problems?")

    Logic:
        Values  ~ Uniform(1, R)
        Weights ~ Uniform(1, R)
        Budget  = h / (H+1) * sum(weights) for h in [1,num_instances]

    Args:
        n_items: Number of items.
        rng: A seeded numpy random generator.
        n_samples: How many distinct instances to generate.
        max_value: The maximum integer value (R) for values/weights.

    Yields:
        Tuple of (values, weights, budget).
    """
    for i in range(1, n_samples + 1):
        values = rng.integers(1, max_value, size=n_items)
        weights = rng.integers(1, max_value, size=n_items)
        # Pisinger's Budget Formula
        budget = int(i / (n_samples + 1) * np.sum(weights))

        yield values, weights, budget

def correlated_instances(
    n_items: int,
    rng: np.random.Generator,
    n_samples: int,
    max_value: int
) -> Generator[InstanceData, None, None]:
    """
    Generates the hardest known random instances.
    (From: "Where are the hard knapsack problems?")

    Logic:
        Values  ~ weights + R / 10
        Weights ~ Uniform(1, R)
        Budget  = h / (H+1) * sum(weights) for h in [1,num_instances]

    Args:
        n_items: Number of items.
        rng: A seeded numpy random generator.
        n_samples: How many distinct instances to generate.
        max_value: The maximum integer value (R) for values/weights.

    Yields:
        Tuple of (values, weights, budget).
    """
    for i in range(1, n_samples + 1):
        # Start with random weights
        weights = rng.integers(1, max_value, size=n_items)
        # Strong Correlation: V = W + R/10
        values = (weights + max_value / 10).astype(np.int64)
        # Pisinger's Budget Formula
        budget = int(i / (n_samples + 1) * np.sum(weights))

        yield values, weights, budget


# ----------------------------------------------------------------------------
# 3. Dict of Generators
# ----------------------------------------------------------------------------

# Registry for easy instance type, to be used by other modules
GENERATORS: dict[str, GeneratorFunc] = {
    "all-ones": all_ones_instances,
    "subset-sum": subset_sum_instances,
    "uncorrelated": uncorrelated_instances,
    "correlated": correlated_instances
}

