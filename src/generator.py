"""
Data generation logic for the hard, Knapsack-like Anytime-CMDPs.

This module handles the creation of synthetic datasets, including:
- Easy instances (Linear correlation of costs):
    - Cause minimum blow-up in augmented state space, ensuring fast solving
    - Best-Case Analysis
- Hard instances (Costs are Powers of two / Subset Sum)
    - Cause maximum blow-up in augmented state space, stressing solvers
    - Worst-Case Analysis
- Random instances (Uniformly Distributed Costs and Rewards)
    - Average Case Analysis
"""

from pathlib import Path
from typing import Generator, Tuple

import numpy as np

# Default configuration constants
DEFAULT_NUM_INSTANCES = 25
DEFAULT_VALUE_RANGE = 1000


# ------------------------------------------------------------------------------
# 1. Generate Raw Knapsack Instances
# ------------------------------------------------------------------------------

def generate_easy_knapsacks(n_items: int) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
    """
    Generates a batch of 'Easy' instances that induce minimum blow up in the
    augmented state space, allowing solvers to run very fast.

    Logic:
        Values  = [1, 1, ...  1]
        Weights = [1, 1, ..., 1]
        Budgets are chosen to induce the 10 hardest instances [n-9,n-8,...,n]

    Args:
        n_items: The number of items, n, in the knapsack problem

    Yields:
        Tuple of (values, weights, budget)
    """
    values = np.ones(n_items, dtype=np.int32)
    weights = np.ones(n_items, dtype=np.int32)
    
    # Subsample budgets
    potential_budgets = 1 + np.arange(n_items, dtype=np.int32)
    selected_budgets = potential_budgets[-10:]
    
    # Generate each instance
    for b in selected_budgets:
        yield values, weights, b


def generate_hard_knapsacks(n_items: int) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
    """
    Generates a batch of 'Hard' instances, that maximizes the size 
    of the augmented state space, to push solvers to their limit.

    Logic:
        Values  = [1, 2, 4, ... 2^(n-1)]
        Weights = [1, 2, 4, ... 2^(n-1)]
        Budgets are chosen to induce the 10 hardest instances [2^(n-10),2^(n-9),...,2^(n-1)]

    Args:
        n_items: The number of items, n, in the knapsack problem

    Yields:
        Tuple of (values, weights, budget)
    """
    exponents = np.arange(n_items, dtype=np.int64)
    powers_of_two = 2**exponents

    values = powers_of_two
    weights = powers_of_two
    
    # Subsample budgets
    selected_budgets = powers_of_two[-10:]
    
    # Generate each instance
    for b in selected_budgets:
        yield values, weights, b


def generate_random_knapsacks(
    n_items: int,
    rng: np.random.Generator,
    num_instances: int = DEFAULT_NUM_INSTANCES,
    value_range: int = DEFAULT_VALUE_RANGE,
) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
    """
    Generates uncorrelated random instances.

    Logic:
        Values  ~ Uniform(1, value_range)
        Weights ~ Uniform(1, value_range)
        Budget  = 50% of total weight

    Args:
        n_items: Number of items
        rng: A seeded numpy random generator
        num_instances: How many distinct instances to generate
        value_range: The maximum integer value for weights/values

    Yields:
        Tuple of (values, weights, budget)
    """
    for _ in range(num_instances):
        values = rng.integers(1, value_range, size=n_items)
        weights = rng.integers(1, value_range, size=n_items)

        # Standard Capacity: 50% of total weight
        total_weight = np.sum(weights)
        budget = int(total_weight * 0.5)

        yield values, weights, budget

# ------------------------------------------------------------------------------
# 2. Convert to ACMDP Instances
# ------------------------------------------------------------------------------


def knapsack_to_acmdp(
    values: np.ndarray, weights: np.ndarray, budget: int
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Translates an instance of the knapsack problem to an ACMDP instance.

    Args:
        values: The knapsack instance's values
        weights: The knapsack instance's weights
        budget: The knapsack instance's budget

    Returns:
        An ACMDP tuple, the translation of the knapsack instance
    """
    # Each item corresponds to a time step of the ACMDP
    horizon = len(values)

    # Reshape the rewards to r(s,a) form
    rewards = np.zeros((horizon, 2))
    rewards[:, 1] = values

    # Reshape the costs to c(s,a) form
    costs = np.zeros((horizon, 2))
    costs[:, 1] = weights

    # Transition from item i to item i+1 no matter the action taken
    transitions = np.zeros((2, horizon, horizon))
    for i in range(horizon-1):
        transitions[:, i, i+1] = 1.0
    
    # Self-loop transition after final item
    transitions[:, horizon-1, horizon -1] = 1.0

    # Start at the first item
    start = 0

    return horizon, rewards, costs, transitions, budget, start

# ------------------------------------------------------------------------------
# 3. Save to File
# ------------------------------------------------------------------------------


def save_instance(
    folder_path: Path, filename: str, horizon: int, rewards: np.ndarray, 
    costs: np.ndarray, transitions: np.ndarray, budget: int, start: int
) -> None:
    """
    Helper to save an ACMDP instance to disk in compressed numpy format.

    Args:
        folder_path: The directory Path object
        filename: The filename (without extension)
        horizon: The finite horizon
        rewards: The reward function
        costs: The cost function
        transitions: The transition distribution
        budget: The budget
        start: The start state
    """
    # Make the directory if it does not already exist
    folder_path.mkdir(parents=True, exist_ok=True)
    file_path = folder_path / f"{filename}.npz"

    # Save the instance to file_path
    np.savez_compressed(file_path, 
        horizon=horizon, 
        rewards=rewards, 
        costs=costs,
        transitions=transitions,
        budget=budget,
        start=start,
    )
