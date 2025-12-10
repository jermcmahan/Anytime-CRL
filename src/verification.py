"""
This module provides functions to verify that the project's pipeline is 
working correctly. We focus on 'Easy' and 'Hard' instances since they have 
analytical solutions to compare against.

Functions Summary:
1. Verfies 'Easy' and 'Hard' ACMDP instances from file are correct.
2. Verfies the Reduction Solver's output matches analytical truth.
3. Verfies the Approx & Feasible Solver's output matches theoretical guarantees.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.solver import ACMDP, SolverResult


# -------------------------------------------------------------------------
# 0. IO Helpers
# -------------------------------------------------------------------------


def load_acmdp_from_file(filepath: Path) -> ACMDP:
    """Loads an .npz file into an ACMDP object."""
    data = np.load(filepath)
    return ACMDP(
        horizon=int(data['horizon']),
        rewards=data['rewards'],
        costs=data['costs'],
        transitions=data['transitions'],
        budget=int(data['budget']),
        start=int(data['start'])
    )

def get_instance_files(base_dir: Path, max_H: int) -> List[Tuple[int, List[Path]]]:
    """Helper to find valid files recursively, sorted numerically by H."""
    valid_groups = []
    if not base_dir.exists(): 
        return []
    
    # 1. Collect all folders first
    folders = []
    for folder in base_dir.glob("H_*"):
        try:
            # Parse H immediately
            h_val = int(folder.name.split("_")[1])
            folders.append((h_val, folder))
        except (ValueError, IndexError):
            continue

    # 2. Sort folders numerically by H (the first element of the tuple)
    folders.sort(key=lambda x: x[0])

    # 3. Collect files
    for h_val, folder in folders:
        if h_val <= max_H:
            # Sort files alphanumerically (works for instance_00x format)
            files = sorted(list(folder.glob("*.npz")))
            if files:
                valid_groups.append((h_val, files))
                
    return valid_groups

# -------------------------------------------------------------------------
# 1. Generated ACMDP Structure Checks 
# -------------------------------------------------------------------------

def check_topology_is_chain(env: ACMDP) -> bool:
    """
    Verifies the instance forms a valid Knapsack-like Chain Graph.

    Logic: 
        Horizon = |States|
        Deterministic transitions: State i -> State i+1
        Terminal state S-1 self-loops
    """
    S = env.states_size
    A = env.actions_size
    
    # Check 1: Horizon matches State count and only 2 actions
    if env.horizon != S or A != 2:
        return False
        
    # Check 2: Action 0 (Skip) has 0 cost/reward
    if not (np.all(env.rewards[:, 0] == 0) and np.all(env.costs[:, 0] == 0)):
        return False

    # Check 3: Transitions form a chain
    # For all states i < S-1, P(i, a, i+1) must be 1.0
    for s in range(S - 1):
        if not np.allclose(env.transitions[:, s, s+1], 1.0):
            return False
            
    # Check 4: Terminal state self-loop
    if not np.allclose(env.transitions[:, S-1, S-1], 1.0):
        return False
        
    return True

def check_easy_acmdp(env: ACMDP) -> bool:
    """
    Verifies 'Easy' instance properties on the matrices directly.
    - Action 1 (Take) has Reward=1 and Cost=1 for all items.
    """
    # 1. Base Topology Check
    if not check_topology_is_chain(env):
        return False
        
    # 2. Check reward/cost of action 1 is always 1
    rewards_ones = np.all(env.rewards[:, 1] == 1.0)
    costs_ones = np.all(env.costs[:, 1] == 1.0)
    
    return rewards_ones and costs_ones

def check_hard_acmdp(env: ACMDP) -> bool:
    """
    Verifies 'Hard' instance properties.
    - Action 1 (Take) has rewards/costs as increasing powers of 2.
    """
    if not check_topology_is_chain(env):
        return False
    
    # Generate expected powers of 2: [1, 2, 4, ..., 2^(H-1)]
    expected = 2.0 ** np.arange(env.horizon)
    
    valid_rewards = np.allclose(env.rewards[:, 1], expected)
    valid_costs = np.allclose(env.costs[:, 1], expected)
    valid_budget = (env.budget & (env.budget - 1)) == 0
    
    return valid_rewards and valid_costs and valid_budget

def check_acmdp(category: str, env: ACMDP) -> bool:
    """
    Dispatcher for instance specific checks.
    """
    if category == "easy":
        return check_easy_acmdp(env)
    elif category == "hard":
        return check_hard_acmdp(env)
    else:
        # Random has no specific structure to enforce
        return True 

# -------------------------------------------------------------------------
# 2. Solver Checks 
# -------------------------------------------------------------------------

def check_exact(result: SolverResult) -> bool:
    """
    For these specific Easy/Hard instances: optimal value and cost are both 
    equal to the budget.
    """
    return (np.isclose(result.value, result.budget) 
        and np.isclose(result.cost, result.budget)
    )

def check_approx(result: SolverResult, epsilon: float) -> bool:
    """
    Theoretical Guarantee for our Bicriteria: 
    1. Value >= Optimal
    2. Cost <= Budget * (1 + epsilon)
    """
    return (result.value >= result.budget 
        and result.cost <= (result.budget * (1.0 + epsilon))
    )

def check_feasible(result: SolverResult) -> bool:
    """
    Theoretical Gurarantee for a Feasible Solver: Cost <= Budget
    """
    return result.cost <= result.budget

