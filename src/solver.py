"""
Abstract base class for Anytime-CMDP solvers.

This module defines the contract that all solver implementations must follow,
ensuring consistent timing and result reporting across experiments.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Union

import numpy as np
import scipy.sparse as sp

@dataclass(frozen=True)
class ACMDP:
    """
    Digital representation of a (finite, stationary) Anytime-CMDP.
    
    Attributes:
        horizon: The finite time horizon (H).
        rewards: The reward function R(s,a).
        costs: The cost function C(s,a).
        transitions: The transition distribution P(s'|s,a).
        budget: The cumulative cost budget (B).
        start: The initial state index (s_0).
    """
    horizon: int
    rewards: np.ndarray      # Shape: (S, A)
    costs: np.ndarray        # Shape: (S, A)
    transitions: Union[np.ndarray, List[sp.csr_array]]  # Shape: (A, S, S)
    budget: float
    start: int

    @property
    def states_size(self) -> int:
        return self.rewards.shape[0]

    @property
    def actions_size(self) -> int:
        return self.rewards.shape[1]

    def __post_init__(self):
        """
        Checks that the instance is a valid Anytime-CMDP
        """

        # Extract dimensions for validation
        S, A = self.rewards.shape
        
        # 1. Shape Validation
        if self.costs.shape != (S, A):
            raise ValueError(f'Shape Mismatch: Rewards {self.rewards.shape} vs Costs {self.costs.shape}')
        
        # 2. Transition Validation
        if isinstance(self.transitions, np.ndarray):
            if self.transitions.shape != (A, S, S):
                raise ValueError(f'Shape Mismatch: Transitions {self.transitions.shape} expected {(A, S, S)}')

            # Check that sum over next_states (axis 2) is 1.0 for all actions/states
            if not np.allclose(self.transitions.sum(axis=2), 1.0):
                raise ValueError('Invalid Transitions: probabilities do not sum to 1.0')
        
        elif isinstance(self.transitions, list):
            # Sparse Checks
            if len(self.transitions) != A:
                raise ValueError(f"Transition list length {len(self.transitions)} != Actions {A}")

            # Check that every row in every matrix sums to 1.0 (approximately)
            for i, mat in enumerate(self.transitions):
                if not sp.issparse(mat):
                    raise ValueError(f"Transition list item {i} is not a sparse matrix")

                if not np.allclose(mat.sum(axis=1), 1.0):
                    raise ValueError(f"Sparse transitions for action {i} do not sum to 1.0")

        # 3. Start State Validation
        if not (0 <= self.start < S):
            raise ValueError(f'Invalid Start State: {self.start} not in [0, {S-1}]')


@dataclass(frozen=True)
class SolverResult:
    """
    Standardized result object returned by all solvers.

    Attributes:
        algorithm_name: The identifier of the algorithm used.
        policy: A deterministic policy.
        value: The expected value of the policy.
        cost: The worst-case cost of the policy.
        budget: The capacity constraint provided.
        success: Whether a policy was found.
        runtime: Execution time in seconds.
        solver_info: Solver specific info for policy evaluation.
    """
    algorithm_name: str
    policy: np.ndarray
    value: float
    cost: float
    budget: int
    success: bool
    runtime: float
    solver_info: dict


class ACMDPSolver(ABC):
    """
    Abstract Base Class (ABC) for Anytime-CMDP solvers.

    Enforces a standard interface for solving, timing, and verifying results.
    Subclasses must implement `_solve_impl` and the `name` property.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The display name of the algorithm for plots and logs.
        """
        pass

    def solve(self, env: ACMDP) -> SolverResult:
        """
        Executes the solver on a specific instance and measures its running time.

        Args:
            env: an Anytime-CMDP instance

        Returns:
            A SolverResult object containing the solution and performance metrics.
        """
        # Timing
        start_time = time.perf_counter()
        
        # Call implementation
        policy, value, cost, solver_info = self._solve_impl(env)
        
        end_time = time.perf_counter()

        # For non-negative rewards, a negative value implies infeasibility
        success = bool(value > -np.max(np.abs(env.rewards)))

        return SolverResult(
            algorithm_name=self.name,
            policy=policy,
            value=value,
            cost=cost,
            budget=env.budget,
            success=success,
            runtime=end_time - start_time,
            solver_info=solver_info,
        )

    @abstractmethod
    def _solve_impl(self, env: ACMDP) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        """
        Internal implementation of the solve method.

        Args:
            env: an Anytime-CMDP instance

        Returns:
            Tuple containing (solution_policy, expected_value, worst_case_cost, solver_info).
        """
        pass

