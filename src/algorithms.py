"""
Implementation of Anytime-CMDP Optimization algorithms.

Includes:
- Reduction to Standard Backward Induction (Exact, Pseudo-Polynomial Time)
- Bicriteria, (0,epsilon)-relative (Approximation, Polynomial Time)
- Feasibility Guaranteeing Heuristic (Heuristic, Polynomial Time)
"""

from typing import Tuple, Dict, Any

import numpy as np
import scipy.sparse as sp

from .solver import ACMDP, ACMDPSolver


class ReductionSolver(ACMDPSolver):
    """
    Solves an Anytime-CMDP by reducing it to an unconstrained MDP via state augmentation,
    then solves the MDP with standard Backward Induction. Uses sparse implementations
    to drastically reducy time and memory needed for augmented transition functions.

    Computes an exactly optimal solution and runs in Pseudo-Polynomial Time
    """

    @property
    def name(self) -> str:
        return "Reduction"

    def _solve_impl(self, env: ACMDP) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        # 1. Construct the augmented MDP
        aug_env = self._augment(env)

        # 2. Solve the augmented MDP (Sparse) 
        S_aug = aug_env.states_size
        H = aug_env.horizon
        A = aug_env.actions_size
        
        # Use int8 to save RAM.
        policy = np.zeros((S_aug, H), dtype=np.int8)
        
        # Optimal value and Q function for one step in the future
        V_future = np.zeros(S_aug)
        Q_values = np.zeros((S_aug, A))
        
        # --- Sparse Backward Induction ---
        for h in range(H - 1, -1, -1):
            for a in range(A):
                # Bellman Update
                expected_future = aug_env.transitions[a] @ V_future
                Q_values[:, a] = aug_env.rewards[:, a] + expected_future
            
            # Policy Optimiztion
            best_actions = np.argmax(Q_values, axis=1)
            policy[:, h] = best_actions
            V_future = Q_values[np.arange(S_aug), best_actions]

        # 3. Compute metadata
        optimal_value = V_future[aug_env.start]
        worst_case_cost = self._policy_cost(aug_env, policy)
        solver_info = {'aug_env': aug_env}

        return policy, optimal_value, worst_case_cost, solver_info

    def _augment(self, env: ACMDP) -> ACMDP:
        """
        Computes the (sparse) cost-augmented MDP representation. Also, computes a 
        modified cost function to be used in the _policy_cost helper method.

        Args:
            env: the original Anytime-CMDP instance

        Returns:
            An ACMDP instance containing the cost-augmented MDP with sparse transitions
            and the augmented cost function
        """
        S = env.states_size
        A = env.actions_size
        B = env.budget

        # A state for every (state, cost)-pair
        C_size = B + 1
        aug_S =  S * C_size

        # 1. Create augmented reward and cost functions
        # r_aug((s,c),a) = r(s,a), unless c + c(s,a) > B, then -penalty
        penalty = np.sum(np.abs(env.rewards)) * env.horizon + 1
        aug_rewards = np.zeros((aug_S, A))

        for s in range(S):
            for a in range(A):
                # Find smallest starting c that will cause violation
                c_over = max(B - int(env.costs[s,a]) + 1, 0)
                aug_rewards[s*C_size + c_over: (s+1) * C_size, a] = -penalty
                aug_rewards[s*C_size: s*C_size + c_over, a] = env.rewards[s,a]

        # c((s,c),a) = c(s,a)
        aug_costs = np.zeros((aug_S, A))
        for s in range(S):
            aug_costs[s * C_size: (s+1) * C_size] = env.costs[s]

        # 2. Create augmented transitions (Sparse)
        aug_transitions = []
        c_range = np.arange(B+1)

        for a in range(A):
            all_rows = []
            all_cols = []
            all_data = []

            for s in range(S):
                s_next = np.where(env.transitions[a,s] > 0)[0]
                next_cs = c_range + int(env.costs[s, a])
                
                # Any cost > B maps to state index B
                next_cs[next_cs > B] = B
                
                # Calculate Source Indices: (s, 0), (s, 1), ..., (s, B)
                current_aug_indices = s * C_size + c_range

                for sn in s_next:
                    next_aug_indices = sn * C_size + next_cs
                    
                    # Store the batch
                    all_rows.append(current_aug_indices)
                    all_cols.append(next_aug_indices)
                    all_data.append(np.full(C_size, env.transitions[a,s,sn]))

            # Concatenate the lists into flat arrays for SciPy
            rows_flat = np.concatenate(all_rows)
            cols_flat = np.concatenate(all_cols)
            data_flat = np.concatenate(all_data)
            
            # Sparse matrix construction
            sparse_mat = sp.csr_array(
                (data_flat, (rows_flat, cols_flat)), 
                shape=(aug_S, aug_S)
            )
            
            aug_transitions.append(sparse_mat)

        # 3. Create augmented start state
        aug_start = env.start * C_size

        return ACMDP(
                horizon=env.horizon,
                rewards=aug_rewards,
                costs=aug_costs,
                transitions=aug_transitions,
                budget=B,
                start=aug_start,
            )

    def _policy_cost(self, env: ACMDP, policy: np.ndarray) -> float:
        """
        Computes the largest cost accumulated by the policy along any realizable history.
        Requires List[sp.csr_matrix] transitions and large state spaces.

        Args:
            env: The augmented environment used by _solve_impl
            policy: The augmented policy solution output by _solve_impl

        Returns:
            The anytime cost of the given policy for the given environment
        """
        S = policy.shape[0]

        # Base case, take immediate costs at time H
        future_cost = env.costs[np.arange(S), policy[:, env.horizon-1]]

        # Anytime-cost recursion, modified for sparse transitions
        for h in range(env.horizon-2, -1, -1):
            current_cost = np.zeros(S)
            
            for a in np.unique(policy[:, h]):
                # Identify which states take action 'a'
                a_states = (policy[:, h] == a)

                if not np.any(a_states):
                    continue
                
                # 1. Get transitions for action 'a' on relevant states
                P_sub = env.transitions[a][a_states,:]
                
                # 2. Set columns to future costs
                P_sub.data[:] = future_cost[P_sub.indices]
                
                # 3. Compute row-wise max
                future_max = P_sub.max(axis=1).toarray().flatten()
                
                # 4. Bellman-like update
                cost = env.costs[a_states, a]
                current_cost[a_states] = cost + np.maximum(0, future_max)
            
            future_cost = current_cost

        return future_cost[env.start]

class ApproxSolver(ReductionSolver):
    """
    Implements the bicriteria approximation algorithm for Anytime-CMDPs. Works by
    appropriately rounding the cost and budget, then solving the rounded instance
    using the Reduction Solver.

    Theoretical Guarantee:
        Cost <= Budget * (1 + epsilon)
        Value >= Optimal Value (of original problem)

    Attributes:
        epsilon: the approximation parameter
    """
    @property
    def name(self) -> str:
        return f"Approximation ($\\epsilon$={self.epsilon})"

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        super().__init__()

    def _solve_impl(self, env: ACMDP) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        # Avoid division by zero
        if env.budget == 0 or self.epsilon == 0:
            return super()._solve_impl(env)
        
        # 1. Scale costs by a base factor defined to keep violation bounded
        base = self.epsilon * env.budget / env.horizon

        # Use floor to allow "fitting more" (violation allowed)
        scaled_costs = np.floor(env.costs / base).astype(int)
        scaled_budget = int(np.floor(env.budget / base))

        scaled_env = ACMDP(
            horizon=env.horizon,
            rewards=env.rewards,
            costs=scaled_costs,
            transitions=env.transitions,
            budget=scaled_budget,
            start=env.start
        )

        # 2. Solve scaled instance through parent (Backward Induction)
        policy, value, _, parent_info = super()._solve_impl(scaled_env)

        # 3. Create unscaled augmented environment for policy cost computation
        scaled_aug_env = parent_info['aug_env']

        # Number of (s,c)-pairs for each s
        C_size = scaled_aug_env.states_size // env.states_size
        
        # Unscale the costs and budget
        aug_budget = env.budget
        aug_costs = np.zeros(scaled_aug_env.costs.shape)

        for s in range(env.states_size):
            aug_costs[s * C_size: (s+1) * C_size] = env.costs[s]

        aug_env = ACMDP(
                horizon=scaled_aug_env.horizon,
                rewards=scaled_aug_env.rewards,
                costs=aug_costs,
                transitions=scaled_aug_env.transitions,
                budget=aug_budget,
                start=scaled_aug_env.start,
            )

        # 4. Calculate the policy's true anytime cost
        actual_cost = super()._policy_cost(aug_env, policy)

        return policy, value, actual_cost, {'aug_env': aug_env}

class FeasibleSolver(ApproxSolver):
    """
    Implements the feasible heuristic algorithm for Anytime-CMDPs. Works by
    reducing the budget before running ApproxSolver to ensure no violation.

    Theoretical Guarantee: (if a policy is found, which is not guaranteed)
        Cost <= Budget

    Attributes:
        epsilon: the approximation parameter
    """
    @property
    def name(self) -> str:
        return f"Feasible Heuristic ($\\epsilon$={self.epsilon})"

    def __init__(self, epsilon: float = 1.0):
        super().__init__(epsilon)

    def _solve_impl(self, env: ACMDP) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        # Reduce the budget to absorb the approximation error
        reduced_budget = int(env.budget / (1 + self.epsilon))
        
        reduced_env = ACMDP(
                horizon=env.horizon,
                rewards=env.rewards,
                costs=env.costs,
                transitions=env.transitions,
                budget=reduced_budget,
                start=env.start
            )

        return super()._solve_impl(reduced_env)

    