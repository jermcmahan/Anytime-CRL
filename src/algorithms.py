"""
Implementation of Anytime-CMDP Optimization algorithms.

Includes:
- Reduction to Standard Backward Induction (Exact, Pseudo-Polynomial Time)
- Bicriteria, (0,epsilon)-relative (Approximation, Polynomial Time)
- Feasibility Guaranteeing Heuristic (Heuristic, Polynomial Time)
- Reinforcement Learning (Heuristic, Polynomial Time Inference) 
"""

from typing import Tuple, Dict, Any
from collections import deque

import numpy as np
import scipy.sparse as sp

# --- Optional Imports for RL Solver ---
# This prevents the script from crashing if ML libraries are missing
try:
    import torch
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from src.rl.knapsack_env import KnapsackBaseEnv
    from src.rl.anytime_wrapper import AnytimeWrapper
    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False

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
        aug_env, state_map = self._augment(env)

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
        solver_info = {'aug_env': aug_env, 'state_map': state_map}

        return policy, optimal_value, worst_case_cost, solver_info

    def _augment(self, env: ACMDP) -> Tuple[ACMDP, Dict]:
        """
        Computes the (sparse) cost-augmented MDP representation.

        Args:
            env: the original Anytime-CMDP instance

        Returns:
            An ACMDP instance containing the cost-augmented MDP with sparse transitions
            and the augmented cost function, and a dictionary that maps raw augmented
            states to their index form
        """
        S = env.states_size
        A = env.actions_size
        B = env.budget

        # --- 1. Augmented State Space Construction ---
        # BFS-style enumeration of all realizable (s,c)-pairs
        # Maps (s, cost) -> the first time this pair is encountered,
        # (i.e. its index in order of the enumeration)
        
        # Agent starts at s_0 with 0 cost accumulated
        state_map = {(env.start, 0.0): 0}

        # Compute rewards, costs, and transitions as we go
        penalty = np.max(np.abs(env.rewards)) * (env.horizon + 1)
        rewards_list = []
        costs_list = []
        transitions_list = [([], [], []) for _ in range(A)]

        # Queue for BFS: (s, cost, h, idx) entries maintained
        queue = deque([(env.start, 0.0, 0, 0)])
        
        # We process layer by layer to avoid cycles 
        while queue:
            s, c, h, idx = queue.popleft()
            
            for a in range(A):
                next_c = c + env.costs[s,a]

                # 1. --- Augmented Rewards and Costs ---
                # If c + c(s,a) > B, apply penalty
                if next_c > B:
                    next_c = B               # Truncate to B
                    rewards_list.append(-penalty)
                else:
                    # r_aug((s,c),a) = r(s,a)
                    rewards_list.append(env.rewards[s,a])

                # c_aug((s,c),a) = c(s,a)
                costs_list.append(env.costs[s,a])

                # 2. --- Augmented Transitions and States ---
                rows, cols, data = transitions_list[a]

                # Stop expanding if we hit the horizon, add self-loops instead
                if h >= env.horizon:
                    rows.append(idx)
                    cols.append(idx)
                    data.append(1.0)
                    continue
                
                for next_s in np.where(env.transitions[a, s] > 0)[0]:
                    next_state = (next_s, next_c)
                    
                    if next_state not in state_map:
                        # Identify augmented states with the time they were found
                        next_idx = len(state_map)
                        state_map[next_state] = next_idx
                        queue.append((next_s, next_c, h + 1, next_idx))
                    else:
                        next_idx = state_map[next_state]

                    # Update transitions csr format
                    rows.append(idx)
                    cols.append(next_idx)
                    data.append(env.transitions[a, s, next_s])

        aug_S = len(state_map) # total number of augmented states

        # Convert lists to arrays
        aug_rewards = np.array(rewards_list).reshape((aug_S, A))
        aug_costs = np.array(costs_list).reshape((aug_S, A))
        aug_transitions = []

        for a in range(A):
            rows, cols, data = transitions_list[a]

            aug_transitions.append(sp.csr_array(
                    (data, (rows, cols)), 
                    shape=(aug_S, aug_S),
                )
            )

        # --- 4. Augmented Start State Construction ---
        aug_start = 0 # (env.start, 0)

        return (ACMDP(
                horizon=env.horizon,
                rewards=aug_rewards,
                costs=aug_costs,
                transitions=aug_transitions,
                budget=B,
                start=aug_start,
            ),
            state_map,
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
        state_map = parent_info['state_map'] 
        
        # Unscale the costs and budget
        aug_budget = env.budget
        aug_costs = np.zeros(scaled_aug_env.costs.shape)

        for (s,c), i in state_map.items():
            aug_costs[i,:] = env.costs[s,:]

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
        solver_info = {'aug_env': aug_env, 'state_map': state_map}

        return policy, value, actual_cost, solver_info

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


class RLSolver(ACMDPSolver):
    """
    Solves Knapscak ACMDP instances using a pre-trained Reinforcement Learning policy.
    
    1. Accepts an 'ACMDP' object (Matrix format).
    2. Extracts the raw arrays (Values/Weights) from the matrix.
    3. Injects them into 'KnapsackBaseEnv'.
    4. Runs the PPO model to produce a solution trajectory.
    """

    def __init__(self, model_path: str, max_n: int, device: str = "auto"):
        # Check if optional dependencies were imported successfully
        if not _RL_AVAILABLE:
            raise ImportError(
                "RLSolver requires 'torch' and 'sb3_contrib'. "
                "Please install them to use this solver."
            )

        self.model_path = model_path
        self.max_n = max_n
        
        # 1. Load the Agent
        try:
            self.model = MaskablePPO.load(model_path, device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

        # 2. Fixed-instance generator that always outputs the current problem
        self._current_problem = None
        
        def _gen(rng):
            if self._current_problem is None:
                raise RuntimeError("No problem instance injected!")
            return self._current_problem

        # 3. Initialize Environment
        self.env = KnapsackBaseEnv(max_n=max_n, generator=_gen)
        self.env = AnytimeWrapper(self.env,safe_actions_exist=True)
        self.env = ActionMasker(self.env, lambda env: env.action_masks())

    @property
    def name(self) -> str:
        return "RL Agent (PPO)"

    def _solve_impl(self, env: ACMDP,
    ) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        # Fixed-instance generator
        values = env.rewards[:,1]
        weights = env.costs[:,1]
        budget = env.budget
        self._current_problem = (values, weights, budget)

        # Episode run
        obs, _ = self.env.reset()
        value = 0
        weight = 0
        solution = np.zeros(len(values), dtype=np.int8)
        
        for i in range(len(values)):
            action, _ = self.model.predict(
                obs, 
                action_masks=self.env.action_masks(), 
                deterministic=True
            )

            # If the item is chosen, update value and weight
            if action == 1:
                solution[i] = 1
                value += values[i]
                weight += weights[i]

            # Move to next item
            obs, _, _, _, _ = self.env.step(action)

        return solution, value, weight, {}




    