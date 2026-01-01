"""
Gymnasium Environment for the 0/1 Knapsack Problem.

This module implements the base MDP for the knapsack problem.
It is designed to be wrapped by `AnytimeWrapper` for constraint enforcement.
"""

from typing import Callable, Optional, Tuple, Dict, Any, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Type alias: Generator returns (values, weights, budget)
GeneratorFn = Callable[[np.random.Generator], Tuple[np.ndarray, np.ndarray, int]]

class KnapsackBaseEnv(gym.Env):
    """
    A Gymnasium environment for the 0/1 Knapsack Problem.

    **The Separation of Concerns:**
    - This Base Env manages: Item sequence, Values (Reward), and Weights (Cost).
    - The Wrapper manages: Budget tracking, Feasibility masks, and Remaining Capacity.
    
    **Observation Space (Dict):**
    - `value_ratios` (Box): All item values normalized by the total value sum.
    - `weight_ratios` (Box): All item weights normalized by the budget.
    - `current_value` (Box): The normalized value of the specific item under consideration.
    """

    def __init__(
        self, 
        max_n: int = 100, 
        generator: Optional[GeneratorFn] = None
    ):
        """
        Args:
            max_n: Fixed size for the observation space (zero-padding limit).
            generator: Function returning (values, weights, budget).
                       Defaults to a random integer generator if None.
        """
        super().__init__()

        self.max_n = max_n

        # Default Generator: Random integers
        if not generator:
            self.generator = lambda rng: (
                rng.integers(1, 100, 50),
                rng.integers(1, 100, 50),
                rng.integers(20, 200)
            )
        else:
            self.generator = generator

        # --- Internal State ---
        self.n = 0
        self.cur_idx = 0
        self.values: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self._temp_budget = 0.0  # Stored only to pass to Wrapper via info

        # --- Buffers (Pre-allocated) ---
        self.norm_values = np.zeros(max_n, dtype=np.float32)
        self.norm_weights = np.zeros(max_n, dtype=np.float32)

        # --- Spaces ---
        # Actions: 0 = Skip, 1 = Take
        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Dict({
            "value_ratios": spaces.Box(low=-1.0, high=1.0, shape=(max_n,), dtype=np.float32),
            "weight_ratios": spaces.Box(low=0.0, high=np.inf, shape=(max_n,), dtype=np.float32),
            "current_value": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        })

    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Resets the environment. Supports 'Instance Injection' via options.
        """
        super().reset(seed=seed)

        # 1. Instance Generation (Injection vs. Random)
        if options and 'instance' in options:
            values, weights, budget = options['instance']
        else:
            values, weights, budget = self.generator(self.np_random)

        # 2. Validation
        self.n = len(values)
        if self.n > self.max_n:
            raise ValueError(f"Instance size {self.n} exceeds max_n {self.max_n}")

        self.values = values.astype(np.float32)
        self.weights = weights.astype(np.float32)
        self._temp_budget = float(budget)

        # 3. Normalization (Safe Division)
        total_val = np.sum(self.values)
        
        self.norm_values.fill(0.0)
        if total_val > 0:
            self.norm_values[:self.n] = self.values / total_val

        self.norm_weights.fill(0.0)
        if self._temp_budget > 0:
            self.norm_weights[:self.n] = self.weights / self._temp_budget

        self.cur_idx = 0

        # 4. Construct Info Contract
        # We assume the first item exists; if n=0, cost is 0.
        first_weight = self.weights[0] if self.n > 0 else 0.0
        
        info = {
            'budget': self._temp_budget,
            'next_costs': np.array([0.0, first_weight], dtype=np.float32)
        }

        return self._get_obs(), info

    def _get_obs(self) -> Dict[str, Any]:
        """Helper to construct the observation dict."""
        if self.cur_idx < self.n:
            curr_val = np.array([self.norm_values[self.cur_idx]], dtype=np.float32)
        else:
            curr_val = np.zeros(1, dtype=np.float32)

        return {
            "value_ratios": self.norm_values.copy(),
            "weight_ratios": self.norm_weights.copy(),
            "current_value": curr_val,
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Steps the environment.
        Delegates cost calculation to the 'info' dictionary for the wrapper.
        """
        reward = 0.0
        terminated = False
        cost = 0.0

        # 1. Apply Action (Calculate Reward & Cost)
        if action == 1: # Take
            reward = self.norm_values[self.cur_idx]
            cost = self.weights[self.cur_idx]
        
        # 2. Advance
        self.cur_idx += 1
        
        # 3. Check Termination & Lookahead
        if self.cur_idx >= self.n:
            terminated = True
            next_cost = 0.0
        else:
            next_cost = self.weights[self.cur_idx]

        # 4. Info Contract
        info = {
            'cost': float(cost),
            'next_costs': np.array([0.0, next_cost], dtype=np.float32)
        }

        return self._get_obs(), reward, terminated, False, info
