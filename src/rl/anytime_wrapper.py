"""
Anytime-Constraint Wrapper.

Wraps a base environment to enforce budget constraints via Action Masking.
Flattens nested dictionary observations to be compatible with Stable-Baselines3.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional

class AnytimeWrapper(gym.Wrapper):
    """
    A wrapper that transforms a constrained MDP into a maskable MDP.

    Key Features:
    - Action Masking: Prevents the agent from taking actions that violate the budget.
    - Generalizability: Observations include normalized ratios (Remaining/Budget),
      allowing policies to transfer between instances of different scales.
    - Feasibility Guarantee: Adds a "Dummy/Abort" action that is always valid but
      terminates the episode with a penalty, ensuring the agent never gets stuck.

    ---------------------------------------------------------------------------
    BASE ENVIRONMENT CONTRACT
    ---------------------------------------------------------------------------
    This wrapper is AGNOSTIC to the specific problem (Knapsack, Gridworld, etc.).
    However, the Base Environment MUST strictly adhere to the following contract
    by populating the `info` dictionary in `reset()` and `step()`:

    1. `info['cost']` (float) -> REQUIRED in step()
       The amount of budget consumed by the action just taken. 
       - Positive values reduce the remaining budget.
       - Negative values (recharging) increase the remaining budget.

    2. `info['next_costs']` (np.ndarray) -> REQUIRED in reset() and step()
       A float array of shape (n_base_actions,) containing the IMMEDIATE cost 
       for every possible action in the current state.
       - Used to compute the feasibility mask for the *next* step.
       - Index `i` must correspond to Action `i`.

    3. `info['budget']` (float) -> OPTIONAL in reset()
       If provided, the wrapper will overwrite its `default_budget` with this value.
       This allows for instance-specific budgets (e.g., generated randomly).

    Note, the Base Environment should have a Discrete action space.
    ---------------------------------------------------------------------------
    """

    def __init__(
        self, 
        base_env: gym.Env, 
        budget: float = 100.0, 
        penalty: float = -10_000.0,
        safe_actions_exist: bool = False
    ):
        """
        Args:
            base_env: The environment to wrap.
            budget: Default budget (optionally, overwritten by reset info).
            penalty: Reward penalty for the 'Abort' action.
            safe_actions_exist: If False, adds an 'Abort' action (safe fallback, slower learning).
                                If True, relies on base env having safe actions (e.g., Skip).
        """
        super().__init__(base_env)

        self.base_env = base_env
        self.budget = float(budget)
        self.penalty = penalty
        self.safe_actions_exist = safe_actions_exist

        # --- Internal State ---
        self.remaining = 0.0
        self.n_base_actions = base_env.action_space.n
        
        # Define Total Actions
        if not self.safe_actions_exist:
            self.n_total_actions = self.n_base_actions + 1
        else:
            self.n_total_actions = self.n_base_actions
        
        self.valid_actions = np.ones(self.n_total_actions, dtype=bool)
        self.next_costs = np.zeros(self.n_total_actions, dtype=np.float32)

        # --- Augmented Observation Space ---
        # We add: 'budget_left' (Scalar) and 'action_costs' (Vector)
        new_spaces = {
            'budget_left': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            'action_costs': spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_total_actions,), dtype=np.float32)
        }

        # Handle Nesting: If base is Dict, merge. If Box, wrap in 'state'.
        if isinstance(base_env.observation_space, spaces.Dict):
            self.is_dict_obs = True
            merged_spaces = base_env.observation_space.spaces.copy()
            merged_spaces.update(new_spaces)
            self.observation_space = spaces.Dict(merged_spaces)
        else:
            self.is_dict_obs = False
            new_spaces['state'] = base_env.observation_space
            self.observation_space = spaces.Dict(new_spaces)
        
        self.cur_obs: Dict[str, Any] = {}

        # Augment action space if no safe actions: Abort = (N+1)
        self.action_space = spaces.Discrete(self.n_total_actions)

    def _build_obs(self, base_obs: Any) -> Dict[str, Any]:
        """Helper to merge wrapper features with base observation."""
        
        # Calculate Ratios
        if self.budget != 0:
            remaining_ratio = self.remaining / self.budget
            cost_ratios = self.next_costs / self.budget
        else:
            remaining_ratio = 0.0
            cost_ratios = np.zeros_like(self.next_costs)

        constraint_features = {
            'budget_left': np.array([remaining_ratio], dtype=np.float32),
            'action_costs': cost_ratios.astype(np.float32),
        }

        if self.is_dict_obs:
            final_obs = base_obs.copy()
            final_obs.update(constraint_features)
            return final_obs
        else:
            constraint_features['state'] = base_obs
            return constraint_features

    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resets the environment and initializes the budget trackers."""
        obs, info = self.base_env.reset(seed=seed, options=options)
        
        # Capture Budget from Info (Instance Injection)
        if 'budget' in info:
            self.budget = float(info['budget'])
        
        self.remaining = self.budget
        self.next_costs[:self.n_base_actions] = info['next_costs'].astype(np.float32)
        self.cur_obs = self._build_obs(obs)

        return self.cur_obs, info

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Executes a step:
            If dummy action taken, terminates with penalty.
            Otherwise, steps base env and updates budget.
        """
        # 1. Abort Check
        if action == self.n_base_actions:
            return self.cur_obs, self.penalty, True, False, {"abort": True}

        # 2. Base Step
        base_obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # 3. Update State
        self.remaining -= float(info['cost'])
        self.next_costs[:self.n_base_actions] = info['next_costs'].astype(np.float32)

        self.cur_obs = self._build_obs(base_obs)
        return self.cur_obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Required by sb3_contrib.MaskablePPO."""
        np.greater_equal(self.remaining, self.next_costs, out=self.valid_actions)
        if not self.safe_actions_exist:
            self.valid_actions[self.n_base_actions] = True
        return self.valid_actions

