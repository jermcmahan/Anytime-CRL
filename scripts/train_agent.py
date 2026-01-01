"""
Training Script.

Features:
- Curriculum Learning (Round-Robin mix of Easy, Hard, Correlated)
- Evaluation Loop (Recursive validation on fixed dataset mix)
- Model Checkpointing (Best Model + Final)
- Learning Rate Schedule (Linear Decay)
- Early Stopping (Patience based on Eval)
- Clean Artifacts (config.json, evaluations.npz)
"""

import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EventCallback, StopTrainingOnNoModelImprovement
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy as evaluate_maskable_policy

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.knapsack_env import KnapsackBaseEnv
from src.rl.anytime_wrapper import AnytimeWrapper

from src.rl.curriculum import (
    make_uncorrelated_stream,
    make_correlated_stream,
    make_hard_stream,
    make_round_robin_generator,
    make_fixed_dataset_generator
)

# --- 1. Helpers ---

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def seed_everything(seed: int):
    """Locks all sources of randomness."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- 2. Custom Callback ---

class MaskableEvalCallback(EventCallback):
    """
    Evaluates the agent periodically using Mask-Aware logic.
    Saves the best model and raw logs (evaluations.npz).
    Triggers 'callback_after_eval' (e.g., Early Stopping) if provided.
    """
    def __init__(
        self, 
        eval_env, 
        best_model_save_path, 
        eval_freq=10000, 
        n_eval_episodes=5,
        callback_after_eval=None
    ):
        super().__init__(callback_after_eval, verbose=1)
        self.eval_env = eval_env
        self.best_model_save_path = Path(best_model_save_path)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        
        # Log Arrays
        self.timesteps = []
        self.results = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # 1. Evaluate
            mean_reward, std_reward = evaluate_maskable_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            
            # 2. Log Data
            self.timesteps.append(self.num_timesteps)
            self.results.append(mean_reward)
            
            np.savez(
                self.best_model_save_path / "eval_logs" / "evaluations.npz",
                timesteps=self.timesteps,
                results=self.results
            )

            if self.verbose > 0:
                print(f"Eval {self.n_calls}: {mean_reward:.2f} +/- {std_reward:.2f}")

            # 3. Save Best Model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"New best mean reward! ({self.best_mean_reward:.2f})")
                self.model.save(self.best_model_save_path / "best_model")
                
            # 4. Trigger Early Stopping
            self.last_mean_reward = mean_reward
            self._on_event()
            
        return True

# --- 3. Env Factories ---

def make_train_env_fn(min_n, max_n, pad_n):
    """
    Creates the 'Training Cocktail': A Round-Robin mix of different difficulty types.
    """
    def _init():
        # 1. Create the Specialists
        gen_random = make_uncorrelated_stream(min_n, max_n)
        gen_corr   = make_correlated_stream(min_n, max_n)
        gen_hard   = make_hard_stream(min_n, max_n)
        
        # 2. Mix them (33% Random, 33% Correlated, 33% Subset Sum)
        mixed_gen = make_round_robin_generator([
            gen_random,
            gen_corr,
            gen_hard
        ])
        
        # 3. Create Env
        env = KnapsackBaseEnv(max_n=pad_n, generator=mixed_gen)
        env = AnytimeWrapper(env, safe_actions_exist=True)
        return ActionMasker(env, lambda env: env.action_masks())
    return _init

def make_eval_env_fn(data_dir, pad_n):
    """
    Creates the 'Evaluation Ruler': A fixed set of instances loaded from disk.
    """
    def _init():
        # Uses the recursive loader to find ALL files in data_dir
        gen_fn = make_fixed_dataset_generator(data_dir)
        env = KnapsackBaseEnv(max_n=pad_n, generator=gen_fn)
        env = AnytimeWrapper(env, safe_actions_exist=True)
        return ActionMasker(env, lambda env: env.action_masks())
    return _init

# --- 4. Main ---

def main():
    parser = argparse.ArgumentParser()
    # Problem Settings
    parser.add_argument("--min_n", type=int, default=10)
    parser.add_argument("--max_n_train", type=int, default=100)
    parser.add_argument("--max_n_pad", type=int, default=100)
    
    # Training Settings
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_envs", type=int, default=8)
    
    # Evaluation & Paths
    parser.add_argument("--eval_data", type=str, default='data/validation', help="Path to validation set")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--model_dir", type=str, default="results/models")
    
    args = parser.parse_args()

    # Setup
    run_id = f"PPO_{datetime.now().strftime('%Y%m%d-%H%M')}"
    run_dir = Path(args.model_dir) / run_id
    (run_dir / "eval_logs").mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f"--- Training: {run_id} (Seed: {args.seed}) ---")
    
    set_random_seed(args.seed)
    seed_everything(args.seed)

    # Envs
    train_env = make_vec_env(
        make_train_env_fn(args.min_n, args.max_n_train, args.max_n_pad),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv
    )
    
    # Count validation files RECURSIVELY
    val_files = list(Path(args.eval_data).rglob("*.npz"))
    n_val = len(val_files)
    if n_val == 0:
        raise ValueError(f"No .npz files found in {args.eval_data}")
    print(f"Validation on {n_val} instances.")

    eval_env = make_vec_env(
        make_eval_env_fn(args.eval_data, args.max_n_pad),
        n_envs=1,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv
    )

    # Callbacks
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=args.patience, 
        min_evals=5,
        verbose=1
    )
    
    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        eval_freq=10000,
        n_eval_episodes=n_val,
        callback_after_eval=stop_callback
    )

    # Model
    model = MaskablePPO(
        "MultiInputPolicy", 
        train_env, 
        verbose=1, 
        seed=args.seed,
        learning_rate=linear_schedule(args.lr),
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        batch_size=256
    )
    
    model.learn(total_timesteps=args.steps, callback=[eval_cb])
    
    model.save(run_dir / "final_model")
    train_env.close()
    eval_env.close()
    print(f"Done. Artifacts saved to {run_dir}")

if __name__ == "__main__":
    main()


