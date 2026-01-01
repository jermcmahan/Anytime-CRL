"""
Execution script for benchmarking Anytime-CMDP solvers.

This script:
1. Scans the data directory for instance folders (n_X).
2. Runs algorithms incrementally.
3. SAVES results to CSV after every folder (n) to prevent data loss on crash.

Usage:
    python -m scripts.run_benchmark
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# All available solvers
from src.algorithms import (
    ReductionSolver,
    ApproxSolver,
    FeasibleSolver,
    RLSolver,
)
from src.solver import ACMDPSolver, SolverResult, ACMDP

def setup_solvers(model_path: Optional[str] = None, max_n_pad: int = 100) -> List[ACMDPSolver]:
    """
    Instantiates the list of solvers to participate in the benchmark.
    If model_path is provided, adds the RL Agent to the list.
    """
    solvers = [
        ReductionSolver(),
        ApproxSolver(epsilon=1.0),
        ApproxSolver(epsilon=0.5),
        ApproxSolver(epsilon=0.25),
        FeasibleSolver(epsilon=1.0),
        FeasibleSolver(epsilon=0.5),
        FeasibleSolver(epsilon=0.25),
    ]

    # Add RL Solver if requested
    if model_path:
        print(f"Loading RL Agent from: {model_path}")
        solvers.append(
            RLSolver(model_path=model_path, max_n=max_n_pad)
        )

    return solvers

# ------------------------------------------------------------------------------
# I/O Helpers
# ------------------------------------------------------------------------------

def knapsack_to_acmdp(
    values: np.ndarray, weights: np.ndarray, budget: int
) -> ACMDP:
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

    return ACMDP(
        horizon=horizon, 
        rewards=rewards, 
        costs=costs, 
        transitions=transitions, 
        budget=budget, 
        start=start
    )

def get_latest_model(models_dir: str = "results/models") -> Optional[str]:
    """Finds the most recently created PPO model directory."""
    p = Path(models_dir)
    if not p.exists():
        return None
    
    # Sort by directory name (ISO timestamps sort alphabetically: PPO_2023...)
    dirs = sorted([d for d in p.iterdir() if d.is_dir() and "PPO_" in d.name])
    
    if not dirs:
        return None
        
    latest_dir = dirs[-1]
    model_path = latest_dir / "best_model.zip"
    
    if not model_path.exists():
        print(f"[Warn] Found {latest_dir} but it has no best_model.zip.")
        return None
        
    return str(model_path)

# ------------------------------------------------------------------------------
# Run Solvers
# ------------------------------------------------------------------------------

def run_solvers_on_instance(
    solvers: List[ACMDPSolver],
    env: ACMDP,
    dataset_name: str
) -> List[Dict[str, Any]]:
    """Runs all applicable solvers on a single problem instance."""
    results = []
    H = env.horizon

    for solver in solvers:
        # Skip Reduction solver for too large H's
        if solver.name == "Reduction" and dataset_name == "subset-sum":
            # For Hard instances, H=20 is often the RAM limit (2^20 states)
            continue
        
        try:
            res = solver.solve(env)
            
            results.append({
                "algorithm": res.algorithm_name,
                "time": res.runtime,
                "value": res.value,
                "cost": res.cost,
                "budget": res.budget,
                "horizon": H,
            })

        except Exception as e:
            # Catch Crashes
            print(f"[Warn] {solver.name} failed on H={H}: {e}")


    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Anytime-CMDP Benchmark")
    
    # Defaults set for Auto-Pilot
    parser.add_argument("--data_dir", type=str, default="data/benchmark", 
                        help="Directory containing subfolders or n_X folders")
    
    parser.add_argument("--out_dir", type=str, default="results", 
                        help="Where to save CSV logs")
    
    parser.add_argument("--max_n", type=int, default=100, 
                        help="Skip folders with n > this value (File Filtering)")
    
    # RL Specific Args
    parser.add_argument("--model_path", type=str, default="latest",
                        help="Path to model.zip. Use 'latest' to find latest, or 'none' to skip.")
    
    parser.add_argument("--max_n_pad", type=int, default=100,
                        help="Tensor padding size for RL Agent.")
    
    args = parser.parse_args()

    # 1. Resolve Model Path
    final_model_path = None
    if args.model_path.lower() == "latest":
        final_model_path = get_latest_model()
        if not final_model_path:
            print("[Info] No RL models found in results/models. Running Classical only.")
    elif args.model_path.lower() != "none":
        final_model_path = args.model_path

    # 2. Setup Solvers
    solvers = setup_solvers(final_model_path, args.max_n_pad)
    
    # 3. Resolve Data Directories
    root_path = Path(args.data_dir)
    if not root_path.exists():
        print(f"Error: Data directory '{root_path}' not found.")
        return

    # Identify if root is a single dataset or a suite
    if list(root_path.glob("n_*")):
        dataset_paths = [root_path]
    else:
        dataset_paths = [x for x in root_path.iterdir() if x.is_dir()]
        
    # 4. Global Configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # CHANGE: Single Master CSV File
    csv_path = out_dir / f"benchmark_{timestamp}.csv"
    
    print(f"--- Starting Experiment: {timestamp} ---")
    print(f"Datasets: {[d.name for d in dataset_paths]}")
    print(f"Saving to: {csv_path}")
    
    # --- MAIN LOOP OVER DATASETS ---
    for data_path in dataset_paths:
        dataset_name = data_path.name # e.g., 'hard', 'easy'
        
        print(f"\n>>> Processing Dataset: {dataset_name}")
        
        folders = sorted(
            data_path.glob("n_*"), 
            key=lambda x: int(x.name.split("_")[1])
        )
        
        if not folders:
            print(f"[Warn] No n_X folders found in {data_path}. Skipping.")
            continue

        for folder in folders:
            try:
                n = int(folder.name.split('_')[1])
            except (IndexError, ValueError):
                continue 

            if n > args.max_n:
                continue

            # Local buffer for THIS folder only
            folder_records = []
            files = sorted(list(folder.glob("*.npz")))
            
            for file in tqdm(files, desc=f"[{dataset_name}] n={n}"):
                # Load
                data = np.load(file, allow_pickle=True)
                env = knapsack_to_acmdp(data['values'], data['weights'], int(data['budget']))
                
                # Run
                instance_results = run_solvers_on_instance(solvers, env, dataset_name)
                
                # Augment
                for res in instance_results:
                    res["timestamp"] = timestamp
                    res["dataset_type"] = dataset_name # <--- This column is key for Analysis
                    res["instance_id"] = file.stem
                    folder_records.append(res)

            # INCREMENTAL SAVE
            if folder_records:
                df_chunk = pd.DataFrame(folder_records)
                write_header = not csv_path.exists()
                df_chunk.to_csv(csv_path, mode='a', header=write_header, index=False)
                del df_chunk, folder_records

    print(f"\nAll Experiments Complete. Master results saved to {csv_path}")

if __name__ == "__main__":
    main()

