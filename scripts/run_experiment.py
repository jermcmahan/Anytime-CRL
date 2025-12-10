"""
Execution script for benchmarking Anytime-CMDP solvers.

This script:
1. Scans the data directory for instance folders (H_X).
2. Runs algorithms incrementally.
3. SAVES results to CSV after every folder (H) to prevent data loss on crash.

Usage:
    python -m scripts.run_experiment --data_dir data/random --max_h 50
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.verification import load_acmdp_from_file, get_instance_files
from src.algorithms import (
    ReductionSolver,
    ApproxSolver,
    FeasibleSolver,
)
from src.solver import ACMDPSolver, SolverResult, ACMDP


def setup_solvers() -> List[ACMDPSolver]:
    """Instantiates the list of solvers to participate in the benchmark."""
    return [
        # Optimal but slow
        ReductionSolver(),
        # Bicriteria Approximation
        ApproxSolver(epsilon=1.0),
        ApproxSolver(epsilon=0.5),
        ApproxSolver(epsilon=0.25),
        # Feasibility Heuristic
        FeasibleSolver(epsilon=1.0),
        FeasibleSolver(epsilon=0.5),
        FeasibleSolver(epsilon=0.25),
    ]


def run_solvers_on_instance(
    solvers: List[ACMDPSolver],
    env: ACMDP,
    slow_limit: int,
    dataset_name: str
) -> List[Dict[str, Any]]:
    """Runs all applicable solvers on a single problem instance."""
    results = []
    H = env.horizon

    for solver in solvers:
        # Skip Reduction solver for too large H's
        if solver.name == "Reduction":
            # For Hard instances, H=20 is often the RAM limit (2^20 states)
            if dataset_name == "hard" and H > 20:
                continue
            # For Random/Easy, we can usually go higher (e.g. 50-60)
            if dataset_name == "random" and H > slow_limit:
                continue
        
        try:
            res = solver.solve(env)
            
            results.append({
                "algorithm": res.algorithm_name,
                "time": res.runtime,
                "value": res.value,
                "cost": res.cost,
                "budget": res.budget,
                "horizon": env.horizon,
                "policy_found": res.success,
            })

        except Exception as e:
            # Catch Crashes
            print(f"[Warn] {solver.name} failed on H={H}: {e}")


    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Anytime-CMDP Benchmark")
    parser.add_argument("--data_dir", type=str, default="data/random", 
                        help="Directory containing H_X folders")
    parser.add_argument("--out_dir", type=str, default="results/random", 
                        help="Where to save CSV logs")
    parser.add_argument("--max_H", type=int, default=100, 
                        help="Skip folders with H > this value")
    parser.add_argument("--slow_limit", type=int, default=50, 
                        help="Skip slow (optimal) solvers on H's > this value")
    args = parser.parse_args()

    # Configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{data_path}' not found.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = data_path.name
    csv_path = out_dir / f"benchmark_{dataset_name}_{timestamp}.csv"

    # Collect Folders
    folders = get_instance_files(data_path, args.max_H)

    solvers = setup_solvers()
    
    print(f"--- Starting Experiment: {timestamp} ---")
    print(f"Dataset: {data_path}")
    print(f"Output: {csv_path}")

    # Main Loop
    for H, files in folders:
        if H > args.max_H:
            continue

        # Local buffer for THIS folder only
        folder_records = []
        
        for file in tqdm(files, desc=f"Processing H={H}"):
            # 1. Load Data
            env = load_acmdp_from_file(file)
            
            # 2. Run Solvers
            instance_results = run_solvers_on_instance(solvers, env, args.slow_limit, dataset_name)
            
            # 3. Augment Records
            for res in instance_results:
                res["timestamp"] = timestamp
                res["dataset_path"] = str(data_path)
                res["instance_id"] = file.stem
                folder_records.append(res)

        # 4. Incremental save per folder
        if folder_records:
            df_chunk = pd.DataFrame(folder_records)
            
            # If file doesn't exist, write header. If it does, append (no header).
            write_header = not csv_path.exists()
            df_chunk.to_csv(csv_path, mode='a', header=write_header, index=False)
            
            # Optional: Flush memory
            del df_chunk, folder_records

    print(f"\nExperiment Complete.")

if __name__ == "__main__":
    main()