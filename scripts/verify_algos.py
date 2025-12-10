"""
Analytical Validation Script (Integration Test). 

Manually checks that data written to file correctly matches the generator 
logic, and then manually checks each algorithm produces solutions matching 
its theoretical guarantees.

This script:
1. Verifies 'Easy' and 'Hard' ACMDP instances from file are correct.
2. Runs the Reduction Solver and checks its output matches Analytical Truth.
3. Runs the Approx & Feasible Solvers and checks their guarantees hold.

Usage:
    python -m scripts.validate_algos

WARNING:
    Validating large H instances, especially 'Hard', will cause huge runtimes.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.verification import *
from src.algorithms import *


def verify(category: str, data_dir: Path, max_H: int) -> List[Dict[str, Any]]:
    """
    Checks correctness of our dataset and solver results on that data.
    """
    cat_dir = data_dir / category
    folders = get_instance_files(cat_dir, max_H)
    
    if not folders:
        print(f"[Warn] No '{category}' files found (checked up to H={max_H})")
        return []

    results = []
    
    # Parameters
    eps1, eps2 = 1.0, 0.5

    # Initialize Solvers
    solver_exact = ReductionSolver()
    solver_approx_1 = ApproxSolver(epsilon=eps1)
    solver_approx_2 = ApproxSolver(epsilon=eps2)
    solver_feas_1 = FeasibleSolver(epsilon=eps1)
    solver_feas_2 = FeasibleSolver(epsilon=eps2)

    print(f"--- Validating {category.title()} Instances (Max H={max_H}) ---")
    for H, files in folders:
        for file in tqdm(files, desc=f"Processing H={H}"):
            record = {
                "category": category,
                "filename": file.name,
                "size": file.parent.name, 
                "status": "PASS",
                "error_msg": ""
            }
            
            try:
                # 1. Load
                env = load_acmdp_from_file(file)
                
                # 2. Native Structure Check
                if not check_acmdp(category, env):
                    raise ValueError(f"ACMDP Structure Check Failed for {category}")

                # 3. Exact Solver
                res_ex = solver_exact.solve(env)
                if not check_exact(res_ex):
                    raise ValueError(f"Exact Failed: V={res_ex.value}, C={res_ex.cost}, B={res_ex.budget}")

                # 4. Approx Solver
                res_ap1 = solver_approx_1.solve(env)
                if not check_approx(res_ap1, eps1):
                    raise ValueError(f"Approx(eps={eps1}) Failed: V={res_ap1.value}, C={res_ap1.cost}, B={res_ap1.budget}")

                res_ap2 = solver_approx_2.solve(env)
                if not check_approx(res_ap2, eps2):
                    raise ValueError(f"Approx(eps={eps2}) Failed: V={res_ap2.value}, C={res_ap2.cost}, B={res_ap2.budget}")

                # 5. Feasible Solver
                res_fs1 = solver_feas_1.solve(env)
                if not check_feasible(res_fs1):
                     raise ValueError(f"Feasible(eps={eps1}) Failed: C={res_fs1.cost} > {res_fs1.budget}")
                     
                res_fs2 = solver_feas_2.solve(env)
                if not check_feasible(res_fs2):
                     raise ValueError(f"Feasible(eps={eps2}) Failed: C={res_fs2.cost} > {res_fs2.budget}")

            except Exception as e:
                record["status"] = "FAIL"
                record["error_msg"] = str(e)

            results.append(record)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run Analytical Validation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--max_H", type=int, default=20, 
                        help="Only validate instances with H <= this value")
    parser.add_argument("--category", type=str, choices=["all", "easy", "hard"], 
                        default="all", help="Which category to validate")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{data_path}' not found.")
        return
    
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    if args.category in ["all", "easy"]:
        all_results.extend(verify("easy", data_path, args.max_H))
        
    if args.category in ["all", "hard"]:
        all_results.extend(verify("hard", data_path, args.max_H))
        
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = out_path / "validation_report.csv"
        df.to_csv(csv_path, index=False)
        
        print("\n--- Validation Summary ---")
        print(df.groupby(["category", "status"]).size())
        
        failures = df[df["status"] == "FAIL"]
        if not failures.empty:
            print(f"\n[ALERT] {len(failures)} failures detected. See {csv_path}")
            pd.set_option('display.max_colwidth', None)
            print(failures[["size", "filename", "error_msg"]].head())
        else:
            print(f"\n[SUCCESS] All {len(df)} tests passed.")
    else:
        print("No tests run.")

if __name__ == "__main__":
    main()
