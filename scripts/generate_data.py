"""
CLI entry point for generating hard Knapsack-like Anytime-CMDP datasets.
Generates the Easy, Hard, and Random instances as described in generator.py
and writes them to file in .npz format.

Instance Summary:
- 'Easy' instances are designed to induce a solver's Best-Case behavior
- 'Hard' instances are designed to induce a solver's Worst-Case behavior
- 'Random' instances are designed to induce a solver's Average-Case behavior

Usage:
    python -m scripts.generate_data
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm 

# Add project root to path to allow importing 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generator import (
    DEFAULT_NUM_INSTANCES,
    DEFAULT_VALUE_RANGE,
    generate_easy_knapsacks,
    generate_hard_knapsacks,
    generate_random_knapsacks,
    knapsack_to_acmdp,
    save_instance,
)

# Sizes: small for validation, large for scaling experiments
SIZES_TO_TEST = list(range(10, 101, 10))

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Anytime-CMDP Dataset")

    # Group 1: General Settings
    core_group = parser.add_argument_group("Core Settings")
    core_group.add_argument(
        "--out_dir", type=str, default="data", help="Output root directory"
    )
    core_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Group 2: Generator Specifics
    gen_group = parser.add_argument_group("Generator Configuration")
    gen_group.add_argument(
        "--samples_per_H",
        type=int,
        default=DEFAULT_NUM_INSTANCES,
        help="Number of instances per H (for Random) or Max Subsamples (for Easy/Hard)",
    )
    gen_group.add_argument(
        "--max_val",
        type=int,
        default=DEFAULT_VALUE_RANGE,
        help="Max reward/cost for random items",
    )

    args = parser.parse_args()

    root_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)

    print(f"Generating datasets in '{root_dir}'...")
    print(f"Config: Seed={args.seed}, Samples={args.samples_per_H}, Range={args.max_val}")

    for H in SIZES_TO_TEST:
        print(f"\n--- Processing Size H={H} ---")

        # ---------------------------------------------------------
        # 1. Easy Instances
        # ---------------------------------------------------------
        easy_out_dir = root_dir / "easy" / f"H_{H}"
        easy_gen = generate_easy_knapsacks(H)
        
        count = 0
        for idx, knapsack in enumerate(tqdm(easy_gen, desc="Easy", leave=False)):
            save_instance(easy_out_dir, 
                f"instance_{idx:03d}", *knapsack_to_acmdp(*knapsack),
            )
            count += 1
        
        print(f"  > Saved {count} Easy instances to {easy_out_dir}")

        # ---------------------------------------------------------
        # 2. Hard Instances
        # ---------------------------------------------------------
        hard_out_dir = root_dir / "hard" / f"H_{H}"
        
        # Limit H <= 50 because Spanner instances grow as 2^H, so
        # converting to a C long fails around H=60
        if H <= 50:
            hard_gen = generate_hard_knapsacks(H)
            count = 0
            for idx, knapsack in enumerate(tqdm(hard_gen, desc="Hard", leave=False)):
                save_instance(hard_out_dir, 
                    f"instance_{idx:03d}", *knapsack_to_acmdp(*knapsack),
                )
                count += 1
            print(f"  > Saved {count} Hard instances to {hard_out_dir}")
        else:
            print(f"  > Skipping Hard instances (H={H} too large for C long)")

        # ---------------------------------------------------------
        # 3. Random Instances
        # ---------------------------------------------------------
        random_out_dir = root_dir / "random" / f"H_{H}"
        random_gen = generate_random_knapsacks(
            n_items=H, 
            rng=rng, 
            num_instances=args.samples_per_H, 
            value_range=args.max_val
        )
        
        count = 0
        for idx, knapsack in enumerate(tqdm(random_gen, total=args.samples_per_H, desc="Random", leave=False)):
            save_instance(random_out_dir, 
                f"instance_{idx:03d}", *knapsack_to_acmdp(*knapsack),
            )
            count += 1
            
        print(f"  > Saved {count} Random instances to {random_out_dir}")

    print("\nGeneration complete.")

if __name__ == "__main__":
    main()
