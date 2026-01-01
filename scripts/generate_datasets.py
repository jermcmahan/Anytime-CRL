"""CLI entry point for Knapsack Data Generation.

Automatically generates a full suite of instance types as defined by src.instances.py

Usage:
    # 1. Generate full benchmarks
    python -m scripts.generate_datasets

    # 2. Generate a validation set for learning
    python -m scripts.generate_datasets --out data/validation --min_n 20 --step_n 20 --n_samples 5
"""

import argparse
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Registry
from src.instances import GENERATORS

def main():
    parser = argparse.ArgumentParser(description="Generate Knapsack Datasets")
    
    # Output & Config
    parser.add_argument("--out", type=str, default="data/benchmark", help="Output directory")
    parser.add_argument("--n_samples", type=int, default=30, help="Instances per type, per size")
    parser.add_argument("--seed", type=int, default=42)
    
    # Instance Specs
    parser.add_argument("--min_n", type=int, default=10)
    parser.add_argument("--max_n", type=int, default=100)
    parser.add_argument("--step_n", type=int, default=10)
    parser.add_argument("--max_value", type=int, default=1000, help="Max value for random gen")
    
    args = parser.parse_args()

    # Setup
    rng = np.random.default_rng(args.seed)
    root_dir = Path(args.out)
    root_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Generating Dataset Suite ---")
    print(f"Output:  {root_dir}")
    print(f"Config:  n=[{args.min_n}:{args.max_n},step={args.step_n}], Samples={args.n_samples}")
    
    # --- Data Generation Loop ---
    # We iterate over every generator defined in src.instances
    for name, gen_func in GENERATORS.items():
        
        type_dir = root_dir / name
        type_dir.mkdir(exist_ok=True)
        
        # Loop over sizes
        for n in range(args.min_n, args.max_n + 1, args.step_n):
            
            n_dir = type_dir / f"n_{n}"
            n_dir.mkdir(exist_ok=True)
            
            # Call Generator (Standardized Interface)
            gen = gen_func(n, rng, args.n_samples, args.max_value)
            
            # Consume
            for i, (v, w, b) in enumerate(tqdm(gen, total=args.n_samples, desc=f"{name:<12} n={n:<3}")):
                if i >= args.n_samples: break
                
                filename = n_dir / f"instance_{i:03d}.npz"
                np.savez_compressed(filename, values=v, weights=w, budget=b)

    print("\nGeneration Complete.")

if __name__ == "__main__":
    main()
