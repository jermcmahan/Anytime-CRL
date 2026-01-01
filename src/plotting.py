"""
Visualization Library for ACMDP Experiments.

This module handles:
1. Loading the latest benchmark CSVs.
2. Generating plots (Runtime, Violation, Optimality).
3. Saving High-DPI images to 'results/plots'.
"""

import os
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# --- Configuration ---
# Dynamically find the project root relative to this file (src/plotting.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Ensure plots directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set global style
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.25)

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'lines.linewidth': 2.5,       # Thicker lines
    'axes.titleweight': 'bold',   # Bold titles
    'axes.titlesize': 16,         # Explicit title size
    'axes.labelsize': 14,         # Explicit label size
    'xtick.labelsize': 12,        # Tick size
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 13,
})


def get_latest_data(dataset_type: str) -> pd.DataFrame:
    """
    Finds and loads the latest CSV for a specific dataset type (easy/hard/random).
    """
    if dataset_type:
        target_dir = RESULTS_DIR / dataset_type
    else:
        target_dir = RESULTS_DIR
    
    if not target_dir.exists():
        print(f"[Warn] Directory not found: {target_dir}")
        return pd.DataFrame()
    
    csv_files = sorted(target_dir.glob("benchmark_*.csv"))
    
    if not csv_files:
        print(f"[Warn] No CSVs found in {target_dir}")
        return pd.DataFrame()
    
    latest_file = csv_files[-1]
    print(f"[{dataset_type.upper()}] Loading: {latest_file.name}")
    return pd.read_csv(latest_file)


def save_current_plot(filename: str):
    """
    Saves the current matplotlib figure to the results/plots directory.
    """
    path = PLOTS_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {path}")


def plot_runtime(df: pd.DataFrame, test_name: str):
    """
    Plots Runtime vs N.
    """
    if df.empty: 
        return

    if 'time' not in df:
        print(f"[Warn] No 'time' column found")
        return

    plt.figure()
    
    sns.lineplot(
        data=df, 
        x="horizon", 
        y="time", 
        hue="algorithm", 
        style="algorithm", 
        markers=True, 
        dashes=False,
        errorbar=('ci', 95),
        alpha=0.8
    )
    
    plt.yscale("log")
    plt.title(f"Runtime Analysis (95% CI)", fontsize=14)
    plt.ylabel("Time (s) - Log Scale", fontsize=12)
    plt.xlabel("Horizon ($H$)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), title='Algorithm', loc='upper left')
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    
    save_current_plot(f"runtime_{test_name.lower()}.png")
    plt.show()

def plot_constraint(df: pd.DataFrame, test_name: str):
    """
    Plots Weight/Budget ratio with Min/Max bands.
    """
    if df.empty: return

    if 'cost-ratio' not in df:
        print(f"[Warn] No 'cost-ratio' column found")
        return
    
    plt.figure()
    
    sns.lineplot(
        data=df,
        x="horizon",
        y="cost-ratio",
        hue="algorithm",
        style="algorithm",
        markers=True,
        errorbar=lambda x: (x.min(), x.max()),
        alpha=0.8
    )
    
    plt.axhline(y=1, color='black', linestyle='-', linewidth=1, label="Budget Threshold")
    
    plt.title(f"Constraint Analysis (Min/Max)", fontsize=14)
    plt.ylabel("Cost / Budget (> 1 = Violation)", fontsize=12)
    plt.xlabel("Horizon ($H$)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), title='Algorithm', loc='upper left')
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    
    save_current_plot(f"constraint_{test_name.lower()}.png")
    plt.show()

def plot_optimality(df: pd.DataFrame, test_name: str, compare: str = ""):
    """
    Plots Optimality Ratio (LB / UB)
    """
    if df.empty: 
        return

    if compare:
        original_algo_order = df['algorithm'].unique()
        # Pivot Data
        df_pivot = df.pivot(
            index=["horizon", "instance_id"], 
            columns="algorithm", 
            values="value"
        ).reset_index()

        plot_data = []
        sorted_algo_cols = [alg for alg in original_algo_order if alg in df_pivot.columns]

        for algo in sorted_algo_cols:
            lb = df_pivot[algo]
            ub = df_pivot[compare]
            
            subset = df_pivot[["horizon", "instance_id"]].copy()
            subset["opt-ratio"] = lb / ub
            subset["algorithm"] = algo
            
            plot_data.append(subset)

        if not plot_data:
            return

        df = pd.concat(plot_data, ignore_index=True)

    if 'opt-ratio' not in df:
        print(f"[Warn] No 'opt-ratio' column found")
        return

    plt.figure()
    
    sns.lineplot(
        data=df,
        x="horizon",
        y="opt-ratio",
        hue="algorithm",
        style="algorithm",
        markers=True,
        errorbar=('ci', 95),
        alpha=0.8
    )

    plt.axhline(y=1, color='black', linestyle='-', linewidth=1, label="Optimal")
    
    plt.title(f"Approximate Value Analysis (95% CI)", fontsize=14)
    plt.ylabel("Alg / Optimal * 100 (100% = Optimal)", fontsize=12)
    plt.xlabel("Horizon ($H$)", fontsize=12)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.legend(bbox_to_anchor=(1.05, 1), title='Algorithm', loc='upper left', labelspacing=1.2)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    save_current_plot(f"optimality_{test_name.lower()}.png")
    plt.show()
