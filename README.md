# Anytime-Constrained Reinforcement Learning

The companion code to the AISTATs paper [Anytime-Constrained Reinforcement Learning](https://proceedings.mlr.press/v238/mcmahan24a). This repository implements a high-performance framework for solving Anytime-Constrained Markov Decision Processes (ACMDPs). We include a general toolkit for working with ACMDPs, including model-based solvers and gymnasium wrappers, and run benchmarking experiments on classic knapsack datasets to verify their theoretical guarantees.

## 🌟 Key Features

* **Robust Pipeline:** Includes modules for synthetic data generation (Best-Case, Worst-Case, and Average Case Instances), infinite streams for learning, and automated benchmarking.

* **Algorithmic Suite:**

    * **Reduction Solver:** An exact, pseudo-polynomial time solver using specialized state augmentation. Allows rational and negative costs!

    * **Approximation Solver:** A bicriteria (0,1+ϵ)-approximation algorithm that guarantees value optimality while bounding cost violation.

    * **Feasibility Heuristic:** A strict feasibility solver that scales the budget to ensure constraints are met, while empirically achieving high value.

    * **Anytime-Constraint Wrapper:** A gymnasium wrapper that transforms *any* standard MDP into a constrained environment via robust action masking and budget tracking.

* **Sparse Matrix Optimization:** Uses scipy.sparse (CSR) alongside a custom vectorized backward induction algorithm to handle augmented state spaces with millions of nodes. 

## 📂 Project Structure

```
Anytime-CRL/
├── data/                    # Generated datasets
│   ├── benchmark/           # Instances for Benchmarking
│   └── validation/          # Instances for RL validation
├── src/                     # Core Library
│   ├── rl/                  # RL Environment and Curriculum
│   ├── algorithms.py        # Solver implementations
│   ├── instances.py         # Data generation logic
│   ├── solver.py            # Abstract Base Class & Contracts
│   └── plotting.py          # Visualization library
├── scripts/                 # CLI Entry points
│   ├── generate_datasets.py # Dataset creation
│   ├── run_benchmark.py     # Benchmark execution
│   └── train_agent.py       # RL training
├── notebooks/
│   └── visualization.ipynb  # Interactive plotting dashboard
├── results/                 # Experiment logs and saved plots
│   ├── benchmark_*.csv      # Summary of Benchmarking results
│   ├── models/              # RL trained models
│   └── plots/               # Visualizations of Benchmarking 
└── requirements.txt         # Dependencies
```

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies.
```
Bash
git clone https://github.com/jermcmahan/Anytime-CRL.git
cd Anytime-CRL
pip install -r requirements.txt
```

### 2. Data Generation

1. Generate the larger benchmarking datasets in data/benchmark

```
Bash
python -m scripts.generate_datasets
```

2. Generate the smaller validation datasets in data/validation

```
Bash
python -m scripts.generate_datasets --min_n 20 --step_n 20 --n_samples 5
--out data/validation
```

### 3. Run Benchmark

Run the solvers on the generated data. This script uses incremental saving to protect against crashes.

```
Bash
python -m scripts.run_benchmark
```

**Optionally:** run scripts.train_agent or use our default trained PPO model.

### 4. Visualization

Open notebooks/visualization.ipynb to generate the plots. The notebook will automatically find the latest CSV in results/ and produce:

* Runtime Analysis: Log-scale runtime with 95% Confidence Intervals.

* Constraint Analysis: Worst-case weight to budget comparison of each algorithm.

* Value Analysis: Comparison of achieved value to optimal or approximate lower bound.

The notebook also includes the expected behavior and conclusions.

## 📦 Benchmark Dataset
Our data set mimics the knapsack-like NP-hard ACMDPs appearing in the paper. Formally, each instance is an ACMDP equivalent of a knapsack instance with varying properties. The specific knapsack instances used are classical benchmarks as implemented in the repository [Knapsack-Approximation](https://github.com/jermcmahan/Knapsack-Approximation).

## 🧠 Algorithms Implemented

| Algorithm             | Type       | Time Complexity     |
| :-------------------- | :--------- | :------------------ |
| Reduction             | Optimal    | Pseudo-Polynomial   |
| Approximation         | Bicriteria | Polynomial          |
| Feasibility Heuristic | Heuristic  | Polynomial          |
| PPO w/ action-masking | Heuristic  | Poly-time Inference |

## 📊 Reproducibility
To reproduce the exact charts found in the report:

1. Run the full generation pipeline

2. Run the experiment suite

3. Execute the cells in notebooks/visualization.ipynb

## 📜 Citation
If you use this code for your research, please cite:

```
Jeremy McMahan. (2025). Anytime-Constrained Reinforcement Learning. 
GitHub Repository: https://github.com/jermcmahan/Anytime-CRL
```