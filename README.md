# Anytime-Constrained Reinforcement Learning

The companion code to the AISTATs paper [Anytime-Constrained Reinforcement Learning](https://proceedings.mlr.press/v238/mcmahan24a). This repository implements a high-performance framework for solving Anytime-Constrained Markov Decision Processes (ACMDPs). We include a general toolkit for working with ACMDPs, and conduct experiments to verify their theoretical guarantees and to compare the scalability of our methods.

## 🌟 Key Features

* **Robust Pipeline:** Includes modules for synthetic data generation (Best-Case, Worst-Case, and Average Case Instances), analytical verification, and automated benchmarking.

* **Algorithmic Suite:**

    * Reduction Solver: An exact, pseudo-polynomial time solver using specialized state augmentation.

    * Approximation Solver: A bicriteria (0,1+ϵ)-approximation algorithm that guarantees value optimality while bounding cost violation.

    * Feasibility Heuristic: A strict feasibility solver that scales the budget to ensure constraints are met, while empirically achieving high value.

* **Sparse Matrix Optimization:** Uses scipy.sparse (CSR) alongside a custom vectorized backward induction algorithm to handle augmented state spaces with millions of nodes. 

## 📂 Project Structure

```
Anytime-CRL/
├── data/                   # Generated datasets
│   ├── easy/
│   ├── hard/
│   └── random/
├── notebooks/
│   └── visualization.ipynb # Interactive plotting dashboard
├── results/                # Experiment logs and saved plots
│   ├── easy/
│   ├── hard/
│   ├── random/
│   └── plots/
├── scripts/                # CLI Entry points
│   ├── generate_data.py    # Dataset creation
│   ├── run_experiment.py   # Benchmark execution
│   └── verify_algos.py     # Theoretical Guarantees tests
├── src/                    # Core Library
│   ├── algorithms.py       # Solver implementations
│   ├── generator.py        # Data generation logic
│   ├── solver.py           # Abstract Base Class & Contracts
│   ├── verification.py     # Instance and Solver Checks
│   └── plotting.py         # Visualization library
├── run_all.sh              # Run all experiments script
└── requirements.txt        # Dependencies
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

Generate the synthetic datasets.

```
Bash
# Generate small validation sets and large random sets (up to H=100)
python -m scripts.generate_data
```

### 3. Validation (Unit Testing)

Before running experiments, verify the solvers against analytical ground truths to ensure correctness.

```
Bash
python -m scripts.verify_algos
Expected Output: [SUCCESS] All tests passed.
```

### 4. Run Benchmark

Run the solvers on the generated data. This script uses incremental saving to protect against crashes.

```
Bash
python -m scripts.run_experiment
```

Alternatively use the provided script "run_all.sh"

### 5. Analytics Visualization

Open notebooks/visualization.ipynb to generate the plots. The notebook will automatically find the latest CSV in results/ and produce:

* Runtime Analysis: Log-scale runtime with 95% Confidence Intervals.

* Constraint Analysis: Worst-case weight to budget comparison of each algorithm.

* Value Analysis: Comparison of achieved value to optimal or approximate lower bound.

## 📦 Dataset Criteria
Our data set mimics the knapsack-like NP-hard ACMDPs appearing in the paper. Formally, each instance is an ACMDP equivalent of a knapsack instance with varying properties.

* **Easy:** Linear correlation (Costs=1, Rewards=1) to ensure minimal blow-up in the augmented states (quadratic), allowing fast solutions. Gives Best-Case performance.
* **Hard:** Increasing powers of two costs and rewards to ensure maximum blow-up in the augmented states (exponential) to stress solvers. Gives Worst-Case performance.
* **Random:** Uniformly Random costs and rewards for Average-Case performance.

## 🧠 Algorithms Implemented

| Algorithm             | Type       | Time Complexity   |
| :-------------------- | :--------- | :---------------- |
| Reduction             | Optimal    | Pseudo-Polynomial |
| Approximation         | Bicriteria | Polynomial        |
| Feasibility Heuristic | Heuristic  | Polynomial        |

## 📊 Reproducibility
To reproduce the exact charts found in the report:

1. Run the full generation pipeline: python -m scripts.generate_data

2. Run the experiment suite: ./run_all.sh

3. Execute the cells in notebooks/visualization.ipynb

## 📜 Citation
If you use this code for your research, please cite:

```
Jeremy McMahan. (2025). Anytime-Constrained Reinforcement Learning. 
GitHub Repository. https://github.com/jermcmahan/Anytime-CRL
```