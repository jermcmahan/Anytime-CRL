#!/bin/bash

# 1. Easy Instances
echo "--- Running Easy Benchmark ---"
python3 -m scripts.run_experiment \
    --data_dir data/easy \
    --out_dir results/easy \
    --max_H 100

# 2. Hard Instances
echo "--- Running Hard Benchmark ---"
python3 -m scripts.run_experiment \
    --data_dir data/hard \
    --out_dir results/hard \
    --max_H 50

# 3. Random Instances
echo "--- Running Random Benchmark ---"
python3 -m scripts.run_experiment \
    --data_dir data/random \
    --out_dir results/random \
    --max_H 100

echo "All experiments complete."