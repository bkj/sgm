#!/bin/bash

# run.sh

# --
# Individual runs

NUM_NODES=500
NUM_SEEDS=32

python kasios.py --num-nodes $NUM_NODES --num-seeds $NUM_SEEDS --backend scipy.classic.jv
python kasios.py --num-nodes $NUM_NODES --num-seeds $NUM_SEEDS --backend scipy.sparse.jv
python kasios.py --num-nodes $NUM_NODES --num-seeds $NUM_SEEDS --backend scipy.fused.jv

python kasios.py --num-nodes $NUM_NODES --num-seeds $NUM_SEEDS --backend scipy.sparse.auction
python kasios.py --num-nodes $NUM_NODES --num-seeds $NUM_SEEDS --backend scipy.fused.auction

python kasios.py --num-nodes $NUM_NODES --num-seeds $NUM_SEEDS --backend torch.classic.jv

# --
# Run experiment across grid of parameter settings and problems

mkdir -p results/grid
python kasios-grid.py --seed 123 > results/grid/grid-00000.jl