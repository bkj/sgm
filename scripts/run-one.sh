#!/bin/bash

# run.sh

# --
# Synthetic data

# Rscript make-data.R | tee best

N=$1

python main-scipy.py \
    --A-path data/A-$N \
    --B-path data/B-$N \
    --P-path data/P-$N \
    --mode exact \
    --symmetric


python main-torch.py \
    --A-path data/A-$N \
    --B-path data/B-$N \
    --P-path data/P-$N \
    --mode exact \
    --symmetric

# # --
# # Connectome data

# python main-scipy.py \
#     --A-path _data/connectome/$DATASET/sparse/A1.ordered.edges \
#     --B-path _data/connectome/$DATASET/sparse/A2.ordered.edges \
#     --P-path _data/connectome/$DATASET/sparse/P_start.edges \
#     --mode exact \
#     --symmetric