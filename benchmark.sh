#!/bin/bash

# benchmark.sh

# --
# Synthetic

# generate data
Rscript utils/make-correlated-gnp.R 0 0.5 0.7 # dense; p=0.5; rho=0.7
Rscript utils/make-correlated-gnp.R 0 0.05 0.7 # dense; p=0.05; rho=0.7


DATASET="_data/synthetic/dense/0.5/5000/"
# python main-scipy.py \
#     --A-path $DATASET/A.csv \
#     --B-path $DATASET/B.csv \
#     --num-seeds 100 \
#     --lap-mode jv

python main-torch.py \
    --A-path $DATASET/A.csv \
    --B-path $DATASET/B.csv \
    --num-seeds 100 \
    --lap-mode jv

python main-torch.py \
    --A-path $DATASET/A.csv \
    --B-path $DATASET/B.csv \
    --num-seeds 100 \
    --lap-mode jv


python main-torch.py \
    --A-path $DATASET/A.csv \
    --B-path $DATASET/B.csv \
    --num-seeds 100 \
    --lap-mode auction