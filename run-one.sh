#!/bin/bash

# run.sh

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
