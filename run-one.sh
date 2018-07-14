#!/bin/bash

# run.sh

# Rscript make-data.R | tee best

N=$1

python main-torch.py \
    --A-path data/A-$N \
    --B-path data/B-$N \
    --P-path data/P-$N \
    --mode exact \
    --symmetric

python main-torch.py \
    --A-path data/A-$N \
    --B-path data/B-$N \
    --P-path data/P-$N \
    --mode "auction" \
    --eps 1000 \
    --cuda \
    --symmetric
