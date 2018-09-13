#!/bin/bash

# run-synthetic.sh

# Make data
mkdir -p _data/synthetic/{sparse,dense}
Rscript utils/make-correlated-gnp.R

# Run (sparse input)
INPATH="_data/synthetic/sparse/0.05/5000/"
time python main-scipy.py \
    --A-path $INPATH/A.edgelist \
    --B-path $INPATH/B.edgelist \
    --P-path $INPATH/P.edgelist \
    --mode exact \
    --symmetric

# Run (dense input)
INPATH="_data/synthetic/dense/0.05/5000/"
time python main-scipy.py \
    --A-path $INPATH/A.csv \
    --B-path $INPATH/B.csv \
    --P-path $INPATH/P.csv \
    --mode exact \
    --symmetric

# Sparse/dense input should produce the same output,
# but sparse IO will be much better