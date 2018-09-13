#!/bin/bash

# run-connectome.sh

for dataset in $(ls _data/connectome); do
    echo $dataset
    INPATH="_data/connectome/$dataset"
    Rscript utils/prep-connectome.R $INPATH $INPATH/sparse 1 # sparse
    Rscript utils/prep-connectome.R $INPATH $INPATH/dense 0 # dense
done


# INPATH="_data/connectome/DS03231/sparse"
INPATH="_data/connectome/DS06481/sparse"
time python main-scipy.py \
    --A-path $INPATH/B.edgelist \
    --B-path $INPATH/A.edgelist \
    --P-path $INPATH/P.edgelist \
    --mode exact \
    --symmetric


INPATH="_data/connectome/DS00833/dense"
time python main-scipy.py \
    --A-path $INPATH/A2.ordered.edges \
    --B-path $INPATH/A1.ordered.edges \
    --P-path $INPATH/P_start.edges \
    --mode exact \
    --symmetric