
#!/bin/bash

# benchmark.sh

# Notes:
#   !! Using `--num-seeds` instead of passing a P matrix means that we start
#       at a vertex of the Birkhoff polytope.  This is probably a little non-standard
#       but appears to work, and allows us to exploit sparsity.

# --
# synthetic w/ p=0.5

# Rscript utils/make-correlated-gnp.R 0 0.5 0.7 # dense; p=0.5; rho=0.7

DENSE_DATASET="_data/synthetic/dense/0.5/5000/"

# too dense to use sparse matrices
# python main-scipy.py ...

python main-torch.py \
    --A-path $DENSE_DATASET/A \
    --B-path $DENSE_DATASET/B \
    --num-seeds 100 \
    --lap-mode jv

python main-torch.py \
    --A-path $DENSE_DATASET/A \
    --B-path $DENSE_DATASET/B \
    --num-seeds 100 \
    --lap-mode jv \
    --cuda

python main-torch.py \
    --A-path $DENSE_DATASET/A \
    --B-path $DENSE_DATASET/B \
    --num-seeds 100 \
    --lap-mode auction \
    --cuda


# {jv,auction}
# {cuda,no_cuda}
# {eps}

# --
# synthetic w/ p=0.05

# Rscript utils/make-correlated-gnp.R 0 0.05 0.7 # dense; p=0.05; rho=0.7
# Rscript utils/make-correlated-gnp.R 1 0.05 0.7 # sparse; p=0.05; rho=0.7

SPARSE_DATASET="_data/synthetic/sparse/0.05/2000/"
DENSE_DATASET="_data/synthetic/dense/0.05/2000/"

python main-torch.py \
    --A-path $SPARSE_DATASET/A \
    --B-path $SPARSE_DATASET/B \
    --num-seeds 100 \
    --lap-mode jv \
    --cuda

python main-torch.py \
    --A-path $SPARSE_DATASET/A \
    --B-path $SPARSE_DATASET/B \
    --num-seeds 100 \
    --lap-mode auction \
    --cuda \
    --eps 100

python main-torch.py \
    --A-path $SPARSE_DATASET/A \
    --B-path $SPARSE_DATASET/B \
    --num-seeds 100 \
    --lap-mode greedy \
    --cuda



# python main-torch.py \
#     --A-path $DENSE_DATASET/A \
#     --B-path $DENSE_DATASET/B \
#     --num-seeds 100 \
#     --lap-mode auction \
#     --cuda \
#     --auction-eps 100

# {scipy,torch}
# {jv,auction}
# {cuda,no_cuda}
# {eps}