#!/usr/bin/env python

"""
    sgm.py
"""

from __future__ import division, print_function

import warnings
warnings.filterwarnings("ignore", module="matplotlib")

import sys
import json
import argparse
import numpy as np
import pandas as pd
from time import time
from functools import partial

import torch
from scipy import sparse

from lap import lapjv
sys.path.append('auction-lap')
from auction_lap import auction_lap

from sgm import sgm

# --
# Helpers

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--A-path', type=str, default='data/connectome/DS01876/A1.ordered.csv')
    parser.add_argument('--B-path', type=str, default='data/connectome/DS01876/A2.ordered.csv')
    parser.add_argument('--P-path', type=str, default='data/connectome/DS01876/P_start.csv')
    parser.add_argument('--outpath', type=str, default='./_simple_corr.txt')
    
    # SGM params
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--num-iters', type=int, default=20)
    parser.add_argument('--tolerance', type=int, default=1)
    parser.add_argument('--symmetric', action="store_true")
    
    # LAP params
    parser.add_argument('--mode', type=str, default='exact')
    parser.add_argument('--eps', type=float, default=100)
    
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--no-double', action="store_true")
    
    args = parser.parse_args()
    assert args.m == 0, "m != 0 -- not implemented yet"
    return args


def load_matrix(path, shape=None):
    mat = pd.read_csv(path).values
    if mat.shape[1] == 2: # sparse matrix w/o weights
        rows, cols = mat.T
        data = np.ones(rows.shape[0])
    elif mat.shape[1] == 3: # sparse matrix w/ weights
        rows, cols, data = mat.T
    else: # dense matrix
        mat = mat.astype(np.float64)
        rows, cols = np.where(mat)
        data = mat[(rows, cols)]
    
    if shape is None:
        shape = max(rows.max(), cols.max()) + 1
    else:
        implied_shape = max(rows.max(), cols.max()) + 1
        if implied_shape > shape:
            raise Exception('implied_shape > shape: switch order of inputs')
    
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(shape, shape))
    mat = ((mat + mat.T) > 0).astype(np.float64) # symmetrize
    
    assert mat.shape[0] == mat.shape[1], "%s must be square" % path
    return mat

torch.set_default_tensor_type('torch.DoubleTensor')
def solve_lap(cost_sparse, cost_offset, mode, cuda, eps, eye):
    cost = cost_sparse + cost_offset
    
    if isinstance(cost, sparse.csr_matrix):
        cost = cost.toarray()
    
    cost = cost - cost.min() # Make >= 0
    
    if mode == 'exact':
        _, idx, _ = lapjv(cost.max() - cost)
    elif mode == 'auction':
        cost = torch.Tensor(cost).cuda()
        _, idx, _ = auction_lap(cost, eps=eps)
        idx = idx.cpu().numpy()
    
    return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(cost.shape[0]), idx)))

def compute_grad(A, P, B, sparse=False):
    AP  = A.dot(P)
    APB = AP.dot(B)
    
    sim          = 4 * APB
    row_offsets  = 2 * AP.sum(axis=1)
    col_offsets  = 2 * B.sum(axis=0)
    
    return sim, - (row_offsets + col_offsets)


args = parse_args()


# --
# IO

# args.A_path = '_data/synthetic/sparse/0.05/5000/A.edgelist'
# args.B_path = '_data/synthetic/sparse/0.05/5000/B.edgelist'

start_time = time()
A = load_matrix(args.A_path)
B = load_matrix(args.B_path, shape=A.shape[0])
# P = load_matrix(args.P_path, shape=A.shape[0])
P = sparse.eye(A.shape[0]).tocsr()
io_time = time() - start_time

# print('density(A) =', A.nnz / np.prod(A.shape), file=sys.stderr)
# print('density(B) =', B.nnz / np.prod(B.shape), file=sys.stderr)

assert (A != A.T).sum() == 0
assert (B != B.T).sum() == 0

# --
# Prep

n_seeds = (P.diagonal() == 1).sum()
# # >>
# P[n_seeds:, n_seeds:] = 0
# # <<
max_nodes = max([A.shape[0], B.shape[0]])
min_nodes = min([A.shape[0], B.shape[0]])

# --
# Run

start_time = time()
P_out = sgm(
    A=A,
    P=P,
    B=B,
    eye=None,
    compute_grad=compute_grad,
    solve_lap=partial(solve_lap, mode=args.mode, cuda=args.cuda, eps=args.eps),
    prod_sum=lambda d, s: s.multiply(d).sum(),
    num_iters=args.num_iters,
    tolerance=args.tolerance,
)
total_time = time() - start_time

# --
# Save results

P_out_small = P_out[:min_nodes,:min_nodes]

B_perm = P_out_small.dot(B).dot(P_out_small.T)

f_orig = np.sqrt(((A[:min_nodes,:min_nodes].toarray() - B[:min_nodes,:min_nodes].toarray()) ** 2).sum())
f_perm = np.sqrt(((A[:min_nodes,:min_nodes].toarray() - B_perm[:min_nodes,:min_nodes].toarray()) ** 2).sum())

print(json.dumps({
    "f_orig"     : float(f_orig),
    "f_perm"     : float(f_perm),
    
    "total_time" : float(total_time),
    "io_time"    : float(io_time),
    
    "mode"      : args.mode,
    "eps"       : args.eps,
    "max_nodes" : int(max_nodes),
    "n_seeds"   : int(n_seeds),
}))

# corr = P_out.nonzero().cpu().numpy() + 1
# np.savetxt(args.outpath, corr, fmt='%d')
