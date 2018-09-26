#!/usr/bin/env python

"""
    sgm.py
"""

import warnings
warnings.filterwarnings("ignore", module="matplotlib")

import sys
import json
import argparse
import numpy as np
import pandas as pd
from time import time
from functools import partial

from scipy import sparse

from lap import lapjv

from sgm import BaseSGM, TruncatedSGM

from hashlib import md5

# --
# Helpers

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data params
    parser.add_argument('--A-path', type=str, default='_data/synthetic/dense/0.5/1000/A.csv')
    parser.add_argument('--B-path', type=str, default='_data/synthetic/dense/0.5/1000/B.csv')
    parser.add_argument('--P-path', type=str)
    parser.add_argument('--num-seeds', type=int)
    
    # SGM params
    parser.add_argument('--num-iters', type=int, default=20)
    parser.add_argument('--tolerance', type=int, default=1)
    
    # LAP params
    parser.add_argument('--lap-mode', type=str, default='jv', choices=['auction', 'jv'])
    parser.add_argument('--auction-eps', type=float, default=100)
    
    # Misc params
    parser.add_argument('--cuda', action="store_true")
    
    args = parser.parse_args()
    assert (args.P_path is not None) or (args.num_seeds is not None)
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


class ScipySparseSGM(BaseSGM):
    def solve_lap(self, cost):
        # cost_all = cost.copy()
        
        if isinstance(cost, sparse.csr_matrix):
            cost = cost.toarray()
        
        cost = cost - cost.min()
        _, idx, _ = lapjv(cost.max() - cost)
        
        # print("score", cost_all[(np.arange(cost.shape[0]), idx)].sum())
        # print(md5(str(idx).encode()).hexdigest())
        
        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(cost.shape[0]), idx)))
        
    def compute_grad(self, A, P, B):
        AP = A.dot(P)
        out = 4 * AP.dot(B) - 2 * AP.sum(axis=1) - 2 * B.sum(axis=0) + A.shape[0]
        out = np.asarray(out)
        return out
        
    def prod_sum(self, x, y):
        return y.multiply(x).sum()

class ScipyTruncatedSGM(TruncatedSGM):
    def solve_lap(self, cost):
        if isinstance(cost, sparse.csr_matrix):
            cost = cost.toarray()
        
        cost = cost - cost.min()
        _, idx, _ = lapjv(cost.max() - cost)
        
        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(cost.shape[0]), idx)))
    
    def solve_lap_fused(self, A, P, B):
        cost = A.dot(P).dot(B)
        
        if isinstance(cost, sparse.csr_matrix):
            cost = cost.toarray()
        
        cost = cost - cost.min()
        _, idx, _ = lapjv(cost.max() - cost)
        
        # cost_all = self.compute_grad(A, P, B)
        # print("score", cost_all[(np.arange(cost.shape[0]), idx)].sum())
        # print(md5(str(idx).encode()).hexdigest())
        
        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(cost.shape[0]), idx)))
        
    def sparse_trace(self, A, X, B, Y):
        AX  = A.dot(X)
        YBt = Y.dot(B.T)
        
        AX_sum = Y.dot(AX.sum(axis=1)).sum()
        B_sum  = Y.T.dot(B.sum(axis=0).T).sum()
        
        return 4 * AX.multiply(YBt).sum() + A.shape[0] * Y.sum() - 2 * (AX_sum + B_sum)

# SGMClass = ScipySparseSGM
# print('ScipySparseSGM')

SGMClass = ScipyTruncatedSGM
print('ScipyTruncatedSGM')

args = parse_args()
np.random.seed(123)

# --
# IO

t = time()

A = load_matrix(args.A_path)
B = load_matrix(args.B_path, shape=A.shape[0])

f_actual = np.sqrt(((A.toarray() - B.toarray()) ** 2).sum())

if args.P_path:
    P = load_matrix(args.P_path, shape=A.shape[0])
    num_seeds = (P.diagonal() == 1).sum()
else:
    num_seeds = args.num_seeds
    
    P = sparse.eye(A.shape[0]).tocsr()
    P.data[num_seeds:] = 0
    P.eliminate_zeros()
    
    perm = np.arange(P.shape[0])
    perm[num_seeds:] = np.random.permutation(perm[num_seeds:])
    A = A[perm][:,perm]

io_time = time() - t
print('loaded in %fs' % io_time, file=sys.stderr)

assert (A != A.T).sum() == 0
assert (B != B.T).sum() == 0

min_nodes, max_nodes = sorted([A.shape[0], B.shape[0]])
f_orig = np.sqrt(((A[:min_nodes,:min_nodes].toarray() - B[:min_nodes,:min_nodes].toarray()) ** 2).sum())

# --
# Run

start_time = time() 
P_out = SGMClass().run(
    A=A,
    P=P,
    B=B,
    num_iters=args.num_iters,
    tolerance=args.tolerance,
    verbose=True
)
compute_time = time() - start_time

# --
# Compute results

P_out_small = P_out[:min_nodes,:min_nodes]
B_perm      = P_out_small.dot(B).dot(P_out_small.T)
f_perm      = np.sqrt(((A[:min_nodes,:min_nodes].toarray() - B_perm[:min_nodes,:min_nodes].toarray()) ** 2).sum())

print(json.dumps({
    "f_actual"     : float(f_actual),
    "f_orig"       : float(f_orig),
    "f_perm"       : float(f_perm),
    
    "io_time"      : float(io_time),
    "compute_time" : float(compute_time),
    
    "lap_mode"     : args.lap_mode,
    "auction_eps"  : args.auction_eps if args.lap_mode == 'auction' else None,
    "max_nodes"    : int(max_nodes),
    "num_seeds"    : int(num_seeds),
}))
