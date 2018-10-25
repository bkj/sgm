#!/usr/bin/env python

"""
    export-problems.py
"""

import sys
import numpy as np
from scipy import sparse
from scipy.io import mmwrite

def load_data():
    print('kasios.py: loading', file=sys.stderr)
    edges = np.load('data/calls.npy')
    X = sparse.csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])))
    X = ((X + X.T) > 0).astype('float64')
    X.eliminate_zeros()
    
    rw = open('./data/calls.rw').read().splitlines()
    rw = np.array([int(xx) for xx in rw])
    
    return X, rw

def make_problem(X, rw, num_nodes, num_seeds, shuffle_A=False, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random
    
    node_sel = np.sort(rw[:num_nodes])
    
    A = X[node_sel][:,node_sel].copy()
    
    # This means that seeds are picked randomly
    if shuffle_A:
        perm = rng.permutation(num_nodes)
        A = A[perm][:,perm]
    
    B = A.copy()
    
    perm = np.arange(num_nodes)
    perm[num_seeds:] = rng.permutation(perm[num_seeds:])
    B = B[perm][:,perm]
    
    P = sparse.eye(num_nodes).tocsr()
    P[num_seeds:, num_seeds:] = 0
    P.eliminate_zeros()
    
    return A, B, P


X, rw = load_data()

num_seeds = 100
all_num_nodes = 2 ** np.arange(10, 15)
for num_nodes in all_num_nodes:
    A, B, _ = make_problem(X, rw, num_nodes=num_nodes, num_seeds=num_seeds, shuffle_A=True, seed=123)
    mmwrite('./data/calls_A_%d_%d.mtx' % (num_nodes, num_seeds), A, symmetry='symmetric', field='pattern')
    mmwrite('./data/calls_B_%d_%d.mtx' % (num_nodes, num_seeds), B, symmetry='symmetric', field='pattern')


