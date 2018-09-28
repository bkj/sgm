#!/usr/bin/env python

"""
    sgm/examples/kasios/kasios.py
"""

import warnings
warnings.filterwarnings("ignore", module="scipy.sparse")

import sys
import json
import argparse
import numpy as np
from time import time

from scipy import sparse

from sgm import factory as sgm_factory

# --
# Helpers

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

def run_experiment(X, rw, backend, num_nodes, num_seeds, seed, jv_backend, verbose):
    # Make datasets
    A, B, P = make_problem(
        X=X, 
        rw=rw, 
        num_nodes=num_nodes, 
        num_seeds=num_seeds,
        shuffle_A=True,
        seed=seed + 111,
    )
    
    # Create solver
    SGMClass = sgm_factory(*backend.split('.'))
    if 'jv' in backend:
        sgm = SGMClass(A=A, B=B, P=P, jv_backend=jv_backend, verbose=verbose)
    else:
        sgm = SGMClass(A=A, B=B, P=P, verbose=verbose)
    
    # --
    # Run
    
    start_time = time() 
    P_out      = sgm.run(num_iters=20, tolerance=1)
    total_time = time() - start_time
    
    P_out  = sparse.csr_matrix((np.ones(num_nodes), (np.arange(num_nodes), P_out)))
    B_perm = P_out.dot(B).dot(P_out.T)
    
    return {
        "backend"    : str(backend),
        "num_nodes"  : int(num_nodes),
        "num_seeds"  : int(num_seeds),
        "total_time" : float(total_time),
        "dist_orig"  : float((A.toarray() != B.toarray()).sum()),
        "dist_perm"  : float((A.toarray() != B_perm.toarray()).sum()),
    }, sgm.lap_times, sgm.iter_times

# --
# IO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='scipy.classic.jv')
    parser.add_argument('--jv-backend', type=str, default='gatagat', choices=['srcd', 'gatagat'])
    parser.add_argument('--num-nodes', type=int, default=1000)
    parser.add_argument('--num-seeds', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    
    X, rw = load_data()
    
    info, _, _ = run_experiment(
        X=X,
        rw=rw,
        backend=args.backend,
        num_nodes=args.num_nodes,
        num_seeds=args.num_seeds,
        seed=args.seed,
        jv_backend=args.jv_backend,
        verbose=True
    )
    
    print(json.dumps(info))



