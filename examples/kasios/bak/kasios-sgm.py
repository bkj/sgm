#!/usr/bin/env python

"""
    sgm.py
"""

import warnings
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module="scipy.sparse")

import sys
import json
import argparse
import numpy as np
import pandas as pd
from time import time

from scipy import sparse
import itertools
from concurrent.futures import ProcessPoolExecutor

from sgm import ScipySGM, ScipyFusedSGM, AuctionSGM, AuctionFusedSGM

_backends = {
    "scipy"         : ScipySGM,
    "scipy_fused"   : ScipyFusedSGM,
    "auction"       : AuctionSGM,
    "auction_fused" : AuctionFusedSGM,
}

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='./data/calls.npy')
    parser.add_argument('--backend', type=str, default='scipy', choices=_backends.keys())
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

# --
# Helpers

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

def run_experiment(params):
    print(params, file=sys.stderr)
    num_nodes, num_seeds = params
    A, B, P = make_problem(
        X=X, 
        rw=rw, 
        num_nodes=num_nodes, 
        num_seeds=num_seeds,
        shuffle_A=True,
        seed=123,
    )
    
    start_time = time() 
    sgm = SGMClass()
    P_out = sgm.run(
        A=A,
        P=P,
        B=B,
        num_iters=20,
        tolerance=1
    )
    sgm_time = time() - start_time
    
    f_orig = np.sqrt(((A.toarray() - B.toarray()) ** 2).sum())
    
    B_perm = P_out.dot(B).dot(P_out.T)
    f_perm = np.sqrt(((A.toarray() - B_perm.toarray()) ** 2).sum())
    f_perm_abs = (A.toarray() != B_perm.toarray()).sum()
    
    return {
        "num_nodes"  : int(num_nodes),
        "num_seeds"  : int(num_seeds),
        "sgm_time"   : float(sgm_time),
        "f_orig"     : float(f_orig),
        "f_perm"     : float(f_perm),
        "f_perm_abs" : float(f_perm_abs),
        "times" : {
            "lap"   : sgm.lap_times,
            "grad"  : sgm.grad_times,
            "check" : sgm.check_times,
        }
    }

# --
# IO

args = parse_args()
SGMClass = _backends[args.backend]
np.random.seed(args.seed)

edges = np.load(args.inpath)
X = sparse.csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])))
X = ((X + X.T) > 0).astype('float64')
X.eliminate_zeros()

rw = open('./data/calls.rw').read().splitlines()
rw = np.array([int(xx) for xx in rw])

# >>
print(run_experiment((6000, 32)))
# <<

# --
# Run a set of problems

# num_node_step = 100
# all_num_nodes = np.arange(num_node_step, 21 * num_node_step, num_node_step)

# num_seed_step = 10
# all_num_seeds = np.arange(num_seed_step, 11 * num_seed_step, num_seed_step)

# params = list(itertools.product(all_num_nodes, all_num_seeds))
# params = np.random.permutation(params)
# params = [tuple(p) for p in params]

# with ProcessPoolExecutor(max_workers=16) as ex:
#     for res in ex.map(run_experiment, params):
#         print(json.dumps(res))
#         sys.stdout.flush()

# for num_nodes in all_num_nodes:
#     for num_seeds in all_num_seeds:
#         run_experiment(num_nodes=num_nodes, num_seeds=num_seeds)