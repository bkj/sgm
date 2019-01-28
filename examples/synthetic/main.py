#!/usr/bin/env python

"""
    examples/synthetic/main.py
"""

import sys
import numpy as np
from scipy import sparse

from sgm import JVSparseSGM

def make_perm(num_nodes, num_seeds):
    P = sparse.eye(num_nodes).tocsr()
    
    perm = np.arange(num_nodes)
    perm[num_seeds:] = np.random.permutation(perm[num_seeds:])
    
    return P[perm]


def make_init(num_nodes, num_seeds):
    P = sparse.csr_matrix((num_nodes, num_nodes))
    P[:num_seeds,:num_seeds] = sparse.eye(num_seeds)
    return P


# --
# Create data

num_nodes = 1024
num_seeds = 32

# Random symmetric matrix
A = sparse.random(num_nodes, num_nodes, density=0.01)
A = ((A + A.T) > 0).astype(np.float32)

# Random permutation matrix that keeps first `num_seeds` nodes the same
P_act = make_perm(num_nodes=num_nodes, num_seeds=num_seeds)

# Permute A according to P_act
B = P_act @ A @ P_act.T

assert (A[:num_nodes,:num_nodes] != B[:num_nodes, :num_nodes]).sum() > 0 
assert (A[:num_seeds,:num_seeds] != B[:num_seeds, :num_seeds]).sum() == 0

# --
# Run SGM

P_init = make_init(num_nodes=num_nodes, num_seeds=num_seeds)

sgm = JVSparseSGM(A=A, B=B, P=P_init, verbose=True)

node_map = sgm.run(num_iters=20, tolerance=1)

# --
# Check number of disagreements after SGM

P_out = sparse.csr_matrix((np.ones(num_nodes), (np.arange(num_nodes), node_map)))

B_perm = P_out @ B @ P_out.T

num_disagreements = (A[:num_nodes,:num_nodes] != B_perm[:num_nodes, :num_nodes]).sum()
print('num_disagreements=%d' % num_disagreements, file=sys.stderr)
