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
from scipy.io import mmread

from sgm import factory

# num_seeds = 100

# path = "/home/bjohnson/projects/davis/csgm/data"
path = "/home/bjohnson/projects/davis/csgm/data/connectome/DS01216/sparse/"
A = mmread('%s/A.mtx' % path).tocsr()
B = mmread('%s/B.mtx' % path).tocsr()

num_nodes = A.shape[0]

P = sparse.eye(A.shape[0]).tocsr()
# P.data[num_seeds:] = 0
P.eliminate_zeros()

# --

sgm_mode = 'scipy.classic.jv'
print(sgm_mode)
# SGMClass = factory(*'scipy.sparse.auction'.split('.'))
# SGMClass = factory(*'scipy.sparse.jv'.split('.'))
SGMClass = factory(*sgm_mode.split('.'))

sgm      = SGMClass(A=A, B=B, P=P, verbose=True)
t = time()
P_out    = sgm.run(num_iters=20, tolerance=1)
elapsed = time() - t

P_out  = sparse.csr_matrix((np.ones(num_nodes), (np.arange(num_nodes), P_out)))
B_perm = P_out.dot(B).dot(P_out.T)

print('orig distance', float((A.toarray() != B.toarray()).sum()))
print('final distance', float((A.toarray() != B_perm.toarray()).sum()))
print('time', elapsed)

# --
print('-' * 50)
sgm_mode = 'scipy.sparse.jv'
print(sgm_mode)
SGMClass = factory(*sgm_mode.split('.'))

sgm      = SGMClass(A=A, B=B, P=P, verbose=True)
t = time()
P_out    = sgm.run(num_iters=20, tolerance=1)
elapsed = time() - t

P_out  = sparse.csr_matrix((np.ones(num_nodes), (np.arange(num_nodes), P_out)))
B_perm = P_out.dot(B).dot(P_out.T)

print('orig distance', float((A.toarray() != B.toarray()).sum()))
print('final distance', float((A.toarray() != B_perm.toarray()).sum()))
print('time', elapsed)
