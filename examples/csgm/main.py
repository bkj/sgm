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

num_seeds = 10
A = mmread('/home/bjohnson/projects/davis/csgm/data/A.mtx').tocsr()
B = mmread('/home/bjohnson/projects/davis/csgm/data/B.mtx').tocsr()

num_nodes = A.shape[0]

P = sparse.eye(A.shape[0]).tocsr()
P.data[num_seeds:] = 0
P.eliminate_zeros()

SGMClass = factory(*'scipy.sparse.auction'.split('.'))
sgm      = SGMClass(A=A, B=B, P=P, verbose=True)
t = time()
P_out    = sgm.run(num_iters=20, tolerance=1)
elapsed = time() - t

P_out  = sparse.csr_matrix((np.ones(num_nodes), (np.arange(num_nodes), P_out)))
B_perm = P_out.dot(B).dot(P_out.T)

print('final distance', float((A.toarray() != B_perm.toarray()).sum()))
print('time', elapsed)