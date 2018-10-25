#!/usr/bin/env python

"""
    sgm/examples/csgm/connectome.py
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

# --
# IO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='scipy.sparse.jv')
    parser.add_argument('--dataset', type=str, default='DS00833')
    parser.add_argument('--num-nodes', type=int, default=1024)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    path = '/home/bjohnson/projects/davis/csgm/data/connectome'
    A = mmread('%s/%s/sparse/A.mtx' % (path,args.dataset)).tocsr()
    B = mmread('%s/%s/sparse/B.mtx' % (path,args.dataset)).tocsr()
    
    num_nodes = A.shape[0]
    
    P = sparse.eye(A.shape[0]).tocsr()
    P.eliminate_zeros()

    # --
    # Run

    SGMClass = factory(*args.backend.split('.'))

    sgm     = SGMClass(A=A, B=B, P=P, verbose=True)
    t       = time()
    P_out   = sgm.run(num_iters=20, tolerance=1)
    elapsed = time() - t

    P_out  = sparse.csr_matrix((np.ones(num_nodes), (np.arange(num_nodes), P_out)))
    B_perm = P_out.dot(B).dot(P_out.T)

    print('orig distance', float((A.toarray() != B.toarray()).sum()))
    print('final distance', float((A.toarray() != B_perm.toarray()).sum()))
    print('time', elapsed)
