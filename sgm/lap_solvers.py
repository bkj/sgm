#!/usr/bin/env python

"""
    lap_solvers.py
"""

import sys
import numpy as np
from scipy import sparse

from lap import lapjv

sys.path.append('/home/bjohnson/projects/cuda_auction/python')
from lap_auction import dense_lap_auction, csr_lap_auction, dot_auction

def _gatagat_lapjv(cost):
    if isinstance(cost, sparse.csr_matrix):
        cost = cost.toarray()
    
    _, idx, _ = lapjv(cost.max() - cost)
    return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(cost.shape[0]), idx)))

jv = _gatagat_lapjv