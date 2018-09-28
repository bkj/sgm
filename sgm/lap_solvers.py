#!/usr/bin/env python

"""
    lap_solvers.py
"""

import sys
import numpy as np
from scipy import sparse
import torch

from lap import lapjv

sys.path.append('/home/bjohnson/projects/cuda_auction/python')
from lap_auction import dense_lap_auction, csr_lap_auction, dot_auction

def _gatagat_lapjv(cost):
    if isinstance(cost, torch.Tensor):
        cost_ = cost.cpu().numpy()
    elif isinstance(cost, sparse.csr_matrix):
        cost_ = cost.toarray()
    elif isinstance(cost, np.ndarray):
        cost_ = cost
    else:
        raise Exception('_gatagat_lapjv: cost has unknown type!')
    
    _, idx, _ = lapjv(cost_.max() - cost_)
    return idx

jv = _gatagat_lapjv