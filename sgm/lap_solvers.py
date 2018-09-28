#!/usr/bin/env python

"""
    lap_solvers.py
"""

import sys
import numpy as np
from scipy import sparse
import torch

from lap import lapjv

try:
    sys.path.append('/home/bjohnson/projects/cuda_auction/python')
    from lap_auction import dense_lap_auction, csr_lap_auction, dot_auction
except:
    print('WARNING: sgm.lap_solvers cannot load `lap_auction`', file=sys.stderr)

def _gatagat_lapjv(cost):
    if isinstance(cost, torch.Tensor) or isinstance(cost, torch.cuda.FloatTensor):
        cost_ = cost.cpu().numpy()
    elif isinstance(cost, sparse.csr_matrix):
        cost_ = cost.toarray()
    elif isinstance(cost, np.ndarray):
        cost_ = cost
    else:
        print(type(cost))
        raise Exception('_gatagat_lapjv: cost has unknown type!')
    
    _, idx, _ = lapjv(cost_.max() - cost_)
    return idx

jv = _gatagat_lapjv