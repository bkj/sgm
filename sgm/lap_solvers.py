#!/usr/bin/env python

"""
    lap_solvers.py
"""

import sys
import numpy as np
from scipy import sparse

try:
    import torch
    have_torch = True
except:
    print('!! Could not import torch', file=sys.stderr)
    have_torch = False


from lap import lapjv as __lapjv_gatagat
from lapjv import lapjv as __lapjv_srcd

try:
    sys.path.append('/home/bjohnson/projects/cuda_auction/python')
    from lap_auction import dense_lap_auction, csr_lap_auction, dot_auction
except:
    print('WARNING: sgm.lap_solvers cannot load `lap_auction`', file=sys.stderr)

def jv(cost, jv_backend):
    if have_torch and (isinstance(cost, torch.Tensor) or isinstance(cost, torch.cuda.FloatTensor)):
        cost_ = cost.cpu().numpy()
    elif isinstance(cost, sparse.csr_matrix):
        cost_ = cost.toarray()
    elif isinstance(cost, np.ndarray):
        cost_ = cost
    else:
        print(type(cost))
        raise Exception('_gatagat_lapjv: cost has unknown type!')
    
    if jv_backend == 'gatagat':
        _, idx, _ = __lapjv_gatagat(cost_.max() - cost_)
    elif jv_backend == 'srcd':
        idx, _, _ = __lapjv_srcd(cost_.max() - cost_)
    else:
        raise Exception('ERROR: sgm.lap_solvers: unknown jv_backend=%s' % jv_backend)
    
    return idx