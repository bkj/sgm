#!/usr/bin/env python

"""
    base_sgm.py
"""

import sys
import json
from time import time

try:
    import torch
except:
    print('!! Could not import torch', file=sys.stderr)

from .utils import sparse2torch

# --
# Base SGM class

class _BaseSGM:
    def __init__(self, A, B, P, verbose=True):
        self.A = A
        self.B = B
        self.P = P
        
        self.verbose = verbose
    
    def _reset_timers(self):
        self.lap_times   = []
        self.iter_times  = []
    
    def _log_times(self):
        if self.verbose:
            print(json.dumps({
                "iter"      : len(self.lap_times),
                "lap_time"  : float(self.lap_times[-1]),
                "iter_time" : float(self.iter_times[-1]),
            }))
    
    def check_convergence(self, c, d, e, tolerance):
        cde = c + e - d 
        d2e = d - 2 * e
        
        if (cde == 0) and (d2e == 0):
            alpha = 0
            falpha = -1 # NA value
        else:
            if (cde == 0):
                alpha  = float('inf')
                falpha = -1 # NA value
            else:
                alpha = -d2e / (2 * cde)
                falpha = cde * alpha ** 2 + d2e * alpha
        
        f1 = c - e
        
        if (alpha > 0) and (alpha < tolerance) and (falpha > max(0, f1)):
            return alpha, False # P <- (alpha * P) + (1 - alpha) * T
        elif f1 < 0:
            return None, False # P <- T
        else:
            return None, True  # stop


class _JVMixin:
    def __init__(self, *args, jv_backend='gatagat', **kwargs):
        # print('jv_backend=%s' % jv_backend, file=sys.stderr)
        self.jv_backend = jv_backend
        super().__init__(*args, **kwargs)


class _TorchMixin:
    def __init__(self, A, B, P, cuda=True, **kwargs):
        self.cuda = cuda
        if not isinstance(A, torch.Tensor):
            A = sparse2torch(A, cuda=cuda)
            B = sparse2torch(B, cuda=cuda)
            P = sparse2torch(P, cuda=cuda)
        
        super().__init__(A=A, B=B, P=P, **kwargs)