#!/usr/bin/env python

"""
    backends/fused.py
"""

from time import time
from ..common import _BaseSGM, _JVMixin
from .. import lap_solvers

import numpy as np
from scipy import sparse

# --
# SGM loop

class BaseSGMFused(_BaseSGM):
    def run(self, num_iters, tolerance, verbose=True):
        A, B, P = self.A, self.B, self.P
        if hasattr(self, '_warmup'):
            self._warmup()
        
        self._reset_timers()
        
        
        AP = A.dot(P)
        
        for i in range(num_iters):
            iter_t = time()
            
            lap_t = time()
            T = self.solve_lap_fused(AP, B)
            self.lap_times.append(time() - lap_t)
            
            AT = A.dot(T)
            
            ps_grad_P  = self.compute_trace(AP, B, P)
            ps_grad_T  = self.compute_trace(AP, B, T)
            ps_gradt_P = self.compute_trace(AT, B, P)
            ps_gradt_T = self.compute_trace(AT, B, T)
            
            alpha, stop = self.check_convergence(
                c=ps_grad_P,
                d=ps_gradt_P + ps_grad_T,
                e=ps_gradt_T,
                tolerance=tolerance,
            )
            
            if not stop:
                if alpha is not None:
                    P  = (alpha * P)  + (1 - alpha) * T
                    AP = (alpha * AP) + (1 - alpha) * AT
                else:
                    P  = T
                    AP = AT
            
            self.iter_times.append(time() - iter_t)
            if verbose:
                self._log_times()
            
            if stop:
                break
        
        return self.solve_lap_final(P)

# --

class _ScipyFusedSGM(_JVMixin, BaseSGMFused):
    def _warmup(self):
        x = sparse.random(100, 100, density=0.5).tocsr()
        y = sparse.random(100, 100, density=0.5).tocsr()
        _ = self.solve_lap_fused(x, y, verbose=False)
    
    def compute_trace(self, AX, B, Y):
        YBt = Y.dot(B.T)
        
        AX_sum = Y.dot(AX.sum(axis=1)).sum()
        B_sum  = Y.T.dot(B.sum(axis=0).T).sum()
        
        return 4 * AX.multiply(YBt).sum() + AX.shape[0] * Y.sum() - 2 * (AX_sum + B_sum)
        
    def solve_lap_final(self, cost):
        return lap_solvers.jv(cost, jv_backend=self.jv_backend)


class JVFusedSGM(_ScipyFusedSGM):
    def solve_lap_fused(self, AP, B, verbose=True):
        rowcol_offsets = - 2 * AP.sum(axis=1) - 2 * B.sum(axis=0) + AP.shape[0]
        idx = lap_solvers.jv(
            AP.dot(B).toarray() + rowcol_offsets, 
            jv_backend=self.jv_backend
        )
        return sparse.csr_matrix((np.ones(AP.shape[0]), (np.arange(idx.shape[0]), idx)))


class AuctionFusedSGM(_ScipyFusedSGM):
    def solve_lap_fused(self, AP, B, verbose=False):
        idx = lap_solvers.dot_auction(AP, B, AP.shape[0], verbose=verbose)
        return sparse.csr_matrix((np.ones(AP.shape[0]), (np.arange(idx.shape[0]), idx)))
