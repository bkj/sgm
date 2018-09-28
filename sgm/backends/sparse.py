#!/usr/bin/env python

"""
    backends/sparse.py
"""

from time import time
from ..common import _BaseSGM
from .. import lap_solvers

class BaseSGMSparse(_BaseSGM):
    def run(self, A, P, B, num_iters, tolerance, verbose=True):
        if hasattr(self, '_warmup'):
            self._warmup()
            
        self._reset_timers()
        
        AP   = A.dot(P)
        grad = AP.dot(B)
        
        for i in range(num_iters):
            iter_t = time()
            
            lap_t = time()
            rowcol_offsets = - 2 * AP.sum(axis=1) - 2 * B.sum(axis=0) + A.shape[0]
            T = self.solve_lap(grad.toarray() + rowcol_offsets)
            self.lap_times.append(time() - lap_t)
            
            AT    = A.dot(T)
            gradt = AT.dot(B)
            
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
                    P    = (alpha * P)    + (1 - alpha) * T
                    grad = (alpha * grad) + (1 - alpha) * gradt
                    AP   = (alpha * AP)   + (1 - alpha) * AT
                else:
                    P    = T
                    grad = gradt
                    AP   = AT
            
            self.iter_times.append(time() - iter_t)
            if verbose:
                self._log_times()
            
            if stop:
                break
        
        P_out = self.solve_lap(P)
        return P_out

# --

class _ScipySGMSparse(BaseSGMSparse):
    def _warmup(self):
        cost = sparse.random(100, 100, density=0.5).tocsr()
        _ = self.solve_lap(cost)
    
    def compute_trace(self, AX, B, Y):
        YBt = Y.dot(B.T)
        
        AX_sum = Y.dot(AX.sum(axis=1)).sum()
        B_sum  = Y.T.dot(B.sum(axis=0).T).sum()
        
        return 4 * AX.multiply(YBt).sum() + AX.shape[0] * Y.sum() - 2 * (AX_sum + B_sum)


class JVSparseSGM(_ScipySGMSparse):
    def solve_lap(self, cost):
        return lap_solvers.jv(cost)


class AuctionSparseSGM(_ScipySGMSparse):
    def solve_lap(self, cost, verbose=False):
        idx = lap_solvers.csr_lap_auction(cost,
            verbose=verbose,
            num_runs=1,
            auction_max_eps=1.0,
            auction_min_eps=1.0,
            auction_factor=0.0
        )
        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(idx.shape[0]), idx)))
