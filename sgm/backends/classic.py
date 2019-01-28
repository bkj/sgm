#!/usr/bin/env python

"""
    backends/classic.py
"""

import sys
from time import time
from ..common import _BaseSGM, _TorchMixin, _JVMixin
from .. import lap_solvers

import numpy as np
from scipy import sparse

try:
    import torch
except:
    print('!! Could not import torch', file=sys.stderr)


# --
# Core SGM loop

class BaseSGMClassic(_BaseSGM):
    def run(self, num_iters, tolerance, verbose=True):
        A, B, P = self.A, self.B, self.P
        if hasattr(self, '_warmup'):
            self._warmup(A, P, B)
        
        self._reset_timers()
        
        grad = self.compute_grad(A, P, B)
        
        for i in range(num_iters):
            iter_t = time()
            
            lap_t = time()
            T = self.solve_lap(grad)
            self.lap_times.append(time() - lap_t)
            
            gradt = self.compute_grad(A, T, B)
            
            ps_grad_P  = self.compute_trace(grad, P)
            ps_grad_T  = self.compute_trace(grad, T)
            ps_gradt_P = self.compute_trace(gradt, P)
            ps_gradt_T = self.compute_trace(gradt, T)
            
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
                else:
                    P    = T
                    grad = gradt
            
            self.iter_times.append(time() - iter_t)
            if verbose:
                self._log_times()
            
            if stop:
                break
        
        return self.solve_lap(P, final=True)

# --
# Scipy backends

class _ScipySGMClassic(BaseSGMClassic):
    def compute_grad(self, A, P, B):
        AP = A.dot(P)
        sparse_part = 4 * AP.dot(B) 
        dense_part  = - 2 * AP.sum(axis=1) - 2 * B.sum(axis=0) + A.shape[0]
        return np.asarray(sparse_part + dense_part)
        
    def compute_trace(self, x, y):
        return y.multiply(x).sum()


class ScipyJVClassicSGM(_JVMixin, _ScipySGMClassic):
    def solve_lap(self, cost, final=False):
        idx = lap_solvers.jv(cost, jv_backend=self.jv_backend)
        if final:
            return idx
        
        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(cost.shape[0]), idx)))


class ScipyAuctionClassicSGM(_ScipySGMClassic):
    def solve_lap(self, cost):
        raise NotImplemented
        # idx = lap_solvers.dense_lap_auction(cost,
        #     verbose=verbose,
        #     num_runs=1,
        #     auction_max_eps=1.0,
        #     auction_min_eps=1.0,
        #     auction_factor=0.0
        # )
        # return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(idx.shape[0]), idx)))

# --
# Torch backends




class _TorchSGMClassic(_TorchMixin, BaseSGMClassic):
    def _warmup(self, A, P, B):
        self.eye = torch.eye(A.shape[0])
        if self.cuda:
            self.eye = self.eye.cuda()
    
    def compute_grad(self, A, P, B):
        AP = torch.mm(A, P)
        sparse_part = 4 * torch.mm(AP, B)
        dense_part  = - 2 * AP.sum(dim=-1).view(-1, 1) - 2 * B.sum(dim=0).view(1, -1) + A.size(0)
        return sparse_part + dense_part
    
    def compute_trace(self, x, y):
        return (x * y).sum()


class TorchJVClassicSGM(_JVMixin, _TorchSGMClassic):
    def solve_lap(self, cost, final=False):
        idx = lap_solvers.jv(cost, jv_backend=self.jv_backend)
        if final:
            return idx
        
        idx = idx.astype(np.int32)
        idx = torch.LongTensor(idx)
        if self.cuda:
            idx = idx.cuda()
        
        return self.eye[idx]


class TorchAuctionClassicSGM(_TorchSGMClassic):
    def solve_lap(self, cost, verbose=False):
        raise NotImplemented
