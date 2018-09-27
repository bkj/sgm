#!/usr/bin/env python

"""
    backends.py
"""

import numpy as np
from lap import lapjv
from hashlib import md5
from scipy import sparse
from time import time

from sgm import BaseSGMClassic, BaseSGMSparse, BaseSGMFused

import sys
sys.path.append('/home/bjohnson/projects/cuda_auction/python')
from lap_auction import csr_lap_auction, dot_auction

# --
# Helpers

def _augment_cost(cost, mode='random'):
    if mode == 'sequential':
        cols = np.arange(cost.shape[0]).reshape(-1, cost.shape[0])
        rows = cost.shape[0] * np.arange(cost.shape[0]).reshape(cost.shape[0], -1)
    elif mode == 'random':
        cols = np.random.choice(100000, cost.shape[0]).reshape(-1, cost.shape[0])
        rows = np.random.choice(100000, cost.shape[0]).reshape(cost.shape[0], -1)
    else:
        raise Exception
    
    return cost + cols + rows

def _lapjv(cost, augment):
    if isinstance(cost, sparse.csr_matrix):
        cost = cost.toarray()
    
    cost_ = _augment_cost(cost) if augment else cost
    cost_ = cost_.max() - cost_
    
    _, idx, _ = lapjv(cost_)
    return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(cost.shape[0]), idx)))

# --
# Classic Backends

class _ScipySGMClassic(BaseSGMClassic):
    def compute_grad(self, A, P, B):
        AP = A.dot(P)
        sparse_part = 4 * AP.dot(B) 
        dense_part  = - 2 * AP.sum(axis=1) - 2 * B.sum(axis=0) + A.shape[0]
        return np.asarray(sparse_part + dense_part)
        
    def compute_trace(self, x, y):
        return y.multiply(x).sum()


class JVClassicSGM(_ScipySGMClassic):
    def solve_lap(self, cost, augment=True):
        return _lapjv(cost, augment)


class AuctionClassicSGM(_ScipySGMClassic):
    def solve_lap(self, cost, augment=True):
        idx = csr_lap_auction(cost,
            verbose=True,
            num_runs=1,
            auction_max_eps=1.0,
            auction_min_eps=1.0,
            auction_factor=0.0
        )
        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(idx.shape[0]), idx)))

# --
# Sparse backends

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
    def solve_lap(self, cost, augment=True):
        return _lapjv(cost, augment)


class AuctionSparseSGM(_ScipySGMSparse):
    def solve_lap(self, cost, augment=True):
        idx = csr_lap_auction(cost,
            verbose=True,
            num_runs=1,
            auction_max_eps=1.0,
            auction_min_eps=1.0,
            auction_factor=0.0
        )
        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(idx.shape[0]), idx)))


# --
# Fused backends

class _ScipyFusedSGM(BaseSGMFused):
    def _warmup(self):
        x = sparse.random(100, 100, density=0.5).tocsr()
        y = sparse.random(100, 100, density=0.5).tocsr()
        _ = self.solve_lap_fused(x, y, verbose=False)
    
    def compute_trace(self, AX, B, Y):
        YBt = Y.dot(B.T)
        
        AX_sum = Y.dot(AX.sum(axis=1)).sum()
        B_sum  = Y.T.dot(B.sum(axis=0).T).sum()
        
        return 4 * AX.multiply(YBt).sum() + AX.shape[0] * Y.sum() - 2 * (AX_sum + B_sum)


class JVFusedSGM(_ScipyFusedSGM):
    def solve_lap_exact(self, cost, augment=True):
        return _lapjv(cost, augment)
    
    def solve_lap_fused(self, AP, B, augment=True, verbose=True):
        return _lapjv(AP.dot(B), augment)


class AuctionFusedSGM(_ScipyFusedSGM):
    def solve_lap_exact(self, cost, augment=True):
        return _lapjv(cost, augment)
    
    def solve_lap_fused(self, AP, B, augment=True, verbose=True):
        idx = dot_auction(AP, B, AP.shape[0], verbose=verbose)
        return sparse.csr_matrix((np.ones(AP.shape[0]), (np.arange(idx.shape[0]), idx)))
