#!/usr/bin/env python

"""
    backends.py
"""

import numpy as np
from lap import lapjv
from hashlib import md5
from scipy import sparse
from time import time

from sgm import BaseSGM, BaseFusedSGM, BaseSparseSGM

import sys
sys.path.append('/home/bjohnson/projects/cuda_auction/python')
from lap_auction import dense_lap_auction, csr_lap_auction, dot_auction

# --
# Helpers

def _augment_cost(cost):
    cols = np.arange(cost.shape[0]).reshape(-1, cost.shape[0])
    rows = cost.shape[0] * np.arange(cost.shape[0]).reshape(cost.shape[0], -1)
    return cost + cols + rows

def _lapjv(cost, augment):
    if isinstance(cost, sparse.csr_matrix):
        cost = cost.toarray()
    
    cost_ = _augment_cost(cost) if augment else cost
    cost_ = cost_.max() - cost_
    
    _, idx, _ = lapjv(cost_)
    return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(cost.shape[0]), idx)))

# --
# Backends

class ScipySGM(BaseSGM):
    """ Basic scipy SGM -- keeps some parts sparse"""
    def solve_lap(self, cost, augment=True):
        return _lapjv(cost, augment)
        
    def compute_grad(self, A, P, B):
        AP = A.dot(P)
        sparse_part = 4 * AP.dot(B) 
        dense_part  = - 2 * AP.sum(axis=1) - 2 * B.sum(axis=0) + A.shape[0]
        return sparse_part, dense_part
        
    def prod_sum(self, x, y):
        return y.multiply(x).sum()


class ScipyFusedSGM(BaseFusedSGM):
    """ 
        Fuses multiplication and LAP
        More sparse operations
    """
    def solve_lap(self, cost, augment=True):
        return _lapjv(cost, augment)
    
    def solve_lap_fused(self, A, P, B, augment=True):
        cost = A.dot(P).dot(B)
        return _lapjv(cost, augment)
        
    def sparse_trace(self, A, X, B, Y):
        AX  = A.dot(X)
        YBt = Y.dot(B.T)
        
        AX_sum = Y.dot(AX.sum(axis=1)).sum()
        B_sum  = Y.T.dot(B.sum(axis=0).T).sum()
        
        return 4 * AX.multiply(YBt).sum() + A.shape[0] * Y.sum() - 2 * (AX_sum + B_sum)

# >>

# rng = np.random.RandomState(seed=111222333)
# _ = dense_lap_auction(rng.choice(100, (10, 10)), 
#     verbose=False,
#     num_runs=1,
#     auction_max_eps=1..0,
#     auction_min_eps=1.0,
#     auction_factor=0.0
# )

class AuctionSGM(ScipySGM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter = 0
    
    def solve_lap(self, cost, augment=True):
        if self.iter < 4:
            print('csr_lap_auction')
            idx = csr_lap_auction(cost,
                verbose=True,
                num_runs=1,
                auction_max_eps=1.0,
                auction_min_eps=1.0,
                auction_factor=0.0
            )
            self.iter += 1
            return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(idx.shape[0]), idx)))
        else:
            print('lapjv')
            return _lapjv(cost, augment)



class AuctionSparseSGM(BaseSparseSGM):
    """
        Fuses multiplication and LAP
        More sparse operations
        Use auction algorithm
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter = 0
        
    def sparse_trace(self, A, X, B, Y):
        AX  = A.dot(X)
        YBt = Y.dot(B.T)
        
        AX_sum = Y.dot(AX.sum(axis=1)).sum()
        B_sum  = Y.T.dot(B.sum(axis=0).T).sum()
        
        return 4 * AX.multiply(YBt).sum() + A.shape[0] * Y.sum() - 2 * (AX_sum + B_sum)
    
    def solve_lap(self, cost, augment=True):
        t = time()
        idx = csr_lap_auction(cost,
            verbose=True,
            num_runs=1,
            auction_max_eps=1.0,
            auction_min_eps=1.0,
            auction_factor=0.0
        )
        # print('solve_lap_fused: csr_lap_auction', time() - t)
        self.iter += 1
        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(idx.shape[0]), idx)))


# --
# Fused

class ScipyFusedSGM(BaseFusedSGM):
    """ 
        Fuses multiplication and LAP
        More sparse operations
    """
    def solve_lap(self, cost, augment=True):
        return _lapjv(cost, augment)
    
    def solve_lap_fused(self, AP, B, augment=True):
        cost = AP.dot(B)
        return _lapjv(cost, augment)
        
    def sparse_trace(self, AX, B, Y):
        YBt = Y.dot(B.T)
        
        AX_sum = Y.dot(AX.sum(axis=1)).sum()
        B_sum  = Y.T.dot(B.sum(axis=0).T).sum()
        
        return 4 * AX.multiply(YBt).sum() + AX.shape[0] * Y.sum() - 2 * (AX_sum + B_sum)


class AuctionFusedSGM(ScipyFusedSGM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter = 0
    
    def solve_lap_fused(self, AP, B, augment=True):
        idx = dot_auction(AP, B, AP.shape[0])
        self.iter += 1
        return sparse.csr_matrix((np.ones(AP.shape[0]), (np.arange(idx.shape[0]), idx)))
