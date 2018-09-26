#!/usr/bin/env python

"""
    backends.py
"""

import numpy as np
from lap import lapjv
from hashlib import md5
from scipy import sparse
from time import time

from sgm import BaseSGM, BaseFusedSGM

class ScipySparseSGM(BaseSGM):
    def solve_lap(self, cost):
        if isinstance(cost, sparse.csr_matrix):
            cost = cost.toarray()
        
        cols = np.arange(cost.shape[0]).reshape(-1, cost.shape[0])
        rows = cost.shape[0] * np.arange(cost.shape[0]).reshape(cost.shape[0], -1)
        cost_ = cost + cols + rows
        cost_ = cost_.max() - cost_
        
        _, idx, _ = lapjv(cost_)
        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(cost.shape[0]), idx)))
        
    def compute_grad(self, A, P, B):
        AP = A.dot(P)
        sparse_part = 4 * AP.dot(B) 
        dense_part  = - 2 * AP.sum(axis=1) - 2 * B.sum(axis=0) + A.shape[0]
        return sparse_part, dense_part
        
    def prod_sum(self, x, y):
        return y.multiply(x).sum()


class ScipyFusedSGM(BaseFusedSGM):
    def solve_lap(self, cost):
        if isinstance(cost, sparse.csr_matrix):
            cost = cost.toarray()
        
        _, idx, _ = lapjv(cost.max() - cost)
        
        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(cost.shape[0]), idx)))
    
    def solve_lap_fused(self, A, P, B):
        cost = A.dot(P).dot(B)
        
        if isinstance(cost, sparse.csr_matrix):
            cost = cost.toarray()
        
        t = time()
        _, idx, _ = lapjv(cost.max() - cost)
        print('lap_time', int(1000 * (time() - t)))
        
        vals = cost[(np.arange(cost.shape[0]), idx)]
        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(idx.shape[0]), idx)))
        
    def sparse_trace(self, A, X, B, Y):
        AX  = A.dot(X)
        YBt = Y.dot(B.T)
        
        AX_sum = Y.dot(AX.sum(axis=1)).sum()
        B_sum  = Y.T.dot(B.sum(axis=0).T).sum()
        
        return 4 * AX.multiply(YBt).sum() + A.shape[0] * Y.sum() - 2 * (AX_sum + B_sum)

# >>

import sys
sys.path.append('/home/bjohnson/projects/cuda_auction/python')
from lap_auction import dense_lap_auction, csr_lap_auction, dot_auction
rng = np.random.RandomState(seed=111222333)
_ = dense_lap_auction(rng.choice(100, (100, 100)), 
    verbose=False,
    num_runs=1,
    auction_max_eps=10.0,
    auction_min_eps=1.0,
    auction_factor=0.0
)

class AuctionFusedSGM(ScipyFusedSGM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter = 0
    
    def solve_lap_fused(self, A, P, B):
        # idx = dense_lap_auction(A.dot(P).dot(B).toarray(),
        #     verbose=True,
        #     num_runs=1,
        #     auction_max_eps=1.0,
        #     auction_min_eps=1.0,
        #     auction_factor=0.0
        # )
        
        if self.iter < 4:
            print('csr_lap_auction')
            idx = csr_lap_auction(A.dot(P).dot(B), 
                verbose=True,
                num_runs=1,
                auction_max_eps=1.0,
                auction_min_eps=1.0,
                auction_factor=0.0
            )
        else:
            print('lapjv')
            cost = A.dot(P).dot(B).toarray()
            _, idx, _ = lapjv(cost.max() - cost)
        
        # idx = dot_auction(A.dot(P), B, 256)
        
        self.iter += 1
        return sparse.csr_matrix((np.ones(A.shape[0]), (np.arange(idx.shape[0]), idx)))


class AuctionSparseSGM(ScipySparseSGM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter = 0
    
    def solve_lap(self, cost):
        if self.iter < 4:
            print('csr_lap_auction')
            idx = csr_lap_auction(cost, 
                verbose=True,
                num_runs=1,
                auction_max_eps=1.0,
                auction_min_eps=1.0,
                auction_factor=0.0
            )
        else:
            print('lapjv')
            cost = cost.toarray()
            _, idx, _ = lapjv(cost.max() - cost)
        
        # idx = dot_auction(A.dot(P), B, 256)
        
        self.iter += 1
        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(idx.shape[0]), idx)))


# <<