#!/usr/bin/env python

"""
    backends.py
"""

import numpy as np
from lap import lapjv
from hashlib import md5
from scipy import sparse

from sgm import BaseSGM

class ScipySparseSGM(BaseSGM):
    def solve_lap(self, cost):
        # cost_all = cost.copy()
        
        if isinstance(cost, sparse.csr_matrix):
            cost = cost.toarray()
        
        cost = cost - cost.min()
        _, idx, _ = lapjv(cost.max() - cost)
        
        # print("score", cost_all[(np.arange(cost.shape[0]), idx)].sum())
        # print(md5(str(idx).encode()).hexdigest())
        
        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(cost.shape[0]), idx)))
        
    def compute_grad(self, A, P, B):
        AP = A.dot(P)
        out = 4 * AP.dot(B) - 2 * AP.sum(axis=1) - 2 * B.sum(axis=0) + A.shape[0]
        out = np.asarray(out)
        return out
        
    def prod_sum(self, x, y):
        return y.multiply(x).sum()


class ScipyTruncatedSGM(BaseSGM):
    def solve_lap(self, cost):
        if isinstance(cost, sparse.csr_matrix):
            cost = cost.toarray()
        
        cost = cost - cost.min()
        _, idx, _ = lapjv(cost.max() - cost)
        
        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(cost.shape[0]), idx)))
    
    def solve_lap_fused(self, A, P, B):
        cost = A.dot(P).dot(B)
        
        if isinstance(cost, sparse.csr_matrix):
            cost = cost.toarray()
        
        cost = cost - cost.min()
        _, idx, _ = lapjv(cost.max() - cost)
        
        # cost_all = self.compute_grad(A, P, B)
        # print("score", cost_all[(np.arange(cost.shape[0]), idx)].sum())
        # print(md5(str(idx).encode()).hexdigest())
        
        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(cost.shape[0]), idx)))
        
    def sparse_trace(self, A, X, B, Y):
        AX  = A.dot(X)
        YBt = Y.dot(B.T)
        
        AX_sum = Y.dot(AX.sum(axis=1)).sum()
        B_sum  = Y.T.dot(B.sum(axis=0).T).sum()
        
        return 4 * AX.multiply(YBt).sum() + A.shape[0] * Y.sum() - 2 * (AX_sum + B_sum)
