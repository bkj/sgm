# #!/usr/bin/env python

# """
#     backends.py
# """

# import numpy as np
# from lap import lapjv
# from hashlib import md5
# from scipy import sparse
# from time import time

# # from sgm import BaseSGMClassic, BaseSGMSparse, BaseSGMFused

# from lap_solvers import gatagat_lapjv


# _lapjv = gatagat_lapjv

# --
# Classic Backends

# class _ScipySGMClassic(BaseSGMClassic):
#     def compute_grad(self, A, P, B):
#         AP = A.dot(P)
#         sparse_part = 4 * AP.dot(B) 
#         dense_part  = - 2 * AP.sum(axis=1) - 2 * B.sum(axis=0) + A.shape[0]
#         return np.asarray(sparse_part + dense_part)
        
#     def compute_trace(self, x, y):
#         return y.multiply(x).sum()


# class JVClassicSGM(_ScipySGMClassic):
#     def solve_lap(self, cost):
#         return _lapjv(cost)


# class AuctionClassicSGM(_ScipySGMClassic):
#     def solve_lap(self, cost, verbose=False):
#         print(cost)
#         idx = dense_lap_auction(cost,
#             verbose=verbose,
#             num_runs=1,
#             auction_max_eps=1.0,
#             auction_min_eps=1.0,
#             auction_factor=0.0
#         )
#         return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(idx.shape[0]), idx)))

# # --
# # Sparse backends

# class _ScipySGMSparse(BaseSGMSparse):
#     def _warmup(self):
#         cost = sparse.random(100, 100, density=0.5).tocsr()
#         _ = self.solve_lap(cost)
    
#     def compute_trace(self, AX, B, Y):
#         YBt = Y.dot(B.T)
        
#         AX_sum = Y.dot(AX.sum(axis=1)).sum()
#         B_sum  = Y.T.dot(B.sum(axis=0).T).sum()
        
#         return 4 * AX.multiply(YBt).sum() + AX.shape[0] * Y.sum() - 2 * (AX_sum + B_sum)


# class JVSparseSGM(_ScipySGMSparse):
#     def solve_lap(self, cost):
#         return _lapjv(cost)


# class AuctionSparseSGM(_ScipySGMSparse):
#     def solve_lap(self, cost, verbose=False):
#         idx = csr_lap_auction(cost,
#             verbose=verbose,
#             num_runs=1,
#             auction_max_eps=1.0,
#             auction_min_eps=1.0,
#             auction_factor=0.0
#         )
#         return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(idx.shape[0]), idx)))


# # --
# # Fused backends

# class _ScipyFusedSGM(BaseSGMFused):
#     def _warmup(self):
#         x = sparse.random(100, 100, density=0.5).tocsr()
#         y = sparse.random(100, 100, density=0.5).tocsr()
#         _ = self.solve_lap_fused(x, y, verbose=False)
    
#     def compute_trace(self, AX, B, Y):
#         YBt = Y.dot(B.T)
        
#         AX_sum = Y.dot(AX.sum(axis=1)).sum()
#         B_sum  = Y.T.dot(B.sum(axis=0).T).sum()
        
#         return 4 * AX.multiply(YBt).sum() + AX.shape[0] * Y.sum() - 2 * (AX_sum + B_sum)


# class JVFusedSGM(_ScipyFusedSGM):
#     def solve_lap_exact(self, cost):
#         return _lapjv(cost)
    
#     def solve_lap_fused(self, AP, B, verbose=True):
#         rowcol_offsets = - 2 * AP.sum(axis=1) - 2 * B.sum(axis=0) + A.shape[0]
#         return _lapjv(AP.dot(B).toarray() + rowcol_offsets)


# class AuctionFusedSGM(_ScipyFusedSGM):
#     def solve_lap_exact(self, cost):
#         return _lapjv(cost)
    
#     def solve_lap_fused(self, AP, B, verbose=False):
#         idx = dot_auction(AP, B, AP.shape[0], verbose=verbose)
#         return sparse.csr_matrix((np.ones(AP.shape[0]), (np.arange(idx.shape[0]), idx)))
