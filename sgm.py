#!/usr/bin/env python

"""
    sgm.py
    
    Agnostic SGM implementation
    
    Notes:
        - `compute_grad` returns a dense matrix, which is the sum of a sparse
        matrix and the outer sum of a row vector and a column vector.  So if 
        we were to write a custom LAP solver, we may be able to take advantage
        of this to same memory space, which is a concern.
    
"""

import sys
import json
from time import time




class _BaseSGM:
    def __reset_timers(self):
        self.lap_times   = []
        self.grad_times  = []
        self.check_times = []
        self.start_time  = time()
    
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


class BaseSGM(_BaseSGM):
    def run(self, A, P, B, num_iters, tolerance, verbose=True):
        self.__reset_timers()
        sparse_grad, dense_grad = self.compute_grad(A, P, B)
        
        stop = False
        for i in range(num_iters):
            if verbose:
                print('iter=%d | %fs' % (i, time() - self.start_time), file=sys.stderr)
            
            T = self.solve_lap(sparse_grad)
            
            sparse_gradt, dense_gradt = self.compute_grad(A, T, B)
            
            ps_grad_P  = self.prod_sum(sparse_grad + dense_grad, P)
            ps_grad_T  = self.prod_sum(sparse_grad + dense_grad, T)
            ps_gradt_P = self.prod_sum(sparse_gradt + dense_gradt, P)
            ps_gradt_T = self.prod_sum(sparse_gradt + dense_gradt, T)
            
            alpha, stop = self.check_convergence(
                c=ps_grad_P,
                d=ps_gradt_P + ps_grad_T,
                e=ps_gradt_T,
                tolerance=tolerance,
            )
            
            if stop:
                break
            
            if alpha is not None:
                P           = (alpha * P)           + (1 - alpha) * T
                sparse_grad = (alpha * sparse_grad) + (1 - alpha) * sparse_gradt
                dense_grad  = (alpha * dense_grad)  + (1 - alpha) * dense_gradt
            else:
                P           = T
                sparse_grad = sparse_gradt
                dense_grad  = dense_gradt
        
        P_out = self.solve_lap(P)
        return P_out

# =====================================

class BaseSparseSGM(_BaseSGM):
    def run(self, A, P, B, num_iters, tolerance, verbose=True):
        self.__reset_timers()
        
        grad = A.dot(P).dot(B)
        
        stop = False
        for i in range(num_iters):
            if verbose:
                print('iter=%d | %fs' % (i, time() - self.start_time), file=sys.stderr)
            
            T = self.solve_lap(grad)
            
            gradt = A.dot(T).dot(B)
            
            # Could avoid recomputation here
            ps_grad_P  = self.sparse_trace(A, P, B, P)
            ps_grad_T  = self.sparse_trace(A, P, B, T)
            ps_gradt_P = self.sparse_trace(A, T, B, P)
            ps_gradt_T = self.sparse_trace(A, T, B, T)
            
            alpha, stop = self.check_convergence(
                c=ps_grad_P,
                d=ps_gradt_P + ps_grad_T,
                e=ps_gradt_T,
                tolerance=tolerance,
            )
            
            if stop:
                break
            
            if alpha is not None:
                P    = (alpha * P) + (1 - alpha) * T
                grad = (alpha * grad) + (1 - alpha) * gradt
            else:
                P = T
                grad = gradt
        
        P_out = self.solve_lap(P)
        return P_out

# =====================================

class BaseFusedSGM(_BaseSGM):
    def run(self, A, P, B, num_iters, tolerance, verbose=True):
        self.__reset_timers()
        
        stop = False
        AP = A.dot(P)
        for i in range(num_iters):
            if verbose:
                print('iter=%d | %fs' % (i, time() - self.start_time), file=sys.stderr)
            
            T = self.solve_lap_fused(AP, B)
            
            AT = A.dot(T)
            ps_grad_P  = self.sparse_trace(AP, B, P)
            ps_grad_T  = self.sparse_trace(AP, B, T)
            ps_gradt_P = self.sparse_trace(AT, B, P)
            ps_gradt_T = self.sparse_trace(AT, B, T)
            
            alpha, stop = self.check_convergence(
                c=ps_grad_P,
                d=ps_gradt_P + ps_grad_T,
                e=ps_gradt_T,
                tolerance=tolerance,
            )
            
            if stop:
                break
            
            if alpha is not None:
                P  = (alpha * P)  + (1 - alpha) * T
                AP = (alpha * AP) + (1 - alpha) * AT
            else:
                P  = T
                AP = AT
        
        P_out = self.solve_lap(P)
        return P_out