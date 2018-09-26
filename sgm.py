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

def _check_convergence(c, d, e):
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
    
    return alpha, f1, falpha

import numpy as np
class BaseSGM:
    def __reset_timers(self):
        self.lap_times   = []
        self.grad_times  = []
        self.check_times = []
    
    def run(self, A, P, B, num_iters, tolerance, verbose=True):
        self.__reset_timers()
        start_time = time()
        
        t = time()
        sparse_grad, dense_grad = self.compute_grad(A, P, B)
        self.grad_times.append(time() - t)
        
        stop = False
        for i in range(num_iters):
            if verbose:
                print('iter=%d | %fs' % (i, time() - start_time), file=sys.stderr)
            
            t = time()
            T = self.solve_lap(sparse_grad)
            self.lap_times.append(time() - t)
            
            t = time()
            sparse_gradt, dense_gradt = self.compute_grad(A, T, B)
            self.grad_times.append(time() - t)
            
            t = time()
            ps_grad_P  = self.prod_sum(sparse_grad + dense_grad, P)
            ps_grad_T  = self.prod_sum(sparse_grad + dense_grad, T)
            ps_gradt_P = self.prod_sum(sparse_gradt + dense_gradt, P)
            ps_gradt_T = self.prod_sum(sparse_gradt + dense_gradt, T)
            
            alpha, f1, falpha = _check_convergence(
                c=ps_grad_P,
                d=ps_gradt_P + ps_grad_T,
                e=ps_gradt_T
            )
            
            if (alpha > 0) and (alpha < tolerance) and (falpha > max(0, f1)):
                P           = (alpha * P)    + (1 - alpha) * T
                sparse_grad = (alpha * sparse_grad) + (1 - alpha) * sparse_gradt
                dense_grad  = (alpha * dense_grad) + (1 - alpha) * dense_gradt
            elif f1 < 0:
                P           = T
                sparse_grad = sparse_gradt
                dense_grad  = dense_gradt
            else:
                stop = True
            
            self.check_times.append(time() - t)
            
            if stop:
                break
        
        t = time()
        P_out = self.solve_lap(P)
        self.lap_times.append(time() - t)
        return P_out


class BaseFusedSGM:
    def __reset_timers(self):
        self.lap_times   = []
        self.grad_times  = []
        self.check_times = []
    
    def run(self, A, P, B, num_iters, tolerance, verbose=True):
        self.__reset_timers()
        start_time = time()
        
        stop = False
        for i in range(num_iters):
            if verbose:
                print('iter=%d | %fs' % (i, time() - start_time), file=sys.stderr)
            
            t = time()
            T = self.solve_lap_fused(A, P, B)
            self.lap_times.append(time() - t)
            
            t = time()
            # Could avoid recomputation here
            ps_grad_P  = self.sparse_trace(A, P, B, P)
            ps_grad_T  = self.sparse_trace(A, P, B, T)
            ps_gradt_P = self.sparse_trace(A, T, B, P)
            ps_gradt_T = self.sparse_trace(A, T, B, T)
            
            alpha, f1, falpha = _check_convergence(
                c=ps_grad_P,
                d=ps_gradt_P + ps_grad_T,
                e=ps_gradt_T
            )
            
            if (alpha > 0) and (alpha < tolerance) and (falpha > max(0, f1)):
                P = (alpha * P) + (1 - alpha) * T
            elif f1 < 0:
                P = T
            else:
                stop = True
            
            self.check_times.append(time() - t)
            
            if stop:
                break
        
        t = time()
        P_out = self.solve_lap(P)
        self.lap_times.append(time() - t)
        return P_out