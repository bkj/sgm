#!/usr/bin/env python

"""
    sgm.py
    
    Agnostic SGM base classes
"""

import sys
from time import time

class _BaseSGM:
    def _reset_timers(self):
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

# --
# Original implementation

class BaseSGMClassic(_BaseSGM):
    def run(self, A, P, B, num_iters, tolerance, verbose=True):
        self._reset_timers()
        
        grad = self.compute_grad(A, P, B)
        
        for i in range(num_iters):
            if verbose:
                print('iter=%d      | %fs' % (i, time() - self.start_time), file=sys.stderr)
            
            T = self.solve_lap(grad)
            
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
            
            if stop:
                break
            
            if alpha is not None:
                P    = (alpha * P)    + (1 - alpha) * T
                grad = (alpha * grad) + (1 - alpha) * gradt
            else:
                P    = T
                grad = gradt
        
        P_out = self.solve_lap(P)
        return P_out

# --
# Sparse gradient + trace

class BaseSGMSparse(_BaseSGM):
    def run(self, A, P, B, num_iters, tolerance, verbose=True):
        if hasattr(self, '_warmup'):
            self._warmup()
            
        self._reset_timers()
        
        AP = A.dot(P)
        grad = AP.dot(B)
        
        for i in range(num_iters):
            if verbose:
                print('iter=%d      | %fs' % (i, time() - self.start_time), file=sys.stderr)
            
            t = time()
            T = self.solve_lap(grad)
            print('solve_lap      ', int(1000 * (time() - t)))
            
            AT = A.dot(T)
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
            
            if stop:
                break
            
            if alpha is not None:
                P    = (alpha * P) + (1 - alpha) * T
                grad = (alpha * grad) + (1 - alpha) * gradt
                AP   = (alpha * AP) + (1 - alpha) * AT
            else:
                P    = T
                grad = gradt
                AP   = AT
        
        P_out = self.solve_lap(P)
        return P_out

# --
# "Fused" operations

class BaseSGMFused(_BaseSGM):
    def run(self, A, P, B, num_iters, tolerance, verbose=True):
        if hasattr(self, '_warmup'):
            self._warmup()
        
        self._reset_timers()
        
        AP = A.dot(P)
        
        for i in range(num_iters):
            if verbose:
                print('iter=%d      | %fs' % (i, time() - self.start_time), file=sys.stderr)
            
            t = time()
            T = self.solve_lap_fused(AP, B, verbose=verbose)
            print('solve_lap_fused', int(1000 * (time() - t)))
            
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
            
            if stop:
                break
            
            if alpha is not None:
                P  = (alpha * P)  + (1 - alpha) * T
                AP = (alpha * AP) + (1 - alpha) * AT
            else:
                P  = T
                AP = AT
        
        P_out = self.solve_lap_exact(P)
        return P_out