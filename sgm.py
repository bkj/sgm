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

import json
from time import time

def _check_convergence(c, d, e):
    if (c - d + e == 0) and (d - 2 * e == 0):
        alpha = 0
    else:
        if (c - d + e == 0):
            alpha = float('inf')
        else:
            alpha = -(d - 2 * e) / (2 * (c - d + e))
            
    f1     = c - e
    falpha = (c - d + e) * alpha ** 2 + (d - 2 * e) * alpha
    
    return alpha, f1, falpha


class BaseSGM:
    def run(self, A, P, B, num_iters, tolerance, verbose=True):
        
        grad = self.compute_grad(A, P, B)
        
        stop = False
        for i in range(num_iters):
            iter_start_time = time()
            T = self.solve_lap(grad)
            lap_time = time() - iter_start_time
            
            gradt = self.compute_grad(A, T, B)
            
            ps_grad_P  = self.prod_sum(grad, P)
            ps_grad_T  = self.prod_sum(grad, T)
            ps_gradt_P = self.prod_sum(gradt, P)
            ps_gradt_T = self.prod_sum(gradt, T)
            
            alpha, f1, falpha = _check_convergence(
                c=ps_grad_P,
                d=ps_gradt_P + ps_grad_T,
                e=ps_gradt_T
            )
            
            if (alpha > 0) and (alpha < tolerance) and (falpha > max(0, f1)):
                P         = (alpha * P)         + (1 - alpha) * T
                grad      = (alpha * grad)      + (1 - alpha) * gradt
            elif f1 < 0:
                P         = T
                grad      = gradt
                ps_grad_P = ps_gradt_P
            else:
                stop = True
            
            if verbose:
                iter_time = time() - iter_start_time
                print(json.dumps({
                    "iter"       : i,
                    "lap_time"   : lap_time,
                    "nolap_time" : iter_time - lap_time,
                    
                    # debugging
                    # "ps_grad_P"  : float(ps_grad_P),
                    # "ps_grad_T"  : float(ps_grad_T),
                    # "ps_gradt_P" : float(ps_gradt_P),
                    # "ps_gradt_T" : float(ps_gradt_T),
                }))
            
            if stop:
                break
        
        return self.solve_lap(P)