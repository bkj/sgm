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

def check_convergence(c, d, e):
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

def sgm(A, P, B, compute_grad, solve_lap, num_iters, tolerance, prod_sum=None, verbose=True):
    
    grad = compute_grad(A, P, B)
    ps_grad_P = prod_sum(grad, P)
    
    stop = False
    for i in range(num_iters):
        iter_start_time = time()
        T = solve_lap(grad)
        lap_time = time() - iter_start_time
        
        gradt = compute_grad(A, T, B)
        
        ps_grad_T  = prod_sum(grad, T)
        ps_gradt_P = prod_sum(gradt, P)
        ps_gradt_T = prod_sum(gradt, T)
        
        alpha, f1, falpha = check_convergence(
            c=ps_grad_P,
            d=ps_gradt_P + ps_grad_T,
            e=ps_gradt_T
        )
        
        if (alpha < tolerance) and (alpha > 0) and (falpha > 0) and (falpha > f1):
            P         = (alpha * P)         + (1 - alpha) * T
            grad      = (alpha * grad)      + (1 - alpha) * gradt
            ps_grad_P = (alpha * ps_grad_P) + (1 - alpha) * ps_gradt_P
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
                "ps_grad_P"  : float(ps_grad_P),
                "ps_grad_T"  : float(ps_grad_T),
                "ps_gradt_P" : float(ps_gradt_P),
                "ps_gradt_T" : float(ps_gradt_T),
            }))
        
        if stop:
            break
    
    return solve_lap(P)