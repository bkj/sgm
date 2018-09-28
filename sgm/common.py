#!/usr/bin/env python

"""
    base_sgm.py
"""

import json
from time import time

# --
# Base SGM class

class _BaseSGM:
    def _reset_timers(self):
        self.lap_times   = []
        self.iter_times  = []
    
    def _log_times(self):
        i = len(self.lap_times)
        print(json.dumps({
            "iter"      : i,
            "lap_time"  : float(self.lap_times[-1]),
            "iter_time" : float(self.iter_times[-1]),
        }))
    
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

