#!/usr/bin/env python

"""
    sgm/utils.py
"""

import sys
import numpy as np

from scipy import sparse

try:
    import torch
except:
    print('!! Could not import torch', file=sys.stderr)


def sparse2numpy(x):
    return np.asarray(x.todense())

def sparse2torch(X, cuda=True):
    X = sparse2numpy(X)
    X = torch.FloatTensor(X)
    if cuda:
        X = X.cuda()
    
    return X