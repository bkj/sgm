#!/usr/bin/env python

"""
    sgm/utils.py
"""

import numpy as np

import torch
from scipy import sparse

def sparse2numpy(x):
    return np.asarray(x.todense())

def sparse2torch(X, cuda=True):
    X = sparse2numpy(X)
    X = torch.FloatTensor(X)
    if cuda:
        X = X.cuda()
    
    return X