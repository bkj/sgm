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
    X = torch.Tensor(sparse2numpy(X))
    if cuda:
        X = X.cuda()
    
    return X