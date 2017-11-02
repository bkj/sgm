#!/usr/bin/env python

"""
    sgm.py
    
    Notes:
    
        !! Had to make one modification to the algorithm, to avoid a divide by zero error
        that python doesn't like.  However, this raises the point that the intermediate
        states of this implementation are not _exactly_ the same as in the R version.  
        
        I don't really know why that would be -- there may be some numerical stability issues.
            Eg, sum(S) in torch != sum(S) in numpy != sum(S) in R
        
        Also, unclear to me whether we need 32- or 64-bit floats
"""

from __future__ import division, print_function

import warnings
warnings.filterwarnings("ignore", module="matplotlib")

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from lap import lapjv

import torch
from torch.nn.functional import pad

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns


# --
# Helpers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--A-path', type=str, default='./data/A1.ordered')
    parser.add_argument('--B-path', type=str, default='./data/A2.ordered')
    parser.add_argument('--P-path', type=str, default='./data/S')
    parser.add_argument('--no-double', action="store_true")
    return parser.parse_args()

def load_matrix(path):
    mat = pd.read_csv(path, index_col=0)
    mat = np.array(mat, dtype='float64')
    return torch.Tensor(mat)

def square_pad(x, n):
    row_pad = n - x.size(0)
    col_pad = n - x.size(1)
    
    assert row_pad >= 0
    assert col_pad >= 0
    
    if row_pad > 0:
        x = torch.cat([x, torch.zeros(row_pad, x.size(1))], dim=0)
    
    if col_pad > 0:
        x = torch.cat([x, torch.zeros(x.size(0), col_pad)], dim=1)
    
    return x

args = parse_args()

if args.no_double:
    print("torch.set_default_tensor_type('torch.FloatTensor')")
    torch.set_default_tensor_type('torch.FloatTensor')
else:
    print("torch.set_default_tensor_type('torch.DoubleTensor')")
    torch.set_default_tensor_type('torch.DoubleTensor')

# --
# IO

A = load_matrix(args.A_path)
B = load_matrix(args.B_path)
P = load_matrix(args.P_path)

A_orig = A.clone()
B_orig = B.clone()
P_orig = P.clone()

# --
# Set parameters

m        = 0
patience = 20
pad      = 0
tol      = 1

assert m == 0, "m != 0 -- not implemented yet"

# --
# Prep

max_nodes  = max([A.size(0), B.size(0)])
n = max_nodes - m

A[A == 0] = -1
B[B == 0] = -1

A = square_pad(A, max_nodes)
B = square_pad(B, max_nodes)

if m != 0:
    raise NotImplemented
    # A12 <- rbind(A[1:m, (m+1):(m+n)])
    # A21 <- cbind(A[(m+1):(m+n), 1:m])
    # B12 <- rbind(B[1:m, (m+1):(m+n)])
    # B21 <- cbind(B[(m+1):(m+n), 1:m])
else:
    A12 = torch.zeros(n, n)
    A21 = torch.zeros(n, n)
    B12 = torch.zeros(n, n)
    B21 = torch.zeros(n, n)

if n == 1:
    raise NotImplemented
#     A12 <- t(A12)
#     A21 <- t(A21)
#     B12 <- t(B12)
#     B21 <- t(B21)

# --
# Run

A22 = A[m:(m+n), m:(m+n)]
B22 = B[m:(m+n), m:(m+n)]

x = torch.mm(A21, B21.t())
y = torch.mm(A12.t(), B12)

eye = torch.eye(n)

for i in tqdm(range(patience)):
    z = torch.mm(torch.mm(A22, P), B22.t()) # !! Order of these might matter
    w = torch.mm(torch.mm(A22.t(), P), B22) # !! Order of these might matter
    
    # Linear Assignment Problem
    grad = x + y + z + w
    cost = (grad + grad.abs().max()).numpy()
    _, ind, _ = lapjv(cost.max() - cost)
    ind = torch.LongTensor(ind.astype(int))
    
    # Matrix multiplications
    T   = eye[ind]
    wt  = torch.mm(torch.mm(A22.t(), T), B22)
    P_t, T_t = P.t(), T.t()
    c   = torch.trace(torch.mm(w, P_t))
    d   = torch.trace(torch.mm(wt, P_t)) + torch.trace(torch.mm(w, T_t))
    e   = torch.trace(torch.mm(wt, T_t))
    u   = torch.trace(torch.mm(P_t, x) + torch.mm(P_t, y))
    v   = torch.trace(torch.mm(T_t, x) + torch.mm(T_t, y))
    
    if (c - d + e == 0) and (d - 2 * e + u - v == 0):
        alpha = 0
    else:
        # !! Escape divide by zero error -- see note at top
        if (c - d + e == 0):
            alpha = float('inf')
        else:
            alpha = -(d - 2 * e + u - v) / (2 * (c - d + e))
    
    f1     = c - e + u - v
    falpha = (c - d + e) * alpha ** 2 + (d - 2 * e + u - v) * alpha
    
    if (alpha < tol) and (alpha > 0) and (falpha > 0) and (falpha > f1):
        P = alpha * P + (1 - alpha) * T
    elif f1 < 0:
        P = T
    else:
        print("breaking at iter=%d" % i)
        break


final_cost = (P.max() - P).numpy()
_, corr, _ = lapjv(final_cost)
P_final = eye[torch.LongTensor(corr.astype(int))]

p = P_final[:B_orig.size(0),:B_orig.size(1)]
B_perm = torch.mm(torch.mm(p, B_orig), p.t())

n_seeds = 145 # hardcoded for now
assert (A_orig[:n_seeds,:n_seeds] == B_perm[:n_seeds,:n_seeds]).all()
print("Ran successfully: A[:n_seeds,:n_seeds] = (p %*% B %*% p.T)[:n_seeds,:n_seeds]")

# --
# Visualization

print('plotting...')

_ = sns.heatmap(A_orig[:n_seeds, :n_seeds].numpy(),
    xticklabels=False, yticklabels=False, cbar=False, square=True)
_ = plt.title('A')
plt.savefig('A.png')
plt.close()

_ = sns.heatmap(B_perm[:n_seeds, :n_seeds].numpy(),
    xticklabels=False, yticklabels=False, cbar=False, square=True)
_ = plt.title('permuted B')
plt.savefig('B_perm.png')
plt.close()