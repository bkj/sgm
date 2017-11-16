#!/usr/bin/env python

"""
    sgm-m0.py
    
    Notes:
        Same caveats apply as in sgm.py
        
        However, this is tuned for the case of `m = 0`
"""

from __future__ import division, print_function

import warnings
warnings.filterwarnings("ignore", module="matplotlib")

import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from lap import lapjv

import torch
from time import time
from torch.nn.functional import pad

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

# --
# Helpers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--A-path', type=str, default='./data/data.sgm/A1.ordered')
    parser.add_argument('--B-path', type=str, default='./data/data.sgm/A2.ordered')
    parser.add_argument('--P-path', type=str, default='./data/data.sgm/S')
    parser.add_argument('--outpath', type=str, default='./corr.txt')
    
    parser.add_argument('--no-double', action="store_true")
    
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--tolerance', type=int, default=1)
    
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--cuda', action="store_true")
    
    args = parser.parse_args()
    assert args.m == 0, "m != 0 -- not implemented yet"
    return args


def load_matrix(path):
    mat = pd.read_csv(path, index_col=0)
    mat = np.array(mat, dtype='float64')
    mat = torch.Tensor(mat)
    assert mat.size(0) == mat.size(1), "%s must be square" % path
    return mat


def square_pad(x, n):
    row_pad = n - x.size(0)
    col_pad = n - x.size(1)
    
    assert row_pad >= 0, 'row_pad < 0'
    assert col_pad >= 0, 'col_pad < 0'
    
    if row_pad > 0:
        x = torch.cat([x, torch.zeros(row_pad, x.size(1))], dim=0)
    
    if col_pad > 0:
        x = torch.cat([x, torch.zeros(x.size(0), col_pad)], dim=1)
    
    return x


args = parse_args()

if args.no_double:
    torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')

# --
# IO

t = time()

A = load_matrix(args.A_path)
B = load_matrix(args.B_path)
P = load_matrix(args.P_path)

A_orig = A.clone()
B_orig = B.clone()
P_orig = P.clone()

print("io_time\t%f" % (time() - t))

# --
# Prep

t = time()

n_seeds = (P.diag() == 1).sum()
max_nodes  = max([A.size(0), B.size(0)])
n = max_nodes - args.m

A[A == 0] = -1
B[B == 0] = -1

A = square_pad(A, max_nodes)
B = square_pad(B, max_nodes)
eye = torch.eye(n)

if args.cuda:
    A, B, P, eye = A.cuda(), B.cuda(), P.cuda(), eye.cuda()

print("prep_time\t%f" % (time() - t))

# --
# Run

t = time()

for i in tqdm(range(args.patience)):
    z = torch.mm(torch.mm(A, P), B.t())
    w = torch.mm(torch.mm(A.t(), P), B)
    
    # Linear Assignment Problem
    grad = z + w
    cost = (grad + grad.abs().max()).cpu().numpy()
    _, ind, _ = lapjv(cost.max() - cost)
    ind = torch.LongTensor(ind.astype(int))
    if args.cuda:
        ind = ind.cuda()
    
    # Matrix multiplications
    T   = eye[ind]
    wt  = torch.mm(torch.mm(A.t(), T), B)
    c   = torch.sum(w * P)
    d   = torch.sum(wt * P) + torch.sum(w * T)
    e   = torch.sum(wt * T)
    
    if (c - d + e == 0) and (d - 2 * e == 0):
        alpha = 0
    else:
        # !! Escape divide by zero error -- see note at top
        if (c - d + e == 0):
            alpha = float('inf')
        else:
            alpha = -(d - 2 * e) / (2 * (c - d + e))
    
    f1     = c - e
    falpha = (c - d + e) * alpha ** 2 + (d - 2 * e) * alpha
    
    if (alpha < args.tolerance) and (alpha > 0) and (falpha > 0) and (falpha > f1):
        P = alpha * P + (1 - alpha) * T
    elif f1 < 0:
        P = T
    else:
        print("breaking at iter=%d" % i, file=sys.stderr)
        break

final_cost = (P.max() - P).cpu().numpy()
_, corr, _ = lapjv(final_cost)
P_final = eye.cpu()[torch.LongTensor(corr.astype(int))]

print("run_time\t%f" % (time() - t))

# --
# Save results

print('sgm-m0.py: saving', file=sys.stderr)

p = P_final[:B_orig.size(0),:B_orig.size(1)]
B_perm = torch.mm(torch.mm(p, B_orig), p.t())

assert (A_orig[:n_seeds,:n_seeds] == B_perm[:n_seeds,:n_seeds]).all()
print("Ran successfully: A[:n_seeds,:n_seeds] = (p %*% B %*% p.T)[:n_seeds,:n_seeds]", file=sys.stderr)

corr = np.vstack([np.arange(corr.shape[0]), corr]).T
np.savetxt(args.outpath, corr, fmt='%d')

# --
# Visualization

if args.plot:
    print('sgm-m0.py: plotting', file=sys.stderr)
    
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
