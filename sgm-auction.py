#!/usr/bin/env python

"""
    sgm.py
"""

from __future__ import division, print_function

import warnings
warnings.filterwarnings("ignore", module="matplotlib")

import sys
import json
import argparse
import numpy as np
import pandas as pd
from lap import lapjv

sys.path.append('auction-lap')
from auction_lap import auction_lap

import torch
from time import time
from torch.nn.functional import pad

# --
# Helpers

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--A-path', type=str, required=True)
    parser.add_argument('--B-path', type=str, required=True)
    parser.add_argument('--P-path', type=str, required=True)
    parser.add_argument('--outpath', type=str, default='./_simple_corr.txt')
    
    # SGM params
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--tolerance', type=int, default=1)
    parser.add_argument('--symmetric', action="store_true")
    
    # LAP params
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--eps', type=float, default=100)
    
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--no-double', action="store_true")
    
    args = parser.parse_args()
    assert args.m == 0, "m != 0 -- not implemented yet"
    return args


def load_matrix(path):
    mat = pd.read_csv(path)
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

def solve_lap(cost, mode, cuda, eps, eye):
    if mode == 'exact':
        cost = cost.cpu().numpy()
        _, idx, _ = lapjv(cost.max() - cost)
        # lap_cost = cost[(np.arange(cost.shape[0]), idx)].sum()
        idx = torch.LongTensor(idx.astype(int))
        if cuda:
            idx = idx.cuda()
    elif mode == 'auction':
        _, idx, _ = auction_lap(cost, eps=eps)
    
    return eye[idx]


args = parse_args()

if args.no_double:
    torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')

# --
# IO

A = load_matrix(args.A_path)
B = load_matrix(args.B_path)
P = load_matrix(args.P_path)

A_orig = A.clone()
B_orig = B.clone()

# --
# Prep

n_seeds = (P.diag() == 1).sum()

# # >>
# # Sparsify
# # thresh = 0.00
# # mask = torch.rand(P.size(0) - n_seeds, P.size(0) - n_seeds) < thresh
# P[n_seeds:,n_seeds:] = 0 # P[n_seeds:,n_seeds:] * mask.double()
# # <<

max_nodes = max([A.size(0), B.size(0)])
min_nodes = min([A.size(0), B.size(0)])

sparse = True
print('sparse', sparse)
if sparse:
    assert args.symmetric

if not sparse:
    A[A == 0] = -1
    B[B == 0] = -1

A = square_pad(A, max_nodes)
B = square_pad(B, max_nodes)
eye = torch.eye(max_nodes)

if args.cuda:
    A, B, P, eye = A.cuda(), B.cuda(), P.cuda(), eye.cuda()
    A_orig, B_orig = A_orig.cuda(), B_orig.cuda()

# --
# Run

auction_steps = 0
stop = False
t0 = time()
for i in range(args.patience):
    t1 = time()
    
    if args.symmetric:
        if sparse:
            AP = torch.mm(A, P)
            z = w = 4 * torch.mm(AP, B) - 2 * AP.sum(dim=-1).view(-1, 1) - 2 * B.sum(dim=0).view(1, -1) + A.size(0)
        else:
            z = w = torch.mm(torch.mm(A, P), B)
        
        grad = 2 * z
    else:
        z = torch.mm(torch.mm(A, P), B.t())
        w = torch.mm(torch.mm(A.t(), P), B)
        grad = z + w
    
    if grad.min() < 0:
        cost = grad - grad.min()
    else:
        cost = grad
    
    # torch.save(cost, 'cost')
    
    t2 = time()
    T = solve_lap(cost, mode=args.mode, cuda=args.cuda, eps=args.eps, eye=eye)
    lap_time = time() - t2
    
    # Matrix multiplications
    if sparse:
        AT = torch.mm(A, T)
        wt = 4 * torch.mm(AT, B) - 2 * AT.sum(dim=-1).view(-1, 1) - 2 * B.sum(dim=0).view(1, -1) + A.size(0)
    else:
        wt  = torch.mm(torch.mm(A.t(), T), B)
    
    c = torch.sum(w * P)
    d = torch.sum(wt * P) + torch.sum(w * T)
    e = torch.sum(wt * T)
    
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
        stop = True
    
    iter_time = time() - t1
    print(json.dumps({
        "iter"          : i,
        "lap_time"      : lap_time,
        "nolap_time"    : iter_time - lap_time,
        "auction_steps" : auction_steps,
        
        "mode"      : args.mode,
        "eps"       : args.eps,
        "max_nodes" : int(max_nodes),
        "n_seeds"   : int(n_seeds),
    }))
    
    if stop:
        break

total_time = time() - t0

P_final = solve_lap(cost, mode=args.mode, cuda=args.cuda, eps=args.eps, eye=eye)

# --
# Save results


p = P_final[:B_orig.size(0),:B_orig.size(1)]
B_perm = torch.mm(torch.mm(p, B_orig), p.t())

f_orig = np.sqrt(((A_orig[:min_nodes,:min_nodes] - B_orig[:min_nodes,:min_nodes]) ** 2).sum())
f_perm = np.sqrt(((A_orig[:min_nodes,:min_nodes] - B_perm[:min_nodes,:min_nodes]) ** 2).sum())

print(json.dumps({
    "f_orig"     : float(f_orig),
    "f_perm"     : float(f_perm),
    "total_time" : float(total_time),
    
    "mode"      : args.mode,
    "eps"       : args.eps,
    "max_nodes" : int(max_nodes),
    "n_seeds"   : int(n_seeds),
}))

# corr = np.vstack([np.arange(corr.shape[0]), corr]).T + 1 # Increment by 1 to match R output
# np.savetxt(args.outpath, corr, fmt='%d')

# # --
# # Visualization

# if args.plot:
#     import matplotlib
#     matplotlib.use('Agg')
#     from matplotlib import pyplot as plt
#     import seaborn as sns
    
#     print('sgm.py: plotting', file=sys.stderr)
    
#     _ = sns.heatmap(A_orig[:n_seeds, :n_seeds].numpy(),
#         xticklabels=False, yticklabels=False, cbar=False, square=True)
#     _ = plt.title('A')
#     plt.savefig('A.png')
#     plt.close()
    
#     _ = sns.heatmap(B_perm[:n_seeds, :n_seeds].numpy(),
#         xticklabels=False, yticklabels=False, cbar=False, square=True)
#     _ = plt.title('permuted B')
#     plt.savefig('B_perm.png')
#     plt.close()
