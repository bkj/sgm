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
from time import time
from functools import partial

import torch
from torch.nn.functional import pad

from lap import lapjv
sys.path.append('auction-lap')
from auction_lap.auction_lap import auction_lap

from sgm import sgm

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
    parser.add_argument('--num-iters', type=int, default=20)
    parser.add_argument('--tolerance', type=int, default=1)
    parser.add_argument('--symmetric', action="store_true")
    parser.add_argument('--sparse', action="store_true")
    
    # LAP params
    parser.add_argument('--mode', type=str, default='exact')
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

def solve_lap(cost_sparse, cost_offset, mode, cuda, eps, eye):
    cost = cost_sparse + cost_offset
    cost = cost - cost.min() # Make >= 0
    if mode == 'exact':
        cost = cost.cpu().numpy()
        _, idx, _ = lapjv(cost.max() - cost)
        idx = torch.LongTensor(idx.astype(int))
        if cuda:
            idx = idx.cuda()
    elif mode == 'auction':
        _, idx, _ = auction_lap(cost, eps=eps)
    
    return eye[idx]

def compute_grad(A, P, B, sparse=False):
    if not sparse:
        return torch.mm(torch.mm(A, P), B)
    else:
        AP          = torch.mm(A, P)
        grad        = 4 * torch.mm(AP, B)
        grad_offset = - 2 * AP.sum(dim=-1).view(-1, 1) - 2 * B.sum(dim=0).view(1, -1) + A.size(0)
        return grad, grad_offset

def prod_sum(x, y):
    return (x * y).sum()


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

if not args.sparse:
    A_orig, B_orig = A.clone(), B.clone()
    A[A == 0] = -1
    B[B == 0] = -1
else:
    A_orig, B_orig = A, B

# --
# Prep

n_seeds = (P.diag() == 1).sum()
# >>
# Start at vertex of polytope corresponding to seed
# P[n_seeds:, n_seeds:] = 0
# <<
max_nodes = max([A.size(0), B.size(0)])
min_nodes = min([A.size(0), B.size(0)])

A = square_pad(A, max_nodes)
B = square_pad(B, max_nodes)
eye = torch.eye(max_nodes)

if args.cuda:
    A, B, P, eye = A.cuda(), B.cuda(), P.cuda(), eye.cuda()

# --
# Run

start_time = time()
P_out = sgm(
    A=A,
    P=P,
    B=B,
    eye=eye,
    compute_grad=partial(compute_grad, sparse=args.sparse),
    solve_lap=partial(solve_lap, mode=args.mode, cuda=args.cuda, eps=args.eps),
    prod_sum=prod_sum,
    num_iters=args.num_iters,
    tolerance=args.tolerance,
)
total_time = time() - start_time

# --
# Save results

P_out_small = P_out[:min_nodes,:min_nodes]
P_out_small = P_out_small.cpu()

B_perm = torch.mm(torch.mm(P_out_small, B_orig), P_out_small.t())

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

corr = P_out.nonzero().cpu().numpy() + 1
np.savetxt(args.outpath, corr, fmt='%d')
