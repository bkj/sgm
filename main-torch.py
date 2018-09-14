#!/usr/bin/env python

"""
    sgm.py
"""

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
torch.set_default_tensor_type('torch.DoubleTensor')

from scipy import sparse

from lap import lapjv
sys.path.append('auction-lap')
from auction_lap.auction_lap import auction_lap

from sgm import sgm

# --
# Helpers

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data params
    parser.add_argument('--A-path', type=str, default='_data/synthetic/dense/0.5/1000/A.csv')
    parser.add_argument('--B-path', type=str, default='_data/synthetic/dense/0.5/1000/B.csv')
    parser.add_argument('--P-path', type=str)
    parser.add_argument('--num-seeds', type=int)
    
    # SGM params
    parser.add_argument('--num-iters', type=int, default=20)
    parser.add_argument('--tolerance', type=int, default=1)
    
    # LAP params
    parser.add_argument('--lap-mode', type=str, default='jv', choices=['auction', 'jv'])
    parser.add_argument('--auction-eps', type=float, default=100)
    
    # Misc params
    parser.add_argument('--cuda', action="store_true")
    
    args = parser.parse_args()
    assert (args.P_path is not None) or (args.num_seeds is not None)
    return args


def load_matrix(path):
    mat = pd.read_csv(path).values
    
    if mat.shape[1] == 2:
        raise NotImplemented
    elif mat.shape[1] == 3:
        raise NotImplemented
    else:
        mat = torch.Tensor(mat.astype(np.float64))
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

def solve_lap(cost, mode, cuda, eps):
    cost = cost - cost.min() # Make >= 0
    print('cost.max()', cost.max())
    if mode == 'jv':
        cost = cost.cpu().numpy()
        _, idx, _ = lapjv(cost.max() - cost)
        idx = torch.LongTensor(idx.astype(int))
        if cuda:
            idx = idx.cuda()
    elif mode == 'auction':
        _, idx, _ = auction_lap(cost, eps=eps)
    
    out = torch.eye(idx.size(0))
    if cuda:
        out = out.cuda()
    return out[idx]

def compute_grad(A, P, B):
    AP = torch.mm(A, P)
    return 4 * torch.mm(AP, B) - 2 * (AP.sum(dim=-1).view(-1, 1) + B.sum(dim=0).view(1, -1)) + A.size(0)

def prod_sum(x, y):
    return (x * y).sum()


args = parse_args()

# --
# IO

t = time()

A = load_matrix(args.A_path)
B = load_matrix(args.B_path)

assert A.size() == B.size()

f_actual = np.sqrt(((A - B) ** 2).sum())

if args.P_path:
    P = load_matrix(args.P_path, shape=A.size(0))
    num_seeds = (P.diagonal() == 1).sum()
else:
    num_seeds = args.num_seeds
    
    P = torch.eye(A.size(0))
    P[num_seeds:, num_seeds:] = 0
    
    perm = np.arange(P.shape[0])
    perm[num_seeds:] = np.random.permutation(perm[num_seeds:])
    perm = torch.LongTensor(perm)
    A = A[perm][:,perm]

io_time = time() - t
print('loaded in %fs' % io_time, file=sys.stderr)

min_nodes, max_nodes = sorted([A.size(0), B.size(0)])
f_orig = np.sqrt(((A[:min_nodes,:min_nodes] - B[:min_nodes,:min_nodes]) ** 2).sum())

# --
# Run

if args.cuda:
    A, B, P = A.cuda(), B.cuda(), P.cuda()

t = time() 
P_out = sgm(
    A=A,
    P=P,
    B=B,
    compute_grad=compute_grad,
    solve_lap=partial(solve_lap, mode=args.lap_mode, cuda=args.cuda, eps=args.auction_eps),
    prod_sum=prod_sum,
    num_iters=args.num_iters,
    tolerance=args.tolerance,
)
compute_time = time() - t

# --
# Save results

P_out_small = P_out[:min_nodes,:min_nodes]
B_perm      = torch.mm(torch.mm(P_out_small, B), P_out_small.t())
f_perm      = np.sqrt(((A[:min_nodes,:min_nodes] - B_perm[:min_nodes,:min_nodes]) ** 2).sum())

print(json.dumps({
    "f_actual"     : float(f_actual),
    "f_orig"       : float(f_orig),
    "f_perm"       : float(f_perm),
    
    "io_time"      : float(io_time),
    "compute_time" : float(compute_time),
    
    "lap_mode"     : args.lap_mode,
    "auction_eps"  : args.auction_eps if args.lap_mode == 'auction' else None,
    "max_nodes"    : int(max_nodes),
    "num_seeds"    : int(num_seeds),
}))
