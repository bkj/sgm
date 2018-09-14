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

from sgm import BaseSGM

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
    parser.add_argument('--lap-mode', type=str, default='jv', choices=['auction', 'jv', 'greedy'])
    parser.add_argument('--auction-eps', type=float, default=100)
    
    # Misc params
    parser.add_argument('--cuda', action="store_true")
    
    args = parser.parse_args()
    assert (args.P_path is not None) or (args.num_seeds is not None)
    return args


def load_matrix(path, shape=None):
    mat = pd.read_csv(path).values
    
    if mat.shape[1] == 2:
        rows, cols = mat.T
        data = np.ones(rows.shape[0])
        
        if shape is None:
            shape = max(rows.max(), cols.max()) + 1
        else:
            implied_shape = max(rows.max(), cols.max()) + 1
            if implied_shape > shape:
                raise Exception('implied_shape > shape: switch order of inputs')
        
        mat = sparse.csr_matrix((data, (rows, cols)), shape=(shape, shape))
        mat = mat.todense()
    elif mat.shape[1] == 3:
        raise NotImplemented
    
    mat = mat.astype(np.float64)
    mat = torch.Tensor(mat)
    assert mat.size(0) == mat.size(1), "%s must be square" % path
    return mat


def initialize(A, num_nodes, num_seeds):
    P = torch.eye(num_nodes)
    P[num_seeds:, num_seeds:] = 0
    
    perm = np.arange(P.shape[0])
    perm[num_seeds:] = np.random.permutation(perm[num_seeds:])
    perm = torch.LongTensor(perm)
    A = A[perm][:,perm]
    
    return A, P


class TorchSGM(BaseSGM):
    def __init__(self, mode, cuda, eps, eye):
        self.mode = mode
        self.cuda = cuda
        self.eps  = eps
        self.eye  = eye
        
        self.lap_calls = 0
    
    def solve_lap(self, cost):
        if self.lap_calls == 0:
            idx = cost.max(dim=1)[1]
        else:
            cost = cost.cpu().numpy()
            _, idx, _ = lapjv(cost.max() - cost)
            idx = torch.LongTensor(idx.astype(int))
            if self.cuda:
                idx = idx.cuda()
        
        self.lap_calls += 1
        return self.eye[idx]
        
        # dim = cost.size(0)
        # cost = cost - cost.min() # Make >= 0
        # if self.mode == 'jv':
        #     cost = cost.cpu().numpy()
        #     _, idx, _ = lapjv(cost.max() - cost)
        #     idx = torch.LongTensor(idx.astype(int))
        #     if self.cuda:
        #         idx = idx.cuda()
        
        # elif self.mode == 'auction':
        #     _, idx, _ = auction_lap(cost, eps=self.eps)
        
        # elif self.mode == 'greedy':
        #     idx = cost.max(dim=1)[1]
        # else:
        #     raise Exception
        
        # return self.eye[idx]
    
    def compute_grad(self, A, P, B):
        AP = torch.mm(A, P)
        return 4 * torch.mm(AP, B) - 2 * (AP.sum(dim=-1).view(-1, 1) + B.sum(dim=0).view(1, -1)) + A.size(0)
    
    def prod_sum(self, x, y):
        return (x * y).sum()


if __name__ == '__main__':
    args = parse_args()
    
    A = load_matrix(args.A_path)
    num_nodes = A.size(0)
    
    B = load_matrix(args.B_path, shape=num_nodes)
    
    # Best match distance
    f_gold = np.sqrt(((A - B) ** 2).sum())
    
    # --
    # Initialize
    
    if args.P_path:
        P = load_matrix(args.P_path, shape=num_nodes)
        num_seeds = (P.diagonal() == 1).sum()
    else:
        num_seeds = args.num_seeds
        A, P = initialize(A, num_nodes=num_nodes, num_seeds=num_seeds)
    
    f_init = np.sqrt(((A - B) ** 2).sum())
    
    # --
    # Run
    
    eye = torch.eye(num_nodes)
    if args.cuda:
        A, B, P, eye = A.cuda(), B.cuda(), P.cuda(), eye.cuda()
        
    torch_sgm = TorchSGM(
        mode=args.lap_mode,
        cuda=args.cuda,
        eps=args.auction_eps,
        eye=eye
    )
    
    t = time()
    P_out = torch_sgm.run(
        A=A,
        P=P,
        B=B,
        num_iters=args.num_iters,
        tolerance=args.tolerance,
    )
    compute_time = time() - t
    
    # --
    # Save results
    
    B_perm = torch.mm(torch.mm(P_out, B), P_out.t())
    f_perm = np.sqrt(((A - B_perm) ** 2).sum())
    
    print(json.dumps({
        "f_gold"       : float(f_gold),
        "f_init"       : float(f_init),
        "f_perm"       : float(f_perm),
        
        "compute_time" : float(compute_time),
        
        "lap_mode"     : args.lap_mode,
        "auction_eps"  : args.auction_eps if args.lap_mode == 'auction' else None,
        "num_nodes"    : int(num_nodes),
        "num_seeds"    : int(num_seeds),
    }))
