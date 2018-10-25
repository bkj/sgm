#!/usr/bin/env python

"""
    rw-sample.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/orig/calls.csv')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--max-nodes', type=int, default=100000)
    parser.add_argument('--max-steps', type=int, default=int(1e6))
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    np.random.seed(args.seed)
    data_type = os.path.basename(args.inpath).split('.')[0]
    
    # --
    # load and map edges
    
    print('rw-sample.py: loading %s' % args.inpath, file=sys.stderr)
    edges = pd.read_csv(args.inpath, header=None)
    edges = edges[[0, 2]].values
    
    print('rw-sample.py: remapping node ids', file=sys.stderr)
    unodes = set(np.hstack(edges))
    unodes = np.random.permutation(list(unodes))
    
    lookup = dict(zip(unodes, range(len(unodes))))
    
    edges[:,0] = [lookup.get(c) for c in edges[:,0]]
    edges[:,1] = [lookup.get(c) for c in edges[:,1]]
    
    print('rw-sample.py: saving data/%s.npy' % data_type, file=sys.stderr)
    np.save('data/%s.npy' % data_type, edges)
    
    # --
    # random walk sampling
    
    print('rw-sample.py: begin random walk', file=sys.stderr)
    g = nx.from_edgelist(edges)
    
    start_node = np.random.choice(g.nodes)
    node = start_node
    visited = set([])
    for _ in tqdm(range(args.max_steps)):
        neighbors = list(g.neighbors(node))
        node = np.random.choice(neighbors)
        if node not in visited:
            visited.add(node)
            print(node)
        
        if np.random.uniform(0, 1) < args.alpha:
            node = start_node
        
        if len(visited) >= args.max_nodes:
            break
        elif len(visited) % 10000 == 0:
            print(len(visited), file=sys.stderr)


