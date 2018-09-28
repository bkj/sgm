#!/usr/bin/env python

"""
    sgm/examples/kasios/kasios-grid.py
"""

import sys
import json
import argparse
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from kasios import load_data, make_problem, run_experiment

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jv-backend', type=str, default='gatagat', choices=['srcd', 'gatagat'])
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    X, rw = load_data()
    
    def _experiment_wrapper(params):
        try:
            info, lap_times, iter_times = \
                run_experiment(
                    X=X, 
                    rw=rw, 
                    seed=args.seed,
                    jv_backend=args.jv_backend, 
                    verbose=False,
                    **params
                )
            info.update({
                "times" : {
                    "lap"  : lap_times,
                    "iter" : iter_times,
                }
            })
            print(json.dumps(info))
        except:
            print('error at %s' % json.dumps(params), file=sys.stderr)
            return
    
    param_grid = {
        "backend" : [
            "scipy.classic.jv",
            "scipy.sparse.jv",
            "scipy.fused.jv",
            "scipy.sparse.auction",
            "scipy.fused.auction",
            "torch.classic.jv",
        ],
        "num_nodes" : [
            500,
            1000,
            2000,
            4000,
            6000,
            8000,
            1000,
            1250,
            1500,
            1750,
            2000,
        ],
        "num_seeds" : [
            16,
            32,
            64,
            128,
        ]
    }
    
    param_list = list(itertools.product(*param_grid.values()))
    param_list = [dict(zip(param_grid.keys(), p)) for p in param_list]
    param_list = [param_list[i] for i in np.random.permutation(len(param_list))]
    
    for params in param_list:
        _experiment_wrapper(params)

