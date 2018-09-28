#!/usr/bin/env python

"""
    plot.py
"""

import json
import numpy as np
import pandas as pd

from rsub import *
from matplotlib import pyplot as plt

# inpath = open(sys.argv[1])
inpath = 'kasios-2.jl'
x = [json.loads(xx) for xx in open(inpath).read().splitlines()]

df = pd.DataFrame(x)

_ = plt.scatter(df.num_nodes, df.compute_time, alpha=0.6)
_ = plt.xlabel('num_nodes')
_ = plt.ylabel('compute_time')
show_plot()

df['grad_time']  = df.times.apply(lambda x: np.sum(x['grad']))
df['lap_time']   = df.times.apply(lambda x: np.sum(x['lap']))
df['check_time'] = df.times.apply(lambda x: np.sum(x['check']))

_ = plt.scatter(df.num_nodes, df.grad_time, alpha=0.5, label='grad_time')
_ = plt.scatter(df.num_nodes, df.lap_time, alpha=0.5, label='lap_time')
_ = plt.scatter(df.num_nodes, df.check_time, alpha=0.5, label='check_time')
_ = plt.xlabel('num_nodes')
_ = plt.ylabel('compute_time')
_ = plt.legend()
show_plot()