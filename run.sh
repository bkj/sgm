#!/bin/bash

# run.sh

# --
# Get data

tar -xzvf data.tar.gz

# --
# Run SGM

time python sgm.py

# --
# Run SGM w/ cuda

time python sgm.py --cuda

# --
# RUN SGM optimized for special case of `m=0`

time python sgm0.py