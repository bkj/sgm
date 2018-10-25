#!/usr/bin/env python

"""
    sgm/__init__.py
"""

from .backends.classic import (
    ScipyAuctionClassicSGM, 
    ScipyJVClassicSGM,
    TorchAuctionClassicSGM, 
    TorchJVClassicSGM,
)

from .backends.fused import AuctionFusedSGM, JVFusedSGM
from .backends.sparse import AuctionSparseSGM, JVSparseSGM

__backends = {
    "scipy" : {
        "classic" : {
            "auction" : ScipyAuctionClassicSGM,
            "jv"      : ScipyJVClassicSGM,
        },
        "fused" : {
            "auction" : AuctionFusedSGM,
            "jv"      : JVFusedSGM,
        },
        "sparse" : {
            "auction" : AuctionSparseSGM,
            "jv"      : JVSparseSGM,
        },
    },
    "torch" : {
        "classic" : {
            "auction" : TorchAuctionClassicSGM,
            "jv"      : TorchJVClassicSGM,
        }
    }
}

def factory(mat, mode, lap):
    
    assert mat in __backends
    __backend = __backends[mat]
    
    assert mode in __backend
    __backend = __backend[mode]
    
    assert lap in __backend
    __backend = __backend[lap]
    
    return __backend