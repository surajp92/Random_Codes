#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:25:41 2021

@author: suraj
"""

import numpy as np
from gaussQuad1D import *
from bas1DP import *

def localVec1D(fun, vert, pd, dind, ng):
    b = np.zeros((pd+1,1))
    gw, gx = gaussQuad1D(vert, ng)
    
    fv = fun(gx)
    
    for i in range(pd+1):
        Li = bas1DP(gx, vert, pd, i+1, dind)
        b[i] = np.sum(gw*fv*Li)
        
    return b