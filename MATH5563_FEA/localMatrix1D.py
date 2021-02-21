#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:36:56 2021

@author: suraj
"""

import numpy as np
from gaussQuad1D import *
from bas1DP import *

def localMatrix1D(fun, vert, pd1, d1, pd2, d2, ng):
    mat = np.zeros((pd1+1, pd2+1))
    gw, gx = gaussQuad1D(vert, ng)
    
    fv = fun(gx)
    
    for i in range(pd1+1):
        Li = bas1DP(gx, vert, pd1, i+1, d1)
        for j in range(pd2+1):
            Lj = bas1DP(gx, vert, pd2, j+1, d2)
            mat[i,j] = np.sum(gw*fv*Li*Lj)
    
    return mat