#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 15:29:03 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from bas1DP import *

#%%
'''
USAGE: evaluate the FE function at certain point(s).
            f = u1*L1(x) + u2*L2(x) + ... + ud*Ld(x).
INPUTS:
x --- query point. x can be scalar, vectors.
uhK --- vector [u1,u2,...,ud] of coefficients of FE function.
         local dimension is pd+1, so uhK is a (pd+1) array. 
vert --- vertices of the element that contains the point(s) x
pd --- the polynomial degree of FE space 
dind --- derivative info for basis function
         dind = 0 if f = u1*L1'(x)+ u2*L2'(x)+... + ud*Ld'(x)
         dind = 1 if f = u1*L1(x) + u2*L2(x) +... + ud*Ld(x)
OUTPUTS:
f --- the value of the FE function at point x. 

'''

def evalFEfun1D(x, uhK, vert, pd, dind):
    f = np.zeros(np.shape(x)[0])
    
    for i in range(pd+1):
        ibas = i + 1
        l = bas1D(x,vert,ibas,dind,pd)
        f = f + uhK[i] * l
        
    return f
        
        
    