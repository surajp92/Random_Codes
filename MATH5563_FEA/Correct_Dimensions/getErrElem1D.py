#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:04:46 2021

@author: suraj
"""

import numpy as np
from gaussQuad1D import *
from evalFEfun1D import *

def getErrElem1D(uhK, fun, vert, pd, dind, ng):
    gw, gx = gaussQuad1D(vert, ng)
    
    tu = fun(gx)
    
    uh = evalFEfun1D(gx, uhK, vert, pd, dind)
    
    errK = np.sqrt(np.sum(gw*(tu - uh)**2))
    
    return errK
    