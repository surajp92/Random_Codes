#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:13:02 2021

@author: suraj
"""

import numpy as np
from getErrElem1D import *

def getErr1D(uh, f, mesh, fem, ng, dind):
    nt = np.shape(mesh.t)[0]
    pd = fem.pd
    errK = np.zeros((nt,1))
    
    for k in range(nt):
        vert = mesh.p[mesh.t[k,:]]
        
        uhK = uh[fem.t[k,:]] #.flatten()
        
        errK[k,:] = getErrElem1D(uhK, f, vert, pd, dind, ng)
    
    err = np.sqrt(np.sum(errK * errK))
    
    return err, errK