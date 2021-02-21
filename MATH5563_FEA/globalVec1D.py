#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:32:10 2021

@author: suraj
"""

import numpy as np
from localVec1D import *

def globalVec1D(f, mesh, fem, dind, ng):
    b = np.zeros((np.shape(fem.p)[0],1))
    pd = fem.pd
    
    for k in range(np.shape(mesh.t)[0]):
        vert = mesh.p[mesh.t[k,:]]
        bK = localVec1D(f, vert, pd, dind, ng)
        b[fem.t[k,:]] = b[fem.t[k,:]] + bK
    
    return b
     