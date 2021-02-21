#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:42:05 2021

@author: suraj
"""

import numpy as np
from localMatrix1D import *
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix

def globalMatrix1D(f, mesh, fem1, d1, fem2, d2, ng):
    pd1 = fem1.pd
    pd2 = fem2.pd
    
    M = csc_matrix((np.shape(fem1.p)[0], np.shape(fem2.p)[0]))
    # M = lil_matrix((np.shape(fem1.p)[0], np.shape(fem2.p)[0]))
    
    for k in range(np.shape(mesh.t)[0]):
        vert = mesh.p[mesh.t[k,:]]
        matK = localMatrix1D(f, vert, pd1, d1, pd2, d2, ng)
        
        ii,jj = np.meshgrid(fem1.t[k,:], fem2.t[k,:], indexing='ij')
        M[ii, jj] += matK 
    
    return M
    
    