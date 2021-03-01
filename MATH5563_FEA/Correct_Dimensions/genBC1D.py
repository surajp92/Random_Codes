#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:51:27 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from genMesh1D import *
from genFEM1D import *

'''
% Usage: enrich FEM structure with node type and degree of freedom

INPUTS:
fem --- struct data contains  
domain --- [xmin, xmax];
bc --- [bcL,bcR] boundary condition at left and right boundary
        bcL (bcR) = 1 Dirichlet BC on the left (right) boundary
        bcL (bcR) = 2 Neumann BC on the left (right) boundary

OUTPUTS:
fem.ptype --- node type of fem.p, same size as fem.p
              0: if the node is internal node
              1: Dirichlet boundary node
              2: Neumann boundary node
fem.dof ---  degree of freedom info, where the unknowns are. 
'''

def genBC1D(fem, domain, bc):
    
    ptype = np.zeros(np.shape(fem.p))
    ptype[np.abs(fem.p - domain[0]) < np.finfo(float).eps] = bc[0]
    ptype[np.abs(fem.p - domain[1]) < np.finfo(float).eps] = bc[1]