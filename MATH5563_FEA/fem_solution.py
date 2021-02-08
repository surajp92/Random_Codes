#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 15:50:30 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

from genMesh1D import *
from genFEM1D import *
from evalFEfun1D import *

domain = [0,1]
n = 5
pd = 1

mesh = genMesh1D(domain, n)
fem = genFEM1D(mesh, pd)
uh = np.array([2,3,4,4,3,2])
dind = 0

for k in range(n):
    vert = mesh.p[mesh.t[k,:]]
    x = np.linspace(vert[0], vert[1], 101)
    
    uhK = uh[fem.t[k,:]]
    u = evalFEfun1D(x, uhK, vert, pd, dind)
    
    plt.plot(x,u,'b')
    
plt.grid()
plt.xlim(domain)    
plt.show()

    