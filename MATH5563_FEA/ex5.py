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

from sympy import *

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
from matplotlib.lines import Line2D


domain = [2,3]
n = 5
pd = 4

mesh = genMesh1D(domain, n)
fem = genFEM1D(mesh, pd)
uh = np.array([2,3,4,4,3,2])
dind = 0

f = lambda x: np.sin(2.0*np.pi*x)

uh = f(fem.p) 

#%%
fig, ax = plt.subplots(1,1,sharex=True,figsize=(6,5))
    
for k in range(n):
    vert = mesh.p[mesh.t[k,:]]
    x = np.linspace(vert[0], vert[1], 101)
    
    uhK = uh[fem.t[k,:]]
    u = evalFEfun1D(x, uhK, vert, pd, dind)
    
    ax.plot(x,u,'r',lw=2)
    ax.plot(x,f(x),'k',lw=2)
    
ax.grid()
custom_lines = [Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], color='k', lw=2)]
ax.legend(custom_lines, ['$Iu$', '$u$'])

ax.set_xlim(domain)   
ax.set_xlabel('$x$')
ax.set_ylabel('$U_h(x)$') 
plt.show()
fig.savefig('ex5.png', dpi=300, bbox_inches="tight")
