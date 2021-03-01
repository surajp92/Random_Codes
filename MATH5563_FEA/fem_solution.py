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

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True


domain = [0,1]
n = 5
pd = 1

mesh = genMesh1D(domain, n)
fem = genFEM1D(mesh, pd)
uh = np.array([2,3,4,4,3,2])
dind = 0

fig, ax = plt.subplots(1,1,sharex=True,figsize=(6,5))

    
for k in range(n):
    vert = mesh.p[mesh.t[k,:]]
    x = np.linspace(vert[0], vert[1], 101).flatten()
    
    uhK = uh[fem.t[k,:]]
    u = evalFEfun1D(x, uhK, vert, pd, dind)
    
    ax.plot(x,u,'b',lw=2)
    
ax.grid()
ax.legend()
ax.set_xlim(domain)   
ax.set_xlabel('$x$')
ax.set_ylabel('$U_h(x)$') 
plt.show()
fig.savefig('fem_solution.png', dpi=300, bbox_inches="tight")

#%%
fig, ax = plt.subplots(3,2,sharex=False,figsize=(11,12))    
axs = ax.flat

for j in range(6):
    uh = np.zeros(6)
    uh[j] = 1.0
    
    for k in range(n):
        vert = mesh.p[mesh.t[k,:]]
        x = np.linspace(vert[0], vert[1], 101)
        
        uhK = uh[fem.t[k,:]]
        u = evalFEfun1D(x, uhK, vert, pd, dind)
        
        axs[j].plot(x,u,'b',lw=2)
        axs[j].grid()
        axs[j].set_xlim(domain)  
        axs[j].set_xlabel('$x$')    
        axs[j].set_ylabel(f'$\phi_{j+1}(x)$') 

plt.show()
fig.savefig('fem_functions.png', dpi=300, bbox_inches="tight")
    