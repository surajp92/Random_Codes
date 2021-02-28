#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 17:32:23 2021

@author: suraj
"""

import numpy as np
from scipy.sparse.linalg import inv
import matplotlib.pyplot as plt

from genMesh1D import *
from genFEM1D import *
from evalFEfun1D import *
from getErr1D import *
from pdeEx1 import *
from getErr1D import *
from globalVec1D import *
from globalMatrix1D import *

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
from matplotlib.lines import Line2D

#%% 1 Domain, PDE, BC
domain = [0,1]
pde = pdeEx1()
bc = [1,2] # Dirichlet on left and Neumann on Right side

#%% 2 Generate Mesh and FEM
n = 20
pd = 2
ng = 2

mesh = genMesh1D(domain, n)
fem = genFEM1D(mesh, pd)

#%% 3 Node type and DOF
fem.genBC1D(domain, bc)

#%% 4 Generate Matrices abd Vectors
Mxx = globalMatrix1D(pde.a, mesh, fem, 1, fem, 1, ng)
M00 = globalMatrix1D(pde.c, mesh, fem, 0, fem, 0, ng)
f = globalVec1D(pde.f, mesh, fem, 0, ng)

#%% 5 Dirichlet boundary condition
uD = pde.gD(fem.p)
uD[fem.dof] = 0.0
bxx = Mxx @ uD
b00 = M00 @ uD

#%% 6 Neumann boundary condition
bN = np.zeros(np.shape(fem.p))
bN[fem.ptype == 2] = pde.gN(fem.p[fem.ptype == 2])

#%% 7 Extract degree of freedom
ii,jj = np.meshgrid(fem.dof,fem.dof,indexing='ij')
A = Mxx[ii,jj] + M00[ii,jj]
b = f[fem.dof] + bN[fem.dof] -bxx[fem.dof] - b00[fem.dof]

#%% 8 Solve Linear Sysetem
uDof = inv(A) @ b
uh = np.copy(uD)
uh[fem.dof] = uDof      

#%%
fig, axs = plt.subplots(1,1,sharex=True,sharey=True,figsize=(8,6))
tu = pde.exactu(fem.p)
for k in range(n):
    vert = mesh.p[mesh.t[k,:]]
    x = np.linspace(vert[0], vert[1], 11)
    
    uhK = uh[fem.t[k,:]]
    u = evalFEfun1D(x, uhK, vert, pd, 0)
    
    axs.plot(x,pde.exactu(x),'k',lw=2)
    axs.plot(x,u,'r--',lw=2)
    
axs.grid()
custom_lines = [Line2D([0], [0], color='k', lw=2),
                    Line2D([0], [0], color='r',ls='--', lw=2)]
axs.legend(custom_lines, ['$u$','$Iu$'])
axs.set_xlim(domain)   
axs.set_xlabel('$x$')
plt.show()

#%%
errL2, errKL2 = getErr1D(uh, pde.exactu, mesh, fem, ng, 0)
errH1, errKH2 = getErr1D(uh, pde.Du, mesh, fem, ng, 1)

print('#----------------------------------------#')
print('n = %d' % (n))
print('L2 error:  %5.3e' % errL2)
print('H1 error:  %5.3e' % errH1)