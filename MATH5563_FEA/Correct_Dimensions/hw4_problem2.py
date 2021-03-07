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

import warnings
warnings.filterwarnings("ignore")

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
from matplotlib.lines import Line2D

import sys
orig_stdout = sys .stdout

class pdePr1:
    def __init__(self):
        
        
        # excact solution
        self.exactu = lambda x: (np.exp(3.0 - 2.0*x) + 2.0*np.exp(x))/(2.0 + np.e**3)

        # right hand side
        self.f = lambda x : np.zeros(np.shape(x))
        
        # Dirichlet boundary condition
        self.gD = lambda x : self.exactu(x)
        
        # Neumann boundary condition
        self.gN = lambda x : self.a(x)*self.Du(x)
        
        # Derivative of the exact solution
        self.Du = lambda x : (-2.0*np.exp(3.0 - 2.0*x) + 2.0*np.exp(x))/(2.0 + np.e**3)
        
        # Diffusion coefficient function
        self.a = lambda x : np.ones(np.shape(x))
        
        # first order term coefficient
        self.b = lambda x : -1.0*np.ones(np.shape(x))
        
        # Reacation coefficient function
        self.c = lambda x : 2.0*np.ones(np.shape(x))


#%% 1 Domain, PDE, BC
domain = [0,1]
pde = pdePr1()
bc = [1,0] # Dirichlet on left and Neumann on Right side
error_list= []
coarse = False

if coarse:
    fig, ax = plt.subplots(1,2,sharex=True,figsize=(12,5))
    nc = 1
    pd_list = [1]
else:
    fig, ax = plt.subplots(3,5,sharex=True,figsize=(18,15))
    nc = 5
    pd_list = [1,2,3]
    
axs = ax.flat
count = 0

for pd in pd_list:
    file = open(f'./logs/hw4_problem2_fem_out_{pd}.log', 'w')
    sys.stdout = file
    for i in range(nc):    
        #%% 2 Generate Mesh and FEM
        n = int(5*(2**i))
        ng = 5
        
        mesh = genMesh1D(domain, n)
        fem = genFEM1D(mesh, pd)
        
        #%% 3 Node type and DOF
        fem.genBC1D(domain, bc)
        
        #%% 4 Generate Matrices abd Vectors
        Mxx = globalMatrix1D(pde.a, mesh, fem, 1, fem, 1, ng)
        M0x = globalMatrix1D(pde.b, mesh, fem, 0, fem, 1, ng)
        M00 = globalMatrix1D(pde.c, mesh, fem, 0, fem, 0, ng)
        f = globalVec1D(pde.f, mesh, fem, 0, ng)
        
        #%% 5 Dirichlet boundary condition
        uD = pde.gD(fem.p)
        uD[fem.dof] = 0.0
        bxx = Mxx @ uD
        b0x = M0x @ uD
        b00 = M00 @ uD
        
        #%% 6 Neumann boundary condition
        bN = np.zeros(np.shape(fem.p))
        bN[fem.ptype == 2] = pde.gN(fem.p[fem.ptype == 2])
        
        #%% 7 Extract degree of freedom
        ii,jj = np.meshgrid(fem.dof,fem.dof,indexing='ij')
        A = Mxx[ii,jj] + M0x[ii,jj] + M00[ii,jj]
        b = f[fem.dof] + bN[fem.dof] - bxx[fem.dof] - b0x[fem.dof] - b00[fem.dof]
        
        #%% 8 Solve Linear Sysetem
        uDof = inv(A) @ b
        uh = np.copy(uD)
        uh[fem.dof] = uDof      
        
        #%%
        
        tu = pde.exactu(fem.p)
        for k in range(n):
            vert = mesh.p[mesh.t[k,:]]
            x = np.linspace(vert[0], vert[1], 11)
            
            uhK = uh[fem.t[k,:]]
            u = evalFEfun1D(x, uhK, vert, pd, 0)
            
            axs[count].plot(x,pde.exactu(x),'k',lw=2 )
            axs[count].plot(x,u,'r--',lw=2)
            if coarse:
                axs[count+1].plot(x,pde.exactu(x) -u,'b',lw=2)
            
        axs[count].grid()
        custom_lines = [Line2D([0], [0], color='k', lw=2),
                            Line2D([0], [0], color='r',ls='--', lw=2)]
        axs[count].legend(custom_lines, ['$u$','$Fu$'])
        if coarse:
            custom_lines = [Line2D([0], [0], color='b', lw=2)]
            axs[count+1].legend(custom_lines, ['$u - Fu$'])
        axs[count].set_xlim(domain)   
        axs[count].set_xlabel('$x$')
        axs[count].set_title(f'PD={pd}, $N$={n}')
        
        count = count + 1
            
        #%%
        errL2, errKL2 = getErr1D(uh, pde.exactu, mesh, fem, ng, 0)
        errH1, errKH2 = getErr1D(uh, pde.Du, mesh, fem, ng, 1)
        
        error_list.append([pd, n, errL2, errH1])
        
        print('#----------------------------------------#')
        print('n = %d' % (n))
        print('L2 error:  %5.3e' % errL2)
        print('H1 error:  %5.3e' % errH1)
        if i>=1:
            rateL2 = np.log(errL2_0/errL2)/np.log(2.0);
            rateH1 = np.log(errH1_0/errH1)/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
            print('H1 order:  %5.3f' % rateH1)   
        
        errL2_0 = errL2
        errH1_0 = errH1

plt.show()
if coarse:
    fig.savefig('./logs/hw4_problem2_fem_u1.png', dpi=300, bbox_inches="tight")
else:
    fig.savefig('./logs/hw4_problem2_fem_u.png', dpi=300, bbox_inches="tight")
    
#%%
if not coarse:
    fig, axs = plt.subplots(1,2,sharex=True,figsize=(12,5))
    error_list = np.array(error_list)
    
    for pd in [1,2,3]:
        slice_ = error_list[:,0] == pd
        axs[0].loglog(error_list[slice_, 1], error_list[slice_, 2],'s-', label=f'PD = {pd}')
        axs[1].loglog(error_list[slice_, 1], error_list[slice_, 3], 's-', label=f'PD = {pd}')
    
    axs[0].set_ylim([1e-14,1e0])
    axs[1].set_ylim([1e-14,1e2])
    axs[0].set_xscale('log', basex=2)
    axs[1].set_xscale('log', basex=2)
    axs[0].legend()
    axs[1].legend()
    plt.show()
    fig.tight_layout()
    fig.savefig(f'./logs/hw4_problem2_fem.png', dpi=300)    