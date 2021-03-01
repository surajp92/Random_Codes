#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:49:22 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import inv

from genMesh1D import *
from genFEM1D import *
from evalFEfun1D import *
from getErr1D import *
from globalVec1D import *
from globalMatrix1D import *

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

import sys
orig_stdout = sys .stdout

domain = [0,1]
pd = 2
ng = 5
dind = 0

fun = lambda x: x**(2.5)
dfun = lambda x: 2.5*x**(1.5)
onefun = lambda x : 1.0

error_list = []
fig, axs = plt.subplots(4,5,sharex=True,sharey=True,figsize=(20,16))

for pd in [1,2,3,4]:
    file = open(f'./logs/hw2_problem2_projection_out_{pd}.log', 'w')
    sys.stdout = file

    for i in range(5):
        n = int(4*(2**i))
        mesh = genMesh1D(domain, n)
        fem = genFEM1D(mesh, pd)
        
        M = globalMatrix1D(onefun, mesh, fem, 0, fem, 0, ng)
        b = globalVec1D(fun, mesh, fem, 0, ng)
    
        uh = inv(M) @ b
        
        for k in range(n):
            vert = mesh.p[mesh.t[k,:]]
            x = np.linspace(vert[0], vert[1], 101)
            
            uhK = uh[fem.t[k,:]]
            u = evalFEfun1D(x, uhK, vert, pd, dind)
            
            axs[pd-1,i].plot(x,fun(x),'k',lw=2)
            axs[pd-1,i].plot(x,u,'r',lw=1)
            
        axs[pd-1,i].grid()
        custom_lines = [Line2D([0], [0], color='k', lw=2),
                        Line2D([0], [0], color='r', lw=1)]
        axs[pd-1,i].legend(custom_lines, ['$u$', '$Pu$'])
        axs[pd-1,i].set_xlim(domain) 
        axs[pd-1,i].set_title(f'$p$={pd}, $h$=1/{n}')
        if pd == 4:
            axs[pd-1,i].set_xlabel('$x$')
        if i == 0:
            axs[pd-1,i].set_ylabel('$u(x)$') 
            
        
        errL2, errKL2 = getErr1D(uh, fun, mesh, fem, ng, 0)
        errH1, errKH2 = getErr1D(uh, dfun, mesh, fem, ng, 1)
        
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

sys.stdout = orig_stdout
file.close()

plt.show()
fig.savefig('./logs/hw2_problem2_projection_u.png', dpi=300, bbox_inches="tight")


#%%
fig, axs = plt.subplots(1,2,sharex=True,figsize=(12,5))
error_list = np.array(error_list)

for pd in [1,2,3,4]:
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
fig.savefig(f'./logs/hw2_problem2_projection.png', dpi=300)