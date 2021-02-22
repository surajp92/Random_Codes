#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 22:38:06 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

from genMesh1D import *
from genFEM1D import *
from evalFEfun1D import *
from getErr1D import *

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
from matplotlib.lines import Line2D

import sys
orig_stdout = sys .stdout

domain = [2,3]
pd = 1
ng = 5
dind = 0

fun = lambda x: np.sin(2.0*np.pi*x) + np.exp(-x)
dfun = lambda x: 2.0*np.pi*np.cos(2.0*np.pi*x) - np.exp(-x)

file = open(f'ex6_interpolation_out_{pd}.log', 'w')
# sys.stdout = file

fig, axs = plt.subplots(1,4,sharex=True,sharey=True,figsize=(16,4))

for i in range(4):
    n = int(5*(2**i))
    mesh = genMesh1D(domain, n)
    fem = genFEM1D(mesh, pd)
    
    uh = fun(fem.p)
    
    for k in range(n):
        vert = mesh.p[mesh.t[k,:]]
        x = np.linspace(vert[0], vert[1], 101)
        
        uhK = uh[fem.t[k,:]]
        u = evalFEfun1D(x, uhK, vert, pd, dind)
        
        axs[i].plot(x,fun(x),'k',lw=2)
        axs[i].plot(x,u,'r',lw=2)
        
    axs[i].grid()
    custom_lines = [Line2D([0], [0], color='k', lw=2),
                    Line2D([0], [0], color='r', lw=2)]
    axs[i].legend(custom_lines, ['$u$', '$Iu$'])
    axs[i].set_xlim(domain)   
    axs[i].set_xlabel('$x$')
    if i == 0:
        axs[i].set_ylabel('$U_h(x)$') 
    
    errL2, errKL2 = getErr1D(uh, fun, mesh, fem, ng, 0)
    errH1, errKH2 = getErr1D(uh, dfun, mesh, fem, ng, 1)
    
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
fig.savefig('ex6.png', dpi=300, bbox_inches="tight")
         