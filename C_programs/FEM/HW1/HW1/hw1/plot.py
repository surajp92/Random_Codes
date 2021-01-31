#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 10:24:51 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 18}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
data1 = np.genfromtxt('solution_1.txt', delimiter=',')
data2 = np.genfromtxt('solution_2.txt', delimiter=',')

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))

ax.plot(data1[:,0], data1[:,1],'ko-', fillstyle='none', label='Galerkin Solution $\psi_D=e^x$')
ax.plot(data2[:,0], data2[:,1],'rs-', fillstyle='none', label='Galerkin Solution $\psi_D=e^1$')
ax.grid()
ax.set_xlim([data1[0,0],data1[-1,0]])
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
ax.legend()

plt.show()
fig.savefig('u.png', dpi=300)