#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:17:22 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(1)
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
func1 = np.load('zdt1.npy')
func2 = np.load('zdt2.npy')
func3 = np.load('zdt3.npy')
func4 = np.load('zdt4.npy')
func6 = np.load('zdt6.npy')

#%%
vmin = -12
vmax = 12

field = [func1,func2,func3,func4,func6]
#label = ['True','True','True','EnKF','EnKF','EnKF','Error','Error','Error']
title = []
xlabel = [r'$f_1(x)$',r'$f_1(x)$',r'$f_1(x)$',r'$f_1(x)$',r'$f_1(x)$']
ylabel = [r'$f_2(x)$',r'$f_2(x)$',r'$f_2(x)$',r'$f_2(x)$',r'$f_2(x)$']

fig,ax = plt.subplots(figsize=(12,14),constrained_layout=True)

AX = gridspec.GridSpec(3,4)
AX.update(wspace = 0.4, hspace = 0.4)
axs1  = plt.subplot(AX[0,0:2])
axs2 = plt.subplot(AX[0,2:])
axs3 = plt.subplot(AX[1,0:2])
axs4 = plt.subplot(AX[1,2:])
axs5 = plt.subplot(AX[2,1:3])

#axs1 = plt.subplot2grid((4, 4), (0, 1), colspan=2)
#axs2 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
#axs3 = plt.subplot2grid((4, 4), (1, 2), colspan=2)
#axs4 = plt.subplot2grid((4, 4), (2, 0), colspan=2)
#axs5 = plt.subplot2grid((4, 4), (2, 2), colspan=2)
#axs6 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
#axs7 = plt.subplot2grid((4, 4), (3, 2), colspan=2)

axs = [axs1, axs2, axs3, axs4, axs5]

for i in range(5):
    function1 = [i[0] for i in field[i]]
    function2 = [i[1] for i in field[i]]
    axs[i].plot(function1, function2,'bo',fillstyle='none',ms=8,mew=2)
    axs[i].grid()
    if i == 4:
        axs[i].set_title(f'ZDT{i+2}',size=16)
    else:
        axs[i].set_title(f'ZDT{i+1}',size=16)
    axs[i].set_xlabel(xlabel[i],size=16)
    axs[i].set_ylabel(ylabel[i],size=16)

fig.tight_layout()
plt.show()  
#fig.savefig('problem1.pdf',bbox_inches='tight',dpi=300)
fig.savefig('problem1.png',bbox_inches='tight',dpi=150)

