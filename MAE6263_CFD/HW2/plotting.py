#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:04:27 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 18}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

re = 1000
x = 1.0

fig, ax = plt.subplots(2,2,figsize=(12,10))
axs = ax.flat

for nx,ny in zip([50,100,200],[50,100,200]):
    data = np.load(f'solution_{int(re)}_{nx}_{ny}.npz')
    
    axs[0].semilogy(data['kc'],data['ru'],label=f'$\Delta x = {x/nx}$')
    axs[1].semilogy(data['kc'],data['rv'],label=f'$\Delta x = {x/nx}$')
    axs[2].semilogy(data['kc'],data['rp'],label=f'$\Delta x = {x/nx}$')
    uc = 0.5*(data['u'][int(ny/2),0:ny+1] + data['u'][int(ny/2),1:ny+2])
    axs[3].plot(uc,data['Y'][:,int(nx/2)],lw=3,fillstyle='none',
                mew=1,ms=8,label=f'$\Delta x = {x/nx}$')

axs[3].plot(data['expt_data'][:,1],data['expt_data'][:,0],'ko',
            fillstyle='none',mew=2,ms=10, label='Experiment data')

xlabels = ['$n$', '$n$', '$n$', '$u$']
ylabels = ['$r_u$', '$r_v$', '$r_p$', '$y$']

for k in range(4):
    axs[k].set_xlabel(xlabels[k])
    axs[k].set_ylabel(ylabels[k])    
    axs[k].legend()
    
fig.tight_layout()
plt.show()   
fig.savefig(f'plots_{re}.png', dpi=300) 


#%%
fig,ax = plt.subplots(1,2, figsize=(12,5))

cbarticks = np.arange(-0.2,1.2,0.2)
levels = np.linspace(-0.2,1.0, 101)
cs = ax[0].contourf(data['X'],data['Y'],data['u'][:,1:ny+2].T, 
                    levels=levels,cmap='jet', extend='both')
fig.colorbar(cs, ax=ax[0], ticks = cbarticks)

cbarticks = np.arange(-0.5,0.7,0.2)
levels = np.linspace(-0.5,0.5, 101)
cs = ax[1].contourf(data['X'],data['Y'],data['v'][1:nx+2,:].T, 
                    levels=levels,cmap='jet', extend='both')
fig.colorbar(cs, ax=ax[1], ticks = cbarticks)

for k in range(2):
    ax[k].set_xlabel('$x$')
    ax[k].set_ylabel('$y$')   
    
fig.tight_layout()
plt.show()    
fig.savefig(f'contour_{re}.png', dpi=300) 
    