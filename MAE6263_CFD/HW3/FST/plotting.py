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
        'size'   : 16}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

re = 1000
x = 1.0

fig, ax = plt.subplots(3,3,figsize=(15,12.5))
axs = ax.flat

i = 0
re_list = [100,400,1000]

for i in range(3):
    re = re_list[i]
    for nx,ny in zip([50,100,200],[50,100,200]):
        
        data = np.load(f'solution_fst_{int(re)}_{nx}_{ny}.npz')
        
        ax[i,0].semilogy(data['kc'],data['rs'],label=f'$\Delta x = {x/nx}$')
        ax[i,1].semilogy(data['kc'],data['rw'],label=f'$\Delta x = {x/nx}$')
        ax[i,2].plot(data['uc'],data['y'],lw=2,label=f'$\Delta x = {x/nx}$')
        if nx == 200:
            ax[i,2].plot(data['expt_data'][:,1],data['expt_data'][:,0],'ko',
                    fillstyle='none',mew=2,ms=8, label='Ghia et al.')
    

xlabels = ['$n$', '$n$', '$u$']
ylabels = ['$r_\psi$', '$r_\omega$', '$y$']

for i in range(3):
    ax[i,0].set_xlabel(xlabels[0])
    ax[i,1].set_xlabel(xlabels[1])
    ax[i,2].set_xlabel(xlabels[2])
    ax[i,0].set_ylabel(ylabels[0])
    ax[i,1].set_ylabel(ylabels[1])
    ax[i,2].set_ylabel(ylabels[2])   
    ax[i,0].set_ylim([1e-8,1e-2])
    ax[i,1].set_ylim([1e-8,1e2])
    ax[i,0].legend()
    ax[i,1].legend()
    ax[i,2].legend()
    
fig.tight_layout()
plt.show()   
fig.savefig(f'plots_fst.png', dpi=300) 

#%%
fig, ax = plt.subplots(3,3,figsize=(15,12.5))

i = 0
re_list = [100,400,1000]
nx_list = [50,100,200]
vmax_list = [10,10,10]
vmin_list = [-10,-10,-10]

for i in range(3):
    re = re_list[i]
    for j in range(3):
        nx = nx_list[j]
        
        data = np.load(f'solution_fst_{int(re)}_{nx}_{nx}.npz')
        
        levels = np.linspace(vmin_list[i], vmax_list[i], 121)
#        cs = ax[i,j].contourf(data['X'],data['Y'],data['w'], 40, levels=levels, 
#               cmap = 'RdBu',extend='both')#,zorder=-20)
        cs = ax[i,j].contour(data['X'],data['Y'],data['w'], 120, colors='black')
        cs = ax[i,j].imshow(data['w'].T, extent=[0, 1, 0, 1], origin='lower',
                   vmin=vmin_list[i], vmax = vmax_list[i], 
                   interpolation='bicubic',cmap='jet', alpha=1.0)
        ax[i,j].set_rasterization_zorder(-10)
        
        cbarlabels = np.linspace(vmin_list[j], vmax_list[j], num=6, endpoint=True)
        cbar = fig.colorbar(cs, ax=ax[i,j], shrink=1.0, orientation='vertical')
        cbar.set_ticks(cbarlabels)
        cbar.ax.tick_params(labelsize=10)
#        cbar.set_ticklabels(['{:.2f}'.format(x) for x in cbarlabels])
        
        

fig.tight_layout()
plt.show()    
fig.savefig(f'contour_fst_w.png', dpi=300) 

#%%
fig, ax = plt.subplots(3,3,figsize=(15,12.5))

i = 0
re_list = [100,400,1000]
nx_list = [50,100,200]
vmax_list = [0.,0,0]
vmin_list = [-0.1,-0.1,-0.1]

for i in range(3):
    re = re_list[i]
    for j in range(3):
        nx = nx_list[j]
        
        data = np.load(f'solution_fst_{int(re)}_{nx}_{nx}.npz')
        
        levels = np.linspace(vmin_list[i], vmax_list[i], 61)
#        cs = ax[i,j].contourf(data['X'],data['Y'],data['s'], 40, levels=levels, 
#               cmap = 'RdBu',extend='both',zorder=-20)
        cs = ax[i,j].contour(data['X'],data['Y'],data['s'], 10, colors='black')
        cs = ax[i,j].imshow(data['s'].T, extent=[0, 1, 0, 1], origin='lower',
                   vmin=vmin_list[i], vmax = vmax_list[i], 
                   interpolation='bicubic',cmap='jet', alpha=1.0)
        ax[i,j].set_rasterization_zorder(-10)
        
        cbarlabels = np.linspace(vmin_list[j], vmax_list[j], num=6, endpoint=True)
        cbar = fig.colorbar(cs, ax=ax[i,j], shrink=1.0, orientation='vertical')
        cbar.set_ticks(cbarlabels)
        cbar.ax.tick_params(labelsize=10)
#        cbar.set_ticklabels(['{:.2f}'.format(x) for x in cbarlabels])
        
        

fig.tight_layout()
plt.show()    
fig.savefig(f'contour_fst_s.png', dpi=300) 



    