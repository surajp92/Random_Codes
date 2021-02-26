#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:15:20 2021

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(1)
import pyfftw
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import os
from numba import jit
from scipy.fftpack import dst, idst

from scipy.ndimage import gaussian_filter
import yaml

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


#%%
def fst(nx,ny,dx,dy,f):
    data = f[1:-1,1:-1]
        
#    e = dst(data, type=2)
    data = dst(data, axis = 1, type = 1)
    data = dst(data, axis = 0, type = 1)
    
    m = np.linspace(1,nx-1,nx-1).reshape([-1,1])
    n = np.linspace(1,ny-1,ny-1).reshape([1,-1])
    
    data1 = np.zeros((nx-1,ny-1))
    
#    for i in range(1,nx):
#        for j in range(1,ny):
    alpha = (2.0/(dx*dx))*(np.cos(np.pi*m/nx) - 1.0) + (2.0/(dy*dy))*(np.cos(np.pi*n/ny) - 1.0)           
    data1 = data/alpha
    
#    u = idst(data1, type=2)/((2.0*nx)*(2.0*ny))
    data1 = idst(data1, axis = 1, type = 1)
    data1 = idst(data1, axis = 0, type = 1)
    
    u = data1/((2.0*nx)*(2.0*ny))
    
    ue = np.zeros((nx+1,ny+1))
    ue[1:-1,1:-1] = u
    
    return ue

#%% 
if __name__ == "__main__":
    nx = 512
    ny = 512
    
    n_level = 8
    max_iterations = 15
    v1 = 2
    v2 = 2
    v3 = 2
    tolerance = 1e-6
    
    x_l = 0.0
    x_r = 1.0
    y_b = 0.0
    y_t = 1.0
    
    dx = (x_r-x_l)/nx
    dy = (y_t-y_b)/ny
    
    x = np.linspace(x_l, x_r, nx+1)
    y = np.linspace(y_b, y_t, ny+1)
    
    xm, ym = np.meshgrid(x,y, indexing='ij')
    
    km = 16.0
    c1 = (1.0/km)**2
    c2 = -2.0*np.pi**2
    
    ue = np.sin(2.0*np.pi*xm) * np.sin(2.0*np.pi*ym) + \
                c1*np.sin(16.0*np.pi*xm) * np.sin(16.0*np.pi*ym)
    
    f = 4.0*c2*np.sin(2.0*np.pi*xm) * np.sin(2.0*np.pi*ym) + \
             c2*np.sin(16.0*np.pi*xm) * np.sin(16.0*np.pi*ym)
                     
    # ue = np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
    #      c1*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)
    
    # f = 4.0*c2*np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
    #     c2*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)
             
    un = fst(nx,ny,dx,dy,f)    
    
    fig, axs = plt.subplots(1,2,figsize=(14,5))
    cs = axs[0].contourf(xm, ym, ue, 60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[0], orientation='vertical')
    
    cs = axs[1].contourf(xm, ym, un,60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[1], orientation='vertical')
    
    plt.show()
    fig.tight_layout()
#    fig.savefig('fst.png', bbox_inches = 'tight', pad_inches = 0, dpi = 300)
    
    print(np.linalg.norm(un-ue)/np.sqrt((nx*ny)))  