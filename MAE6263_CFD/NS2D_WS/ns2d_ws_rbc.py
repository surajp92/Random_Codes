#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 14:16:04 2020

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

from poisson import *
from rhs import *
from euler import *
from rk3 import *

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


#%%
# second-order bc
def bc(nx,ny,w,s):
    
    w[:,0] = -(2.0/dy**2)*(s[:,1]) # bottom wall
    w[:,ny] = -(2.0/dy**2)*(s[:,ny-1]) # top wall
    
    # w[0,:] = -(2.0/dx**2)*(s[1,:]) # left wall
    # w[nx,:] = -(2.0/dx**2)*(s[nx-1,:]) # right wall
    
    return w

# B.C. for vorticity (third-order)
def bc3(nx,ny,w,s):
    
    # bottom wall
    w[:,0] = (1.0/(18*dy**2))*(85.0*s[:,0] - 108.0*s[:,1] + \
                               27.0*s[:,2] - 4.0*s[:,3]) 
    # top wall
    w[:,ny] = (1.0/(18*dy**2))*(85.0*s[:,ny] - 108.0*s[:,ny-1] + \
                                27.0*s[:,ny-2] - 4.0*s[:,ny-3]) 
    
    # # left wall
    # w[0,:] = (1.0/(18*dx**2))*(85.0*s[0,:] - 108.0*s[1,:] + \
    #                            27.0*s[2,:] - 4.0*s[3,:]) 
    # # right wall
    # w[nx,:] = (1.0/(18*dx**2))*(85.0*s[nx,:] - 108.0*s[nx-1,:] + \
    #                             27.0*s[nx-2,:] - 4.0*s[nx-3,:]) 
    
    return w

#%% 
# read input file
with open(r'rbc_parameters.yaml') as file:
    input_data = yaml.load(file, Loader=yaml.FullLoader)
#    input_data = yaml.load(file)
    
file.close()

nx = input_data['nx']
ny = input_data['ny']
ra = float(input_data['ra'])
pr = input_data['pr']
nt = input_data['nt']
lx = input_data['lx']
ly = input_data['ly']
wc = input_data['wc']
wh = input_data['wh']
isolver = input_data['isolver']
ip = input_data['ip']
its = input_data['its']
eps = float(input_data['eps'])
freq = input_data['freq']
nlevel = input_data['nlevel']
pmax = input_data['pmax']
v1 = input_data['v1']
v2 = input_data['v2']
v3 = input_data['v3']
tolerance = float(input_data['tolerance'])


re = np.sqrt(ra/pr)

if ip == 1:
    directory = f'RBC_FST_{isolver}'
    if not os.path.exists(directory):
        os.makedirs(directory)

elif ip == 2:
    directory = f'MG_{isolver}'
    if not os.path.exists(directory):
        os.makedirs(directory)        

#%% 
pi = np.pi

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)

time = 0.0

x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

dtc = np.min((dx,dy))
dtv = 0.25*re*np.min((dx**2, dy**2))
sigma = 0.5
dt = sigma*np.min((dtc, dtv))

#%%
w = np.zeros((nx+1,ny+1)) 
s = np.zeros((nx+1,ny+1))
th = np.zeros((nx+1,ny+1))

w = 1.0e-3*np.sin(np.pi*X)*np.sin(np.pi*Y) #np.random.randn(nx-1,ny-1)

th = 1.0 - Y
th[:,0] = wh
th[:,ny] = wc

#%% 
w0 = np.copy(w)
s0 = np.copy(s)
th0 = np.copy(th)

kc = np.zeros(nt+1)
rw = np.zeros(nt+1)
rs = np.zeros(nt+1)
rth = np.zeros(nt+1)
   
clock_time_init = tm.time()

for k in range(nt+1):
    w0 = np.copy(w)
    s0 = np.copy(s)
    th0 = np.copy(th)
    
    if its == 1:    
        w,s,th = euler(nx,ny,dx,dy,dt,re,pr,w,s,th,input_data,bc,bc3)
    elif its == 2:
        w,s,th = rk3(nx,ny,dx,dy,dt,re,pr,w,s,th,input_data,bc,bc3)

    kc[k] = k
    rw[k] = np.linalg.norm(w - w0)/np.sqrt(np.size(w))
    rs[k] = np.linalg.norm(s - s0)/np.sqrt(np.size(s))
    rth[k] = np.linalg.norm(th - th0)/np.sqrt(np.size(th))
#    
    if k % freq == 0:
        print('%0.3i %0.3e %0.3e %0.3e' % (kc[k], rw[k], rs[k], rth[k]))

    if rw[k] <= eps and rs[k] <= eps and rth[k] <= eps:
        break

kc = kc[:k+1]
rw = rw[:k+1]
rs = rs[:k+1]
rth = rth[:k+1]
    
total_clock_time = tm.time() - clock_time_init
print('Total clock time=', total_clock_time)


#%%
fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.semilogy(kc, rw)
ax.semilogy(kc, rs)
ax.semilogy(kc, rth)
plt.show()

#%%
fig, axs = plt.subplots(3,1,figsize=(10,15))
cs = axs[0].contour(X,Y,w,40,vmin=-4.5, vmax=4.5,colors='black')
cs = axs[0].imshow(w.T,extent=[0, 2, 0, 1], origin='lower',
        interpolation='bicubic',cmap='RdBu', alpha=1.0)
#cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
fig.colorbar(cs, ax=axs[0], shrink=0.8, orientation='vertical')
axs[0].set_aspect('equal')

cs = axs[1].contour(X,Y,s,40,vmin=-0.12, vmax=0.12,colors='black')
cs = axs[1].imshow(s.T,extent=[0, 2, 0, 1], origin='lower',
        interpolation='bicubic',cmap='RdBu', alpha=1.0)
#cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
fig.colorbar(cs, ax=axs[1], shrink=0.8, orientation='vertical')
axs[1].set_aspect('equal')

cs = axs[2].contour(X,Y,th,40,vmin=-0, vmax=1,colors='black')
cs = axs[2].imshow(th.T,extent=[0, 2, 0, 1], origin='lower',
        interpolation='bicubic',cmap='RdBu', alpha=1.0)
#cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
fig.colorbar(cs, ax=axs[2], shrink=0.8, orientation='vertical')
axs[2].set_aspect('equal')

plt.show()
fig.tight_layout()
fig.savefig(f'rbc_ws_{its}_{isolver}.png', bbox_inches = 'tight', pad_inches = 0.1, dpi = 200)

#%%
u = np.zeros((nx+1,ny+1))
v = np.zeros((nx+1,ny+1))
u[:,ny] = 1.0

i = np.arange(1,nx)
j = np.arange(1,ny)
ii,jj = np.meshgrid(i,j, indexing='ij')

u[ii,jj] = (s[ii,jj+1] - s[ii,jj-1])/(2.0*dy) 
v[ii,jj] = -(s[ii+1,jj] - s[ii-1,jj])/(2.0*dx)

#%%
# expt_data = np.loadtxt(f'plot_u_y_Ghia{int(re)}.csv', delimiter=',', skiprows=1)

# uc = u[int(nx/2),:]
# plt.plot(uc,y,'r-',lw=2,fillstyle='none',mew=1,ms=8)
# plt.plot(expt_data[:,1],expt_data[:,0],'go',fillstyle='none',mew=1,ms=8)
# plt.show()            

filename = os.path.join(directory, f'solution_{int(re)}_{nx}_{ny}.npz')
np.savez(filename,
         X = X, Y = Y,
         w = w, s = s, 
         kc = kc, 
         rw = rw, rs = rs)  

