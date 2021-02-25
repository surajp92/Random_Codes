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
def grad_spectral(nx,ny,u):
    
    '''
    compute the gradient of u using spectral differentiation
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    u : solution field 
    
    Output
    ------
    ux : du/dx (size = [nx+1,ny+1])
    uy : du/dy (size = [nx+1,ny+1])
    '''
    
    ux = np.empty((nx+1,ny+1))
    uy = np.empty((nx+1,ny+1))
    
    uf = np.fft.fft2(u[0:nx,0:ny])

    kx = np.fft.fftfreq(nx,1/nx)
    ky = np.fft.fftfreq(ny,1/ny)
    
    kx = kx.reshape(nx,1)
    ky = ky.reshape(1,ny)
    
    uxf = 1.0j*kx*uf
    uyf = 1.0j*ky*uf 
    
    ux[0:nx,0:ny] = np.real(np.fft.ifft2(uxf))
    uy[0:nx,0:ny] = np.real(np.fft.ifft2(uyf))
    
    # periodic bc
    ux[:,ny] = ux[:,0]
    ux[nx,:] = ux[0,:]
    ux[nx,ny] = ux[0,0]
    
    # periodic bc
    uy[:,ny] = uy[:,0]
    uy[nx,:] = uy[0,:]
    uy[nx,ny] = uy[0,0]
    
    return ux,uy

#%%
# set periodic boundary condition for ghost nodes. 
# Index (0,1) and (n+3,n+4) are the ghost boundary locations
@jit
def bc(nx,ny,w,s):
    
    w[0,:] = -(2.0/dx**2)*(s[1,:])
    w[nx,:] = -(2.0/dx**2)*(s[nx-1,:])
    
    w[:,0] = -(2.0/dy**2)*(s[:,1]) # bottom wall
    w[:,ny] = -(2.0/dy**2)*(s[:,ny-1]) - 2.0/dy # top wall
    
    
    
    return w

#%% 
# compute rhs using arakawa scheme
# computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
# no ghost points
def rhs_arakawa(nx,ny,dx,dy,re,w,s):
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    dd = 1.0/(2.0*dx)
    
    f = np.zeros((nx+1,ny+1))
    
    ii = np.arange(1,nx)
    jj = np.arange(1,ny)
    i,j = np.meshgrid(ii,jj, indexing='ij')

    #Arakawa    
#    j1 = gg*( (w[3:nx+4,2:ny+3]-w[1:nx+2,2:ny+3])*(s[2:nx+3,3:ny+4]-s[2:nx+3,1:ny+2]) \
#             -(w[2:nx+3,3:ny+4]-w[2:nx+3,1:ny+2])*(s[3:nx+4,2:ny+3]-s[1:nx+2,2:ny+3]))
#
#    j2 = gg*( w[3:nx+4,2:ny+3]*(s[3:nx+4,3:ny+4]-s[3:nx+4,1:ny+2]) \
#            - w[1:nx+2,2:ny+3]*(s[1:nx+2,3:ny+4]-s[1:nx+2,1:ny+2]) \
#            - w[2:nx+3,3:ny+4]*(s[3:nx+4,3:ny+4]-s[1:nx+2,3:ny+4]) \
#            + w[2:nx+3,1:ny+2]*(s[3:nx+4,1:ny+2]-s[1:nx+2,1:ny+2]))
#    
#    j3 = gg*( w[3:nx+4,3:ny+4]*(s[2:nx+3,3:ny+4]-s[3:nx+4,2:ny+3]) \
#            - w[1:nx+2,1:ny+2]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
#            - w[1:nx+2,3:ny+4]*(s[2:nx+3,3:ny+4]-s[1:nx+2,2:ny+3]) \
#            + w[3:nx+4,1:ny+2]*(s[3:nx+4,2:ny+3]-s[2:nx+3,1:ny+2]) )
    
    j1 = gg*((w[i+1,j]-w[i-1,j])*(s[i,j+1]-s[i,j-1]) - \
                 (w[i,j+1]-w[i,j-1])*(s[i+1,j]-s[i-1,j]))

    j2 = gg*(w[i+1,j]*(s[i+1,j+1]-s[i+1,j-1]) - \
             w[i-1,j]*(s[i-1,j+1]-s[i-1,j-1]) - \
             w[i,j+1]*(s[i+1,j+1]-s[i-1,j+1]) + \
             w[i,j-1]*(s[i+1,j-1]-s[i-1,j-1]))

    j3 = gg*(w[i+1,j+1]*(s[i,j+1]-s[i+1,j]) - \
             w[i-1,j-1]*(s[i-1,j]-s[i,j-1]) - \
        	 w[i-1,j+1]*(s[i,j+1]-s[i-1,j]) + \
        	 w[i+1,j-1]*(s[i+1,j]-s[i,j-1]))


    jac = (j1+j2+j3)*hh
    
#    lap = aa*(w[3:nx+4,2:ny+3]-2.0*w[2:nx+3,2:ny+3]+w[1:nx+2,2:ny+3]) \
#        + bb*(w[2:nx+3,3:ny+4]-2.0*w[2:nx+3,2:ny+3]+w[2:nx+3,1:ny+2])
    
    lap = aa*(w[i+1,j]-2.0*w[i,j]+w[i-1,j]) + bb*(w[i,j+1]-2.0*w[i,j]+w[i,j-1])
                                
#    f[3:nx+2,3:ny+2] = -jac[1:nx,1:ny] + lap[1:nx,1:ny]/re 
    
    f[i,j] = -jac + lap/re 
        
    return f
   
#%% 
# read input file
with open(r'ldc_parameters.yaml') as file:
#    input_data = yaml.load(file, Loader=yaml.FullLoader)
    input_data = yaml.load(file)
    
file.close()

nx = input_data['nx']
ny = input_data['ny']
re = input_data['re']
nt = input_data['nt']
lx = input_data['lx']
ly = input_data['ly']
isolver = input_data['isolver']
eps = float(input_data['eps'])

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
t = np.zeros((nx+1,ny+1))
r = np.zeros((nx+1,ny+1))

#%% 
w0 = np.copy(w)
s0 = np.copy(s)
kc = []
wr = []
sr = []

#%%
def rhs(nx,ny,dx,dy,re,w,s,isolver):
    if isolver == 1:
        return rhs_arakawa(nx,ny,dx,dy,re,w,s)


# time integration using third-order Runge Kutta method
aa = 1.0/3.0
bb = 2.0/3.0
clock_time_init = tm.time()

i = np.arange(1,nx)
j = np.arange(1,ny)
ii,jj = np.meshgrid(i,j, indexing='ij')

for k in range(1,nt+1):
    w0 = np.copy(w)
    s0 = np.copy(s)

    w = bc(nx,ny,w,s)
    
    #stage-1
    r[:,:] = rhs(nx,ny,dx,dy,re,w[:,:],s[:,:],isolver)
    t[ii,jj] = w[ii,jj] + dt*r[ii,jj]
    t = bc(nx,ny,t,s)    
    s = fst(nx,ny,dx,dy,-w)

    #stage-2
    r[:,:] = rhs(nx,ny,dx,dy,re,t[:,:],s[:,:],isolver)
    t[ii,jj] = 0.75*w[ii,jj] + 0.25*t[ii,jj] + 0.25*dt*r[ii,jj]
    t[:,:] = bc(nx,ny,t,s)    
    s[:,:] = fst(nx,ny,dx,dy,-t)

    #stage-3
    r[:,:] = rhs(nx,ny,dx,dy,re,t[:,:],s[:,:],isolver)
    w[ii,jj] = aa*w[ii,jj] + bb*t[ii,jj] + bb*dt*r[ii,jj]
    w[:,:] = bc(nx,ny,w[:,:], s)
    s[:,:] = fst(nx,ny,dx,dy,-w)
    
    kc.append(k)
    wr.append(np.linalg.norm(w - w0)/np.sqrt(np.size(w)))
    sr.append(np.linalg.norm(s - s0)/np.sqrt(np.size(s)))
    
    print('%0.3i %0.3e %0.3e' % (kc[k-1], wr[k-1], sr[k-1]))
    
    if wr[k-1] <= eps and sr[k-1] <= eps:
        break

    
total_clock_time = tm.time() - clock_time_init
print('Total clock time=', total_clock_time)
np.save('cpu_time.npy',total_clock_time)


#%%
fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.semilogy(kc, wr)
ax.semilogy(kc, sr)
plt.show()

#%%
fig, axs = plt.subplots(1,2,figsize=(14,5))
cs = axs[0].contourf(X,Y,w,60,cmap='jet')
#cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
fig.colorbar(cs, ax=axs[0], orientation='vertical')

cs = axs[1].contourf(X,Y,s,60,cmap='jet')
#cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
fig.colorbar(cs, ax=axs[1], orientation='vertical')

plt.show()
fig.tight_layout()
fig.savefig('ldc_ws.png', bbox_inches = 'tight', pad_inches = 0, dpi = 300)


#%%
#sx, sy = grad_spectral(nx,ny,s)
#u = sy
#v = -sx

u = np.zeros((nx+1,ny+1))
v = np.zeros((nx+1,ny+1))
u[:,ny] = 1.0

i = np.arange(1,nx)
j = np.arange(1,ny)
ii,jj = np.meshgrid(i,j, indexing='ij')

u[ii,jj] = (s[ii,jj+1] - s[ii,jj-1])/(2.0*dy) 
v[ii,jj] = -(s[ii+1,jj] - s[ii-1,jj])/(2.0*dx)

#%%
expt_data = np.loadtxt('plot_u_y_Ghia100.csv', delimiter=',', skiprows=1)

uc = u[int(nx/2),:]
plt.plot(uc,y,'r-',lw=2,fillstyle='none',mew=1,ms=8)
plt.plot(expt_data[:,1],expt_data[:,0],'go',fillstyle='none',mew=1,ms=8)
plt.show()            


