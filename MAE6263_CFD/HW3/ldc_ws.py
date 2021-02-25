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
    fd = f[2:nx+3,2:ny+3]
    data = fd[1:-1,1:-1]
        
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
    
    ue = np.zeros((nx+5,ny+5))
    ue[3:nx+2,3:ny+2] = u
    
    return ue

#%%
# fast poisson solver using second-order central difference scheme
def fps(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    aa = -2.0/(dx*dx) - 2.0/(dy*dy)
    bb = 2.0/(dx*dx)
    cc = 2.0/(dy*dy)
    hx = 2.0*np.pi/np.float64(nx)
    hy = 2.0*np.pi/np.float64(ny)
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[:] = hx*np.float64(np.arange(0, nx))

    ky[:] = hy*np.float64(np.arange(0, ny))
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(np.cos(kx), np.cos(ky), indexing='ij')
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
    
    data[:,:] = np.vectorize(complex)(f[2:nx+2,2:ny+2],0.0)

    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    e = fft_object(data)
    #e = pyfftw.interfaces.scipy_fftpack.fft2(data)
    
    e[0,0] = 0.0
    
    data1[:,:] = e[:,:]/(aa + bb*kx[:,:] + cc*ky[:,:])

    ut = np.real(fft_object_inv(data1))
    
    #periodicity
    u = np.empty((nx+5,ny+5)) 
    u[2:nx+2,2:ny+2] = ut
    u[:,ny+2] = u[:,2]
    u[nx+2,:] = u[2,:]
    u[nx+2,ny+2] = u[2,2]
    
    return u

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
    w[:,0] = 0.0
    w[:,1] = 0.0
    w[:,ny+3] = 0.0
    w[:,ny+4] = 0.0
    w[:,2] = -(2.0/dy**2)*(s[:,3]) # bottom wall
    w[:,ny+2] = -(2.0/dy**2)*(s[:,ny+1]) - 2.0/dy # top wall

    w[0,:] = 0.0
    w[1,:] = 0.0
    w[nx+3,:] = 0.0
    w[nx+4,:] = 0.0
    
    w[2,:] = -(2.0/dx**2)*(s[3,:])
    w[nx+2,:] = -(2.0/dx**2)*(s[nx+1,:])
    
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
    
    f = np.zeros((nx+5,ny+5))
    
    #Arakawa    
    j1 = gg*( (w[3:nx+4,2:ny+3]-w[1:nx+2,2:ny+3])*(s[2:nx+3,3:ny+4]-s[2:nx+3,1:ny+2]) \
             -(w[2:nx+3,3:ny+4]-w[2:nx+3,1:ny+2])*(s[3:nx+4,2:ny+3]-s[1:nx+2,2:ny+3]))

    j2 = gg*( w[3:nx+4,2:ny+3]*(s[3:nx+4,3:ny+4]-s[3:nx+4,1:ny+2]) \
            - w[1:nx+2,2:ny+3]*(s[1:nx+2,3:ny+4]-s[1:nx+2,1:ny+2]) \
            - w[2:nx+3,3:ny+4]*(s[3:nx+4,3:ny+4]-s[1:nx+2,3:ny+4]) \
            + w[2:nx+3,1:ny+2]*(s[3:nx+4,1:ny+2]-s[1:nx+2,1:ny+2]))
    
    j3 = gg*( w[3:nx+4,3:ny+4]*(s[2:nx+3,3:ny+4]-s[3:nx+4,2:ny+3]) \
            - w[1:nx+2,1:ny+2]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
            - w[1:nx+2,3:ny+4]*(s[2:nx+3,3:ny+4]-s[1:nx+2,2:ny+3]) \
            + w[3:nx+4,1:ny+2]*(s[3:nx+4,2:ny+3]-s[2:nx+3,1:ny+2]) )

    jac = (j1+j2+j3)*hh
    
    lap = aa*(w[3:nx+4,2:ny+3]-2.0*w[2:nx+3,2:ny+3]+w[1:nx+2,2:ny+3]) \
        + bb*(w[2:nx+3,3:ny+4]-2.0*w[2:nx+3,2:ny+3]+w[2:nx+3,1:ny+2])
        
    f[3:nx+2,3:ny+2] = -jac[1:nx,1:ny] + lap[1:nx,1:ny]/re 
    
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
x, y = np.meshgrid(x, y, indexing='ij')

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

# boundary condition for vorticity

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
    time = time + dt
    
    w = bc(nx,ny,w,s)
    
    #stage-1
    r[:,:] = rhs(nx,ny,dx,dy,re,w[:,:],s[:,:],isolver)
    t[3:nx+2,3:ny+2] = w[3:nx+2,3:ny+2] + dt*r[3:nx+2,3:ny+2]
    t = bc(nx,ny,t,s)    
    s = fst(nx,ny,dx,dy,-w)

    #stage-2
    r[:,:] = rhs(nx,ny,dx,dy,re,t[:,:],s[:,:],isolver)
    t[2:nx+3,2:ny+3] = 0.75*w[2:nx+3,2:ny+3] + 0.25*t[2:nx+3,2:ny+3] + 0.25*dt*r[2:nx+3,2:ny+3]
    t[:,:] = bc(nx,ny,t,s)    
    s[:,:] = fst(nx, ny, dx, dy, -t[:,:])

    #stage-3
    r[:,:] = rhs(nx,ny,dx,dy,re,t[:,:],s[:,:],isolver)
    w[2:nx+3,2:ny+3] = aa*w[2:nx+3,2:ny+3] + bb*t[2:nx+3,2:ny+3] + bb*dt*r[2:nx+3,2:ny+3]
    w[:,:] = bc(nx,ny,w[:,:], s)
    s[:,:] = fst(nx, ny, dx, dy, -w[:,:])

        
total_clock_time = tm.time() - clock_time_init
print('Total clock time=', total_clock_time)
np.save('cpu_time.npy',total_clock_time)


#%%
fig, axs = plt.subplots(1,2,figsize=(12,5))

cs = axs[0].contourf(x,y,w[2:nx+3,2:ny+3],60,cmap='jet')
#cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
fig.colorbar(cs, ax=axs[0], orientation='vertical')

cs = axs[1].contourf(x,y,s[2:nx+3,2:ny+3],60,cmap='jet')
#cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
fig.colorbar(cs, ax=axs[1], orientation='vertical')

plt.show()
fig.tight_layout()
fig.savefig('ldc_ws.png', bbox_inches = 'tight', pad_inches = 0, dpi = 300)



