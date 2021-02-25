#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:29:12 2021

@author: suraj
"""


import numpy as np
from scipy.fftpack import dst, idst
#from numpy import empty,arange,exp,real,imag,pi
#from numpy.fft import rfft,irfft
import matplotlib.pyplot as plt 
import time

nx = 64
ny = 64

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
c2 = -8.0*np.pi**2

ue = np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
     c1*np.sin(km*2.0*np.pi*xm)*np.sin(km*2.0*np.pi*ym)

f = c2*np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
    c2*np.sin(km*2.0*np.pi*xm)*np.sin(km*2.0*np.pi*ym)

#%%
def compute_residual(nx, ny, dx, dy, f, u_n):
    r = np.zeros((nx+1, ny+1))
    ii = np.arange(1,nx)
    jj = np.arange(1,ny)
    i,j = np.meshgrid(ii,jj,indexing='ij')
    
    d2udx2 = (u_n[i+1,j] - 2*u_n[i,j] + u_n[i-1,j])/(dx**2)
    d2udy2 = (u_n[i,j+1] - 2*u_n[i,j] + u_n[i,j-1])/(dy**2)
    r[i,j] = f[i,j]  - d2udx2 - d2udy2
    
    return r
    
def restriction(nxf, nyf, nxc, nyc, r):
    ec = np.zeros((nxc+1, nyc+1))
    ii = np.arange(1,nxc)
    jj = np.arange(1,nyc)
    i,j = np.meshgrid(ii,jj,indexing='ij')
    
    # grid index for fine grid for the same coarse point
    center = 4.0*r[2*i, 2*j]
    
    # E, W, N, S with respect to coarse grid point in fine grid
    grid = 2.0*(r[2*i, 2*j+1] + r[2*i, 2*j-1] +
                r[2*i+1, 2*j] + r[2*i-1, 2*j])
    
    # NE, NW, SE, SW with respect to coarse grid point in fine grid
    corner = 1.0*(r[2*i+1, 2*j+1] + r[2*i+1, 2*j-1] +
                  r[2*i-1, 2*j+1] + r[2*i-1, 2*j-1])
    
    # restriction using trapezoidal rule
    ec[i,j] = (center + grid + corner)/16.0
    
    i = np.arange(0,nxc+1)
    ec[i,0] = r[2*i, 0]
    ec[i,nyc] = r[2*j, nyf]
    
    j = np.arange(0,nyc+1)
    ec[0,j] = r[0, 2*j]
    ec[nxc,j] = r[nxf, 2*j]
    
    return ec

def proplongation(nxc, nyc, nxf, nyf, unc):
    ef = np.zeros((nxf+1, nyf+1))
    ii = np.arange(0,nxc)
    jj = np.arange(0,nyc)
    i,j = np.meshgrid(ii,jj,indexing='ij')
    
    ef[2*i, 2*j] = unc[i,j]
    # east neighnour on fine grid corresponding to coarse grid point
    ef[2*i, 2*j+1] = 0.5*(unc[i,j] + unc[i,j+1])
    # north neighbout on fine grid corresponding to coarse grid point
    ef[2*i+1, 2*j] = 0.5*(unc[i,j] + unc[i+1,j])
    # NE neighbour on fine grid corresponding to coarse grid point
    ef[2*i+1, 2*j+1] = 0.25*(unc[i,j] + unc[i,j+1] + unc[i+1,j] + unc[i+1,j+1])
    
    i = np.arange(0,nxc+1)
    ef[2*i,nyf] = unc[i,nyc]
    
    j = np.arange(0,nyc+1)
    ef[nxf,2*j] = unc[nxc,j]
    
    return ef

def gaiss_seidel_mg(nx, ny, dx, dy, f, un, V):
    rt = np.zeros((nx+1,ny+1))
    den = -2.0/dx**2 - 2.0/dy**2
    omega = 1.0
    
    ii = np.arange(1,nx)
    jj = np.arange(1,ny)
    i,j = np.meshgrid(ii,jj,indexing='ij')
    
    for k in range(V):
        rt[i,j] = f[i,j] - \
                  (un[i+1,j] - 2.0*un[i,j] + un[i-1,j])/dx**2 - \
                  (un[i,j+1] - 2.0*un[i,j] + un[i,j-1])/dy**2
                  
        un[i,j] = un[i,j] + omega*rt[i,j]/den
        
#%%    
u_n = np.zeros((nx+1,ny+1))    
u_mg = []
f_mg = []    

u_mg.append(u_n)
f_mg.append(f)