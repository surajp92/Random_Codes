#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 12:50:32 2020

@author: suraj
"""

import numpy as np
from scipy.fftpack import dst, idst
#from numpy import empty,arange,exp,real,imag,pi
#from numpy.fft import rfft,irfft
import matplotlib.pyplot as plt 
import time

a = np.random.randn(7,7)

#%%
######################################################################
# 2D DST

def dst2(y):
    M = y.shape[0]
    N = y.shape[1]
    a = np.empty([M,N],float)
    b = np.empty([M,N],float)

    for i in range(M):
        a[i,:] = dst(y[i,:], type=1)
    for j in range(N):
        b[:,j] = dst(a[:,j], type=1)

    return b


######################################################################
# 2D inverse DST

def idst2(b):
    M = b.shape[0]
    N = b.shape[1]
    a = np.empty([M,N],float)
    y = np.empty([M,N],float)

    for i in range(M):
        a[i,:] = idst(b[i,:], type=1)
    for j in range(N):
        y[:,j] = idst(a[:,j], type=1)

    return y

#%%
def fst2(nx,ny,dx,dy,f):
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

def fste(nx,ny,dx,dy,f):
    
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
    
    return u

def fst(nx,ny,dx,dy,f):
    data = f[1:-1,1:-1]
        
#    e = dst(data, type=2)
    e = dst2(data)
    
    m = np.linspace(1,nx-1,nx-1).reshape([-1,1])
    n = np.linspace(1,ny-1,ny-1).reshape([1,-1])
    
    data1 = np.zeros((nx-1,ny-1))
    
    alpha = (2.0/(dx*dx))*(np.cos(np.pi*m/nx) - 1.0) + (2.0/(dy*dy))*(np.cos(np.pi*n/ny) - 1.0)           
    data1 = e/alpha
    
          
#    for i in range(1,nx):
#        for j in range(1,ny):
#            alpha = (2.0/(dx*dx))*(np.cos(np.pi*i/nx) - 1.0) + (2.0/(dy*dy))*(np.cos(np.pi*j/ny) - 1.0)           
#            data1[i-1,j-1] = e[i-1,j-1]/alpha
    
#    u = idst(data1, type=2)/((2.0*nx)*(2.0*ny))
    u = idst2(data1)/((2.0*nx)*(2.0*ny))
    
    return u

#m = np.linspace(1,nx-1,nx-1)
#n = np.linspace(1,ny-1,ny-1)
#mm, nm = np.meshgrid(m,n, indexing='ij')
#    
#alpham = (2.0/(dx*dx))*(cos(pi*mm/nx) - 1.0) + (2.0/(dy*dy))*(cos(pi*nm/ny) - 1.0)

#%%
nx = 256
ny = 512

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

un = np.zeros((nx+1,ny+1))
un1 = np.zeros((nx+1,ny+1))

start = time.time()


un[1:-1,1:-1] = fst(nx,ny,dx,dy,f)
print('T1 = ', time.time() - start)

un1[1:-1,1:-1] = fste(nx,ny,dx,dy,f)
print('T2 = ', time.time() - start)

uee = np.zeros((nx+5,ny+5))
uee[2:nx+3,2:ny+3] = ue[:,:]

fe = np.zeros((nx+5,ny+5))
fe[2:nx+3,2:ny+3] = f[:,:]

start = time.time()
un2 = fst2(nx,ny,dx,dy,fe)
print('T3 = ', time.time() - start)

print(np.linalg.norm(un-ue)/np.sqrt((nx*ny)))
print(np.linalg.norm(un1-ue)/np.sqrt((nx*ny)))
print(np.linalg.norm(un2[2:nx+3,2:ny+3]-ue)/np.sqrt((nx*ny)))

#%%
plt.contourf(xm, ym, ue, 120, cmap='jet')
plt.colorbar()
plt.show()

plt.contourf(xm, ym, un, 120, cmap='jet')
plt.colorbar()
plt.show()

plt.contourf(xm, ym, f, 120, cmap='jet')
plt.colorbar()
plt.show()
