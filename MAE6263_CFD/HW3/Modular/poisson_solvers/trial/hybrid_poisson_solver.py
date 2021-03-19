#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 18:23:30 2021

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

#%%
def tdma(a,b,c,r,s,e):
    
    a_ = np.copy(a)
    b_ = np.copy(b)
    c_ = np.copy(c)
    r_ = np.copy(r)
    
    un = np.zeros((np.shape(r)[0],np.shape(r)[1]), dtype='complex128')
    
    for i in range(s+1,e+1):
        b_[i,:] = b_[i,:] - a_[i,:]*(c_[i-1,:]/b_[i-1,:])
        r_[i,:] = r_[i,:] - a_[i,:]*(r_[i-1,:]/b_[i-1,:])
        
    un[e,:] = r_[e,:]/b_[e,:]
    
    for i in range(e-1,s-1,-1):
        un[i,:] = (r_[i,:] - c_[i,:]*un[i+1,:])/b_[i,:]
    
    del a_, b_, c_, r_
    
    return un

#%%
nx = 32
ny = 32

x_l = 0.0
x_r = 2.0
y_b = 0.0
y_t = 2.0

dx = (x_r-x_l)/nx
dy = (y_t-y_b)/ny

x = np.linspace(x_l, x_r, nx+1)
y = np.linspace(y_b, y_t, ny+1)

xm, ym = np.meshgrid(x,y, indexing='ij')

km = 16.0
c1 = (1.0/km)**2
c2 = -2.0*np.pi**2
                         
ue = np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
      c1*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)

f = 4.0*c2*np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
    c2*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)

#%%            
epsilon = 1.0e-6
aa = -2.0/(dx*dx) - 2.0/(dy*dy)
bb = 2.0/(dx*dx)
cc = 2.0/(dy*dy)

beta = dx/dy
a4 = -10.0*(1.0 + beta**2)
b4 = 5.0 - beta**2
c4 = 5.0*beta**2 -1.0
d4 = 0.5*(1.0 + beta**2)
e4 = 0.5*(dx**2)

# wave_number = np.arange(-int(nx/2), int(nx/2)) #*(2.0*np.pi)        

Lx = nx*dx

# # wave_number_coord = np.fft.fftfreq(nx, d = 1/nx)

# define discrete wavenumbers
wave_number = np.fft.fftfreq(nx, d = dx)*(2.0*np.pi)

wave_number = np.arange(0,nx)
 
kx = np.copy(wave_number)
kx[0] = epsilon

# cos_kx = np.cos(kx)
# cos_kx = np.cos(kx*Lx/nx)
cos_kx = np.cos(2.0*np.pi*kx/nx) 


data = np.empty((nx,ny+1), dtype='complex128')
data1 = np.empty((nx,ny+1), dtype='complex128')

data[:,:] = np.vectorize(complex)(f[0:nx,0:ny+1],0.0)

a = pyfftw.empty_aligned((nx,ny+1),dtype= 'complex128')
b = pyfftw.empty_aligned((nx,ny+1),dtype= 'complex128')

fft_object = pyfftw.FFTW(a, b, axes = (0,), direction = 'FFTW_FORWARD')
fft_object_inv = pyfftw.FFTW(a, b,axes = (0,), direction = 'FFTW_BACKWARD')

data_f = np.fft.fft(data, axis=0)
# data_f = fft_object(data)
# data_f = np.abs(data_f)
#e = pyfftw.interfaces.scipy_fftpack.fft2(data)

# data_f[0,0] = 0.0
j = 0
data1[:,j] = data_f[:,j]

j = ny
data1[:,j] = data_f[:,j]

alpha_k = c4 + 2.0*d4*cos_kx
beta_k = a4 + 2.0*b4*cos_kx

alpha_k = np.reshape(alpha_k,[-1,1])
beta_k = np.reshape(beta_k,[-1,1])

A = np.zeros((nx,ny))
for i in range(nx):
    A[i,i] = beta_k[i,0]
    if i > 0:
        A[i,i-1] = alpha_k[i,0]
    if i < nx-1:
        A[i,i+1] = alpha_k[i,0]

AI = np.linalg.inv(A)        

for j in range(1,ny):
    # print(j)
    
    rr = e4*(data_f[:,j-1] + (8.0 + 2.0*cos_kx)*data_f[:,j] + data_f[:,j+1]) 
    rr = np.reshape(rr,[-1,1])
    
    # temp = AI @ rr 
    temp = tdma(alpha_k,beta_k,alpha_k,rr,0,nx-1)
    
    data1[:,j] = temp.flatten()
# data1[:,:] = data_f[:,:]/(aa + bb*kx[:,:] + cc*ky[:,:])

ut = np.real(np.fft.ifft(data1, axis=0))
# ut = np.real(fft_object_inv(data1))

#periodicity
u = np.zeros((nx+1,ny+1)) 
u[0:nx,0:ny+1] = ut
# u[:,ny] = u[:,0]
u[nx,:] = u[0,:]
u[nx,ny] = u[0,0]

fig, axs = plt.subplots(1,2,figsize=(14,5))
cs = axs[0].contourf(xm, ym, ue, 60,cmap='jet')
#cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
fig.colorbar(cs, ax=axs[0], orientation='vertical')

cs = axs[1].contourf(xm, ym, u,60,cmap='jet')
#cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
fig.colorbar(cs, ax=axs[1], orientation='vertical')

plt.show()
fig.tight_layout()

print(np.linalg.norm(ue - u))