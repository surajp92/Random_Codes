#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 14:50:14 2019

@author: Suraj Pawar
"""

import numpy as np
import pyfftw
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
from scipy.fftpack import fft2, ifft2

#%%
nx = 512
ny = 512
x = np.linspace(0.0,2.0*np.pi,nx+1)
y = np.linspace(0.0,2.0*np.pi,ny+1)

ue = np.empty((nx+1,ny+1))
f = np.empty((nx+1,ny+1))
for i in range(nx+1):
    ue[i,:] = np.sin(3.0*x[i]) + np.cos(2.0*y)
    f[i,:] = -9.0*np.sin(3.0*x[i]) -4.0*np.cos(2.0*y)
    
#%%
epsilon = 1.0e-6

dx = 2.0*np.pi/(nx)
dy = 2.0*np.pi/(ny)

#hx = 2.0*np.pi/np.float64(nx)
#hy = 2.0*np.pi/np.float64(ny)

kx = np.empty(nx)
ky = np.empty(ny)

for i in range(int(nx/2)):
    kx[i] = 2*np.pi/(np.float64(nx)*dx)*np.float64(i)
    kx[i+int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(i-int(nx/2))
    
for i in range(ny):
    ky[i] = kx[i]

kx[0] = epsilon
ky[0] = epsilon
    

#%%
data = np.empty((nx,ny), dtype='complex128')
data1 = np.empty((nx,ny), dtype='complex128')

for i in range(nx):
    for j in range(ny):
        data[i,j] = complex(f[i,j],0.0)
   

a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')

fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')

e = fft_object(data)
#e = pyfftw.interfaces.scipy_fftpack.fft2(data)

e[0,0] = 0.0

for i in range(nx):
    for j in range(ny):
#        data1[i,j] = e[i,j]/(aa + bb*np.cos(kx[i]) + cc*np.cos(ky[j]))
        data1[i,j] = e[i,j]/(-kx[i]**2 - ky[j]**2)

#%%
u = np.real(fft_object_inv(data1))

#u = np.real(pyfftw.interfaces.scipy_fftpack.ifft2(data1))
        
plt.contourf(u-ue[:nx,:ny])
plt.colorbar()
        
        
        

