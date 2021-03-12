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

# import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

# fast poisson solver using second-order central difference scheme
def fps(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    aa = -2.0/(dx*dx) - 2.0/(dy*dy)
    bb = 2.0/(dx*dx)
    cc = 2.0/(dy*dy)
    
    kx = (2.0*np.pi/nx)*np.float64(np.arange(0, nx))
    ky = (2.0*np.pi/ny)*np.float64(np.arange(0, ny))
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(np.cos(kx), np.cos(ky), indexing='ij')
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
    
    data[:,:] = np.vectorize(complex)(f[0:nx,0:ny],0.0)

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
    u = np.zeros((nx+1,ny+1)) 
    u[0:nx,0:ny] = ut
    u[:,ny] = u[:,0]
    u[nx,:] = u[0,:]
    u[nx,ny] = u[0,0]
    return u

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
# fast poisson solver using second-order central difference scheme
def hybrid(nx, ny, dx, dy, f):
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
    
    # kx = (2.0*np.pi/nx)*np.arange(0, nx)        
    # kx[0] = epsilon
    
    kx = np.fft.fftfreq(nx, d = dx)*(2.0*np.pi)
    kx[0] = epsilon
    
    cos_kx = np.cos(2.0*np.pi*kx/nx) 
    
    data = np.empty((nx,ny+1), dtype='complex128')
    data1 = np.empty((nx,ny+1), dtype='complex128')
    
    data[:,:] = np.vectorize(complex)(f[0:nx,0:ny+1],0.0)

    a = pyfftw.empty_aligned((nx,ny+1),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny+1),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,), direction = 'FFTW_BACKWARD')
    
    data_f = fft_object(data)
    #e = pyfftw.interfaces.scipy_fftpack.fft2(data)
    
    # data_f[0,0] = 0.0
    j = 0
    data1[:,j] = data_f[:,j]

    j = ny
    data1[:,j] = data_f[:,j]
    
    alpha_k = c4 + 2.0*d4*cos_kx
    beta_k = a4 + 2.0*b4*cos_kx
        
    for j in range(1,ny):
        # print(j)
        
        rr = e4*(data_f[:,j-1] + (8.0 + 2.0*cos_kx)*data_f[:,j] + data_f[:,j+1]) 
        
        alpha_k = np.reshape(alpha_k,[-1,1])
        beta_k = np.reshape(beta_k,[-1,1])
        rr = np.reshape(rr,[-1,1])
        
        temp = tdma(alpha_k,beta_k,alpha_k,rr,0,nx-1)
        data1[:,j] = temp.flatten()
    # data1[:,:] = data_f[:,:]/(aa + bb*kx[:,:] + cc*ky[:,:])

    ut = np.real(fft_object_inv(data1))
    
    #periodicity
    u = np.zeros((nx+1,ny+1)) 
    u[0:nx,0:ny+1] = ut
    # u[:,ny] = u[:,0]
    u[nx,:] = u[0,:]
    u[nx,ny] = u[0,0]
    return u


#%%
def fps_s(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    aa = -2.0/(dx*dx) - 2.0/(dy*dy)
    bb = 2.0/(dx*dx)
    cc = 2.0/(dy*dy)
    hx = 2.0*np.pi/np.float64(nx)
    hy = 2.0*np.pi/np.float64(ny)
    
    kx = np.fft.fftfreq(nx, d=dx)*(2.0*np.pi)
    ky = np.fft.fftfreq(ny, d=dx)*(2.0*np.pi)
    
    # kx = np.zeros(nx)
    # ky = np.zeros(ny)

    # for i in range(int(nx/2)):
    #     kx[i] = 2*np.pi*np.float64(i)
    #     kx[i+int(nx/2)] = 2*np.pi*np.float64(i-int(nx/2))
        
    # for i in range(ny):
    #     ky[i] = kx[i]
    
    kx[0] = epsilon
    ky[0] = epsilon
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
    
    # create data using the dource term as the real part and 0.0 as the imaginary part
    # for i in range(nx):
    #     for j in range(ny):
    #         data[i,j] = complex(f[i,j],0.0)
    
    data[:,:] = np.vectorize(complex)(f[0:nx,0:ny],0.0)
    
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    # compute the fourier transform
    e = fft_object(data)
    #e = pyfftw.interfaces.scipy_fftpack.fft2(data)
    
    e[0,0] = 0.0
    
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    data1 = e/(-kx**2 - ky**2)
    
    # # the donominator is based on the scheme used for discrtetizing the Poisson equation
    # for i in range(nx):
    #     for j in range(ny):
    #         data1[i,j] = e[i,j]/(-kx[i]**2 - ky[j]**2)
    
    # compute the inverse fourier transform
    ut = np.real(fft_object_inv(data1))
    
    #periodicity
    u = np.zeros((nx+1,ny+1)) 
    u[0:nx,0:ny] = ut
    u[:,ny] = u[:,0]
    u[nx,:] = u[0,:]
    u[nx,ny] = u[0,0]
    
    return u

#%%
def fst(nx,ny,dx,dy,f):
    data = f[1:-1,1:-1]
        
#    e = dst(data, type=2)
    data = dst(data, axis = 1, type = 1)
    data = dst(data, axis = 0, type = 1)
    
    m = np.linspace(1,nx-1,nx-1).reshape([-1,1])
    n = np.linspace(1,ny-1,ny-1).reshape([1,-1])
        
    data1 = np.zeros((nx-1,ny-1))

    alpha = (2.0/(dx*dx))*(np.cos(np.pi*m/nx) - 1.0) + (2.0/(dy*dy))*(np.cos(np.pi*n/ny) - 1.0)           
    data1 = data/alpha
    
    data1 = idst(data1, axis = 1, type = 1)
    data1 = idst(data1, axis = 0, type = 1)
    
    u = data1/((2.0*nx)*(2.0*ny))
    
    ue = np.zeros((nx+1,ny+1))
    ue[1:-1,1:-1] = u
    
    return ue

def fst4(nx,ny,dx,dy,f):
    
    beta = dx/dy
    a = -10.0*(1.0 + beta**2)
    b = 5.0 - beta**2
    c = 5.0*beta**2 -1.0
    d = 0.5*(1.0 + beta**2)
    
    data = f[1:-1,1:-1]

    data = dst(data, axis = 1, type = 1)
    data = dst(data, axis = 0, type = 1)
    
    m = np.linspace(1,nx-1,nx-1).reshape([-1,1])
    n = np.linspace(1,ny-1,ny-1).reshape([1,-1])
    
    data1 = np.zeros((nx-1,ny-1))
    
    alpha = a + 2.0*b*np.cos(np.pi*m/nx) + 2.0*c*np.cos(np.pi*n/ny) + \
            4.0*d*np.cos(np.pi*m/nx)*np.cos(np.pi*n/ny)
            
    gamma = 8.0 + 2.0*np.cos(np.pi*m/nx) + 2.0*np.cos(np.pi*n/ny)
               
    data1 = data*(dx**2)*0.5*gamma/alpha
    
    data1 = idst(data1, axis = 1, type = 1)
    data1 = idst(data1, axis = 0, type = 1)
    
    data1 = data1/((2.0*nx)*(2.0*ny))
    
    ue = np.zeros((nx+1,ny+1))
    ue[1:-1,1:-1] = data1
    
    return ue

#%% 
if __name__ == "__main__":
    for i in range(3):
        nx = 32*(2**i)
        ny = 32*(2**i)
        
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
                                 
        ue = np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
              c1*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)
        
        f = 4.0*c2*np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
            c2*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)
                 
        un = hybrid(nx,ny,dx,dy,f)    
        
        errL2 = np.linalg.norm(un - ue)/np.sqrt(np.size(ue))
        
        print('#----------------------------------------#')
        print('n = %d' % (nx))
        print('L2 error:  %5.3e' % errL2)
        if i>=1:
            rateL2 = np.log(errL2_0/errL2)/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
        
        errL2_0 = errL2
        
    fig, axs = plt.subplots(1,2,figsize=(14,5))
    cs = axs[0].contourf(xm, ym, ue, 60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[0], orientation='vertical')
    
    cs = axs[1].contourf(xm, ym, 10*un,60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[1], orientation='vertical')
    
    plt.show()
    fig.tight_layout()
        
