#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 14:50:14 2019

@author: Suraj Pawar
fast Poisson solver in 2D periodic domain (spectral method)
"""

import numpy as np
import pyfftw
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
from scipy.fftpack import fft2, ifft2

font = {'family' : 'Times New Roman',
        'size'   : 14}	
plt.rc('font', **font)

#%%
def fps(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    aa = -2.0/(dx*dx) - 2.0/(dy*dy)
    bb = 2.0/(dx*dx)
    cc = 2.0/(dy*dy)
    hx = 2.0*np.pi/np.float64(nx)
    hy = 2.0*np.pi/np.float64(ny)
    
    kx = np.empty(nx)
    ky = np.empty(ny)

    for i in range(int(nx/2)):
        kx[i] = 2*np.pi/(np.float64(nx)*dx)*np.float64(i)
        kx[i+int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(i-int(nx/2))
        
    for i in range(ny):
        ky[i] = kx[i]
    
    kx[0] = epsilon
    ky[0] = epsilon
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
    
    # create data using the dource term as the real part and 0.0 as the imaginary part
    for i in range(nx):
        for j in range(ny):
            data[i,j] = complex(f[i,j],0.0)
       
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    # compute the fourier transform
    e = fft_object(data)
    #e = pyfftw.interfaces.scipy_fftpack.fft2(data)
    
    e[0,0] = 0.0
    
    # the donominator is based on the scheme used for discrtetizing the Poisson equation
    for i in range(nx):
        for j in range(ny):
            data1[i,j] = e[i,j]/(-kx[i]**2 - ky[j]**2)
    
    # compute the inverse fourier transform
    u = np.real(fft_object_inv(data1))
    return u
    
    
    
#%%
# 
    
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

#ue = np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
#     c1*np.sin(km*2.0*np.pi*xm)*np.sin(km*2.0*np.pi*ym)

ue = np.sin(2.0*np.pi*xm)*np.cos(2.0*np.pi*ym) + \
     c1*np.sin(km*2.0*np.pi*xm)*np.cos(km*2.0*np.pi*ym)

#f = c2*np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
#    c2*np.sin(km*2.0*np.pi*xm)*np.sin(km*2.0*np.pi*ym)

f = c2*np.sin(2.0*np.pi*xm)*np.cos(2.0*np.pi*ym) + \
    c2*np.sin(km*2.0*np.pi*xm)*np.cos(km*2.0*np.pi*ym)

#nx = 8
#ny = 8
#x = np.linspace(0.0,2.0*np.pi,nx+1)
#y = np.linspace(0.0,2.0*np.pi,ny+1)
#dx = 2.0*np.pi/(nx)
#dy = 2.0*np.pi/(ny)
#
#ue = np.empty((nx+1,ny+1))
#f = np.empty((nx+1,ny+1))
#
## test case Ue = sin(3x) + cos(2y); f = -9sin(3x) - 4cos(2y)
##for i in range(nx+1):
##    ue[i,:] = np.sin(3.0*x[i]) + np.cos(2.0*y)
##    f[i,:] = -9.0*np.sin(3.0*x[i]) -4.0*np.cos(2.0*y)
#
#for i in range(nx+1):
#    ue[i,:] = np.sin(3.0*x[i]) + np.cos(2.0*y)
#    f[i,:] = -9.0*np.sin(3.0*x[i]) -4.0*np.cos(2.0*y)
    
u = np.zeros((nx+1, ny+1))
# fps subroutine skips the last row and last column since the boundary condition is periodic
u[:nx, :ny] = fps(nx, ny, dx, dy, f)

# extend boundary of the domain
u[:,ny] = u[:,0]
u[nx,:] = u[0,:]

#%%
# contour plot for initial and final vorticity
fig, axs = plt.subplots(1,2,sharey=True,figsize=(9,5))

cs = axs[0].contourf(x,y,ue.T, 120, cmap = 'jet', interpolation='bilinear')
axs[0].text(0.4, -0.1, 'Exact', transform=axs[0].transAxes, fontsize=16, va='top')
cs = axs[1].contourf(x,y,u.T, 120, cmap = 'jet', interpolation='bilinear')
axs[1].text(0.4, -0.1, 'Numerical', transform=axs[1].transAxes, fontsize=16,  va='top')

fig.tight_layout() 

fig.subplots_adjust(bottom=0.15)

cbar_ax = fig.add_axes([0.22, -0.05, 0.6, 0.04])
fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
plt.show()

fig.savefig("s_fps.eps", bbox_inches = 'tight')


print(np.linalg.norm(u-ue)/(nx*ny))
