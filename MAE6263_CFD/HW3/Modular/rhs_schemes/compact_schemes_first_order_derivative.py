#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 18:52:37 2021

@author: suraj
"""

import numpy as np
from thomas_algorithms import *
import matplotlib.pyplot as plt

def c4d(f,h,nx,ny,isign):
    
    if isign == 'X':
        u = np.copy(f)
    if isign == 'Y':
        u = np.copy(f.T)
    
    a = np.zeros((nx+1,ny+1))
    b = np.zeros((nx+1,ny+1))
    c = np.zeros((nx+1,ny+1))
    r = np.zeros((nx+1,ny+1))
    
    i = 0
    a[i,:] = 0.0
    b[i,:] = 1.0
    c[i,:] = 2.0
    r[i,:] = (-5.0*u[i,:] + 4.0*u[i+1,:] + u[i+2,:])/(2.0*h)
    
    ii = np.arange(1,nx)
    
    a[ii,:] = 1.0/4.0
    b[ii,:] = 1.0
    c[ii,:] = 1.0/4.0
    r[ii,:] = 3.0*(u[ii+1,:] - u[ii-1,:])/(4.0*h)
    
    i = nx
    a[i,:] = 2.0
    b[i,:] = 1.0
    c[i,:] = 0.0
    r[i,:] = -1.0*(-5.0*u[i,:] + 4.0*u[i-1,:] + u[i-2,:])/(2.0*h)
    
    start = 0
    end = nx
    ud = tdma(a,b,c,r,start,end)
    
    if isign == 'X':
        fd = np.copy(ud)
    if isign == 'Y':
        fd = np.copy(ud.T)
        
    return fd

if __name__ == "__main__":
    xl = -1.0
    xr = 1.0
    
    print('#-----------------Dx-------------------#')
    for i in range(3):
        dx = 0.05/(2**i)
        nx = int((xr - xl)/dx)
          
        ny = nx
        
        x = np.linspace(xl,xr,nx+1)
        y = np.linspace(xl,xr,ny+1)
        X,Y = np.meshgrid(x,y, indexing='ij')
        xx = (np.ones((ny+1,1))*x).T
        
        u = np.zeros((nx+1,ny+1))
        udx = np.zeros((nx+1,ny+1))
        udn = np.zeros((nx+1,ny+1))
        
        u[:,:] = np.sin(np.pi*X) + np.sin(2.0*np.pi*Y)
        udx[:,:] = (np.pi)*np.cos(np.pi*X) 
        
        udn = c4d(u,dx,nx,ny,'X')
        
        errL2 = np.linalg.norm(udx - udn)
        
        print('#----------------------------------------#')
        print('n = %d' % (nx))
        print('L2 error:  %5.3e' % errL2)
        if i>=1:
            rateL2 = np.log(errL2_0/errL2)/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
        
        errL2_0 = errL2
    
    print('#-----------------Dy-------------------#')
    for i in range(3):
        dx = 0.05/(2**i)
        nx = int((xr - xl)/dx)
          
        ny = nx
        
        x = np.linspace(xl,xr,nx+1)
        y = np.linspace(xl,xr,ny+1)
        X,Y = np.meshgrid(x,y, indexing='ij')
        xx = (np.ones((ny+1,1))*x).T
        
        u = np.zeros((nx+1,ny+1))
        udy = np.zeros((nx+1,ny+1))
        udn = np.zeros((nx+1,ny+1))
        
        u[:,:] = np.sin(np.pi*X) + np.sin(2.0*np.pi*Y)
        udy[:,:] = (2.0*np.pi)*np.cos(2.0*np.pi*Y) 
        
        udn = c4d(u,dx,ny,nx,'Y')
                
        errL2 = np.linalg.norm(udy - udn)
        
        print('#----------------------------------------#')
        print('n = %d' % (nx))
        print('L2 error:  %5.3e' % errL2)
        if i>=1:
            rateL2 = np.log(errL2_0/errL2)/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
        
        errL2_0 = errL2
        
    fig, axs = plt.subplots(1,2,figsize=(14,5))
    cs = axs[0].contourf(X, Y, udy, 60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[0], orientation='vertical')
    
    cs = axs[1].contourf(X, Y, udn,60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[1], orientation='vertical')
    
    plt.show()
    fig.tight_layout()