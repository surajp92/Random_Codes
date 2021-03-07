#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 18:52:37 2021

@author: suraj
"""

import numpy as np
from .thomas_algorithms import *
import matplotlib.pyplot as plt

def c4dd(f,h,nx,ny,isign):
    
    if isign == 'XX':
        u = np.copy(f)
    if isign == 'YY':
        u = np.copy(f.T)
        
    a = np.zeros((nx+1,ny+1))
    b = np.zeros((nx+1,ny+1))
    c = np.zeros((nx+1,ny+1))
    r = np.zeros((nx+1,ny+1))
    
    i = 0
    a[i,:] = 0.0
    b[i,:] = 1.0
    c[i,:] = 11.0
    r[i,:] = (13.0*u[i,:] - 27.0*u[i+1,:] + 15.0*u[i+2,:] - u[i+3,:])/(h**2)
    
    ii = np.arange(1,nx)
    
    a[ii,:] = 1.0/10.0
    b[ii,:] = 1.0
    c[ii,:] = 1.0/10.0
    r[ii,:] = 6.0*(u[ii-1,:] - 2.0*u[ii,:] + u[ii+1,:])/(5*h*h)
    
    i = nx
    a[i,:] = 11.0
    b[i,:] = 1.0
    c[i,:] = 0.0
    r[i,:] = (13.0*u[i,:] - 27.0*u[i-1,:] + 15.0*u[i-2,:] - u[i-3,:])/(h**2)
    
    start = 0
    end = nx
    udd = tdma(a,b,c,r,start,end)
    
    if isign == 'XX':
        fdd = np.copy(udd)
    if isign == 'YY':
        fdd = np.copy(udd.T)
    
    return fdd

if __name__ == "__main__":
    xl = -1.0
    xr = 1.0
    
    print('#-----------------Dxx-------------------#')
    for i in range(3):
        dx = 0.1/(2**i)
        nx = int((xr - xl)/dx)
          
        ny = nx
        
        x = np.linspace(xl,xr,nx+1)
        y = np.linspace(xl,xr,ny+1)
        X,Y = np.meshgrid(x,y, indexing='ij')
        xx = (np.ones((ny+1,1))*x).T
        
        u = np.zeros((nx+1,ny+1))
        uddx = np.zeros((nx+1,ny+1))
        uddn = np.zeros((nx+1,ny+1))
        
        u[:,:] = np.sin(np.pi*X) + np.sin(2.0*np.pi*Y)
        uddx[:,:] = -(np.pi**2)*np.sin(np.pi*X) 
        
        uddn = c4dd(u,dx,nx,ny,'XX')
        
        errL2 = np.linalg.norm(uddx - uddn)
        
        print('#----------------------------------------#')
        print('n = %d' % (nx))
        print('L2 error:  %5.3e' % errL2)
        if i>=1:
            rateL2 = np.log(errL2_0/errL2)/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
        
        errL2_0 = errL2
    
    print('#-----------------Dyy-------------------#')
    for i in range(5):
        dx = 0.1/(2**i)
        nx = int((xr - xl)/dx)
          
        ny = nx
        
        x = np.linspace(xl,xr,nx+1)
        y = np.linspace(xl,xr,ny+1)
        X,Y = np.meshgrid(x,y, indexing='ij')
        xx = (np.ones((ny+1,1))*x).T
        
        u = np.zeros((nx+1,ny+1))
        uddy = np.zeros((nx+1,ny+1))
        uddn = np.zeros((nx+1,ny+1))
        
        u[:,:] = np.sin(np.pi*X) + np.sin(2.0*np.pi*Y)
        uddy[:,:] = -(4.0*np.pi**2)*np.sin(2.0*np.pi*Y) 
        
        uddn = c4dd(u,dx,ny,nx,'YY')
        
        errL2 = np.linalg.norm(uddy - uddn)
        
        print('#----------------------------------------#')
        print('n = %d' % (nx))
        print('L2 error:  %5.3e' % errL2)
        if i>=1:
            rateL2 = np.log(errL2_0/errL2)/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
        
        errL2_0 = errL2
        
    fig, axs = plt.subplots(1,2,figsize=(14,5))
    cs = axs[0].contourf(X, Y, uddy, 60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[0], orientation='vertical')
    
    cs = axs[1].contourf(X, Y, uddn,60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[1], orientation='vertical')
    
    plt.show()
    fig.tight_layout()
    