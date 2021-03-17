#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:21:44 2021

@author: suraj
"""
import numpy as np
from rhs import *
from poisson import *

def rk3(nx,ny,dx,dy,dt,re,pr,w,s,th,input_data,bc,bc3):
    ip = input_data['ip']
    isolver = input_data['isolver']
    
    tw = np.zeros((nx+1,ny+1))
    rw = np.zeros((nx+1,ny+1))
    
    tth = np.zeros((nx+1,ny+1))
    rth = np.zeros((nx+1,ny+1))

    i = np.arange(0,nx+1)
    j = np.arange(1,ny)
    ii,jj = np.meshgrid(i,j, indexing='ij')

    # time integration using third-order Runge Kutta method
    aa = 1.0/3.0
    bb = 2.0/3.0
    
    #stage-1
    if isolver == 3:
        w = bc3(nx,ny,w,s)
    else:
        w = bc(nx,ny,w,s) 
    rw, rth = rhs(nx,ny,dx,dy,re,pr,w,s,th,input_data)
    tw[ii,jj] = w[ii,jj] + dt*rw[ii,jj]
    tth[ii,jj] = th[ii,jj] + dt*rth[ii,jj]
    
    tw[nx,:] = tw[0,:]
    tth[nx,:] = tth[0,:]
    
    s = poisson(nx,ny,dx,dy,tw,input_data)

    #stage-2
    if isolver == 3:
        tw = bc3(nx,ny,tw,s)
    else:
        tw = bc(nx,ny,tw,s) 
    rw, rth = rhs(nx,ny,dx,dy,re,pr,tw,s,tth,input_data)
    tw[ii,jj] = 0.75*w[ii,jj] + 0.25*tw[ii,jj] + 0.25*dt*rw[ii,jj]
    tth[ii,jj] = 0.75*th[ii,jj] + 0.25*tth[ii,jj] + 0.25*dt*rth[ii,jj]
    
    tw[nx,:] = tw[0,:]
    tth[nx,:] = tth[0,:]
    
    s[:,:] = poisson(nx,ny,dx,dy,tw,input_data)

    #stage-3
    if isolver == 3:
        tw = bc3(nx,ny,tw,s)
    else:
        tw = bc(nx,ny,tw,s) 
    rw, rth = rhs(nx,ny,dx,dy,re,pr,tw,s,tth,input_data)
    w[ii,jj] = aa*w[ii,jj] + bb*tw[ii,jj] + bb*dt*rw[ii,jj]
    th[ii,jj] = aa*th[ii,jj] + bb*tth[ii,jj] + bb*dt*rth[ii,jj]
    
    w[nx,:] = w[0,:]
    th[nx,:] = th[0,:]
    
    s[:,:] = poisson(nx,ny,dx,dy,w,input_data)
    
    return w, s, th