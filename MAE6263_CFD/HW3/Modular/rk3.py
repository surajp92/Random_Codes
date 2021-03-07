#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:21:44 2021

@author: suraj
"""
import numpy as np
from rhs import *
from poisson import *

def rk3(nx,ny,dx,dy,dt,re,w,s,ii,jj,input_data,bc,bc3):
    ip = input_data['ip']
    isolver = input_data['isolver']
    t = np.zeros((nx+1,ny+1))
    r = np.zeros((nx+1,ny+1))

    # time integration using third-order Runge Kutta method
    aa = 1.0/3.0
    bb = 2.0/3.0
    
    #stage-1
    r[:,:] = rhs(nx,ny,dx,dy,re,w[:,:],s[:,:],input_data)
    t[ii,jj] = w[ii,jj] + dt*r[ii,jj]
    if isolver == 3:
        t = bc3(nx,ny,t,s)
    else:
        t = bc(nx,ny,t,s) 
    s = poisson(nx,ny,dx,dy,t,input_data)

    #stage-2
    r[:,:] = rhs(nx,ny,dx,dy,re,t[:,:],s[:,:],input_data)
    t[ii,jj] = 0.75*w[ii,jj] + 0.25*t[ii,jj] + 0.25*dt*r[ii,jj]
    if isolver == 3:
        t = bc3(nx,ny,t,s)
    else:
        t = bc(nx,ny,t,s)    
    s[:,:] = poisson(nx,ny,dx,dy,t,input_data)

    #stage-3
    r[:,:] = rhs(nx,ny,dx,dy,re,t[:,:],s[:,:],input_data)
    w[ii,jj] = aa*w[ii,jj] + bb*t[ii,jj] + bb*dt*r[ii,jj]
    if isolver == 3:
        w = bc3(nx,ny,w,s)
    else:
        w = bc(nx,ny,w,s) 
    s[:,:] = poisson(nx,ny,dx,dy,w,input_data)
    
    return w, s