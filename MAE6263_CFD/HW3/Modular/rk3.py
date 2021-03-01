#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:21:44 2021

@author: suraj
"""
from rhs import *
from poisson_solvers import *

def rk3(nx,ny,dx,dy,dt,re,w,s,t,ii,jj,isolver,bc,ip):
    # time integration using third-order Runge Kutta method
    aa = 1.0/3.0
    bb = 2.0/3.0
    
    r = np.zeros((nx+1,ny+1))

    #stage-1
    r[:,:] = rhs(nx,ny,dx,dy,re,w[:,:],s[:,:],isolver)
    t[ii,jj] = w[ii,jj] + dt*r[ii,jj]
    t = bc(nx,ny,t,s)    
    s = poisson(nx,ny,dx,dy,t,ip)

    #stage-2
    r[:,:] = rhs(nx,ny,dx,dy,re,t[:,:],s[:,:],isolver)
    t[ii,jj] = 0.75*w[ii,jj] + 0.25*t[ii,jj] + 0.25*dt*r[ii,jj]
    t[:,:] = bc(nx,ny,t,s)    
    s[:,:] = poisson(nx,ny,dx,dy,t,ip)

    #stage-3
    r[:,:] = rhs(nx,ny,dx,dy,re,t[:,:],s[:,:],isolver)
    w[ii,jj] = aa*w[ii,jj] + bb*t[ii,jj] + bb*dt*r[ii,jj]
    w[:,:] = bc(nx,ny,w[:,:], s)
    s[:,:] = poisson(nx,ny,dx,dy,w,ip)
    
    return w, s