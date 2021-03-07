#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 21:49:29 2021

@author: suraj
"""

import numpy as np
#from .thomas_algorithms import *
import matplotlib.pyplot as plt
#from .compact_schemes_first_order_derivative import *
#from .compact_schemes_second_order_derivative import *
#from .rhs_conservative import *

def tdma(a,b,c,r,s,e):
    
    a_ = np.copy(a)
    b_ = np.copy(b)
    c_ = np.copy(c)
    r_ = np.copy(r)
    
    un = np.zeros((np.shape(r)[0],np.shape(r)[1]))
    
    for i in range(s+1,e+1):
        b_[i,:] = b_[i,:] - a_[i,:]*(c_[i-1,:]/b_[i-1,:])
        r_[i,:] = r_[i,:] - a_[i,:]*(r_[i-1,:]/b_[i-1,:])
        
    un[e,:] = r_[e,:]/b_[e,:]
    
    for i in range(e-1,s-1,-1):
        un[i,:] = (r_[i,:] - c_[i,:]*un[i+1,:])/b_[i,:]
    
    del a_, b_, c_, r_
    
    return un

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

def rhs_compact_scheme(nx,ny,dx,dy,re,w,s):
    
    # viscous terms for vorticity transport equation
    # wxx
    uw = np.copy(w)
    us = np.copy(s)

    wxx = c4dd(uw,dx,nx,ny,'XX')
        
    # wyy
    wyy = c4dd(uw,dy,ny,nx,'YY')
    
    lap = wxx + wyy
    
    # convective terms
    
    # sx
    sx = c4d(us,dx,nx,ny,'X')
    
    # sy
    sy = c4d(us,dx,nx,ny,'Y')

    # wx
    wx = c4d(uw,dx,nx,ny,'X')
    
    # wy
    wy = c4d(uw,dx,nx,ny,'Y')
    
    jac = sy*wx - sx*wy
    
    f = np.zeros((nx+1,ny+1))
    
    f[:,:] = -jac + lap/re
    
    return f
    
    