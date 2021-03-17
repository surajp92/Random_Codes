#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:20:57 2021

@author: suraj
"""
from rhs import *
from poisson import *

def euler(nx,ny,dx,dy,dt,re,pr,w,s,th,input_data,bc,bc3):
    
    i = np.arange(0,nx+1)
    j = np.arange(1,ny)
    ii,jj = np.meshgrid(i,j, indexing='ij')

    if input_data['isolver'] == 3:
        w = bc3(nx,ny,w,s)
    else:
        w = bc(nx,ny,w,s) 
        
    rw, rth = rhs(nx,ny,dx,dy,re,pr,w,s,th,input_data)

    w[ii,jj] = w[ii,jj] + dt*rw[ii,jj]
    th[ii,jj] = th[ii,jj] + dt*rth[ii,jj]
    
    w[nx,:] = w[0,:]
    th[nx,:] = th[0,:]
    
    s = poisson(nx,ny,dx,dy,w,input_data)
    
    return w, s, th