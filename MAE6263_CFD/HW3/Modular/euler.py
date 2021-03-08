#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:20:57 2021

@author: suraj
"""
from rhs import *
from poisson import *

def euler(nx,ny,dx,dy,dt,re,w,s,ii,jj,input_data,bc,bc3):
    
    if input_data['isolver'] == 3:
        w = bc3(nx,ny,w,s)
    else:
        w = bc(nx,ny,w,s) 
        
    r = rhs(nx,ny,dx,dy,re,w,s,input_data)
    w[ii,jj] = w[ii,jj] + dt*r[ii,jj]
    
    
        
    s = poisson(nx,ny,dx,dy,w,input_data)
    
    return w, s