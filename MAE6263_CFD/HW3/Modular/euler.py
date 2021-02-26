#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:20:57 2021

@author: suraj
"""
from rhs import *
from poisson_solvers import *

def euler(nx,ny,dx,dy,dt,re,w,s,ii,jj,isolver,ip,bc):
    r = rhs(nx,ny,dx,dy,re,w,s,isolver)
    w[ii,jj] = w[ii,jj] + dt*r[ii,jj]
    w = bc(nx,ny,w,s)    
    s = poisson(nx,ny,dx,dy,w,ip)
    
    return w, s