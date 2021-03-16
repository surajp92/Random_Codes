#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 21:49:29 2021

@author: suraj
"""

import numpy as np
from .thomas_algorithms import *
import matplotlib.pyplot as plt
from .compact_schemes_first_order_derivative import *
from .compact_schemes_second_order_derivative import *
from .rhs_conservative import *

def rhs_compact_scheme(nx,ny,dx,dy,re,w,s):
    
    # viscous terms for vorticity transport equation
    # wxx
    uw = np.copy(w)
    us = np.copy(s)

    wxx = c4dd(uw,dx,dy,nx,ny,'XX')
        
    # wyy
    wyy = c4dd(uw,dx,dy,nx,ny,'YY')
    
    lap = wxx + wyy
    
    # convective terms
    
    # sx
    sx = c4d(us,dx,dy,nx,ny,'X')
    
    # sy
    sy = c4d(us,dx,dy,nx,ny,'Y')

    # wx
    wx = c4d(uw,dx,dy,nx,ny,'X')
    
    # wy
    wy = c4d(uw,dx,dy,nx,ny,'Y')
    
    jac = sy*wx - sx*wy
    
    f = np.zeros((nx+1,ny+1))
    
    f[:,:] = -jac + lap/re
    
    return f
    
    