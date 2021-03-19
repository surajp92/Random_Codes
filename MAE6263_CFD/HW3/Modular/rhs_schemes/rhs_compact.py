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

    wxx = c6dd_b5_d(uw,dx,nx,ny,'XX')
        
    # wyy
    wyy = c6dd_b5_d(uw,dy,ny,nx,'YY')
    
    lap = wxx + wyy
    
    # convective terms
    
    # sx
    sx = c6d_b5_d(us,dx,nx,ny,'X')
    
    # sy
    sy = c6d_b5_d(us,dy,ny,nx,'Y')

    # wx
    wx = c6d_b5_d(uw,dx,nx,ny,'X')
    
    # wy
    wy = c6d_b5_d(uw,dy,ny,nx,'Y')
    
    jac = sy*wx - sx*wy
    
    f = np.zeros((nx+1,ny+1))
    
    f[:,:] = -jac + lap/re
    
    return f
    
    