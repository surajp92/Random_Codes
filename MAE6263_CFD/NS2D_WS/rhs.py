#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:19:49 2021

@author: suraj
"""

from rhs_schemes.rhs_conservative import *
from rhs_schemes.rhs_compact import *

def rhs(nx,ny,dx,dy,re,w,s,input_data):
    isolver = input_data['isolver']
        
    if isolver == 1:
        r = rhs_arakawa(nx,ny,dx,dy,re,w,s)
        return r
    elif isolver == 2:
        r = rhs_cs(nx,ny,dx,dy,re,w,s)
        return r
    elif isolver == 3:
        r = rhs_compact_scheme(nx,ny,dx,dy,re,w,s)
        return r