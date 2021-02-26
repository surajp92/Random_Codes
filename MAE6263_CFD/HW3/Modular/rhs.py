#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:19:49 2021

@author: suraj
"""

from rhs_arakawa import *
from rhs_cs import *

def rhs(nx,ny,dx,dy,re,w,s,isolver):
    if isolver == 1:
        r = rhs_arakawa(nx,ny,dx,dy,re,w,s)
        return r
    elif isolver == 2:
        r = rhs_cs(nx,ny,dx,dy,re,w,s)
        return r