#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:13:52 2021

@author: suraj
"""
from fst import *
from mg import *

def poisson(nx,ny,dx,dy,f,ip,):
    if ip == 1:
        u = fst(nx,ny,dx,dy,-f)
    elif ip == 2:
        u = mg_n_solver(-f, dx, dy, nx, ny)
    
    return u



