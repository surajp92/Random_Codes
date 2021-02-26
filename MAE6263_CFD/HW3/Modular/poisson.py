#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:13:52 2021

@author: suraj
"""
from poisson_solvers.fst import *
from poisson_solvers.mg import *

def poisson(nx,ny,dx,dy,f,input_data):
    ip = input_data['ip']
    if ip == 1:
        u = fst(nx,ny,dx,dy,-f)
    elif ip == 2:
        u = mg_n_solver(-f, dx, dy, nx, ny, input_data)
    
    return u



