#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 21:22:24 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
nx = 5
ny = 5

ne = (nx-1)*(ny-1)*2
ne_x = (nx - 1)*2

t = np.zeros((ne,3), dtype=int)

for j in range(ny-1):
    i1 = np.arange(nx-1) 
    ise = 2*i1 + j*ne_x
    # print(is_)
    
    t[ise,0] = i1 + j*nx
    t[ise,1] = t[ise,0] + 1
    t[ise,2] = t[ise,0] + nx
    
    iso = ise + 1

    t[iso,0] = t[ise,2] + 1
    t[iso,1] = t[iso,0] - 1
    t[iso,2] = t[iso,0] - nx
    
    # t[iso,0] = is_ + nx - 2*j - i1
    # t[iso,1] = is_ + nx - 2*j - 1 - i1
    # t[iso,2] = is_ + nx - 2*j - nx - i1
    
# t[2*i1+1,0] = i1 + nx
# t[2*i1+1,1] = i1 + nx - 1
# t[2*i1+1,2] = i1 + nx - nx


print(t)