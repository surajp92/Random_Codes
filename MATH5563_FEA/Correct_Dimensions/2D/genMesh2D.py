#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 21:22:24 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
domain = [1.0,1.0]
xl = domain[0]
yl = domain[1]

nx = 3
ny = 3

ne = (nx-1)*(ny-1)*2
ne_x = (nx - 1)*2

t = np.zeros((ne,3), dtype=int)

for j in range(ny-1):
    i1 = np.arange(nx-1) 
    ise = 2*i1 + j*ne_x
    print(i1, j, ise)
    
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

#%%

t1 = np.zeros((ne,3), dtype=int)
i = np.arange(nx-1).reshape(1,-1) 
j = np.arange(ny-1).reshape(-1,1) 

ise = 2*i + j*ne_x
ise = ise.flatten()


t1[ise,0] = (i + j*nx).flatten()
t1[ise,1] = t1[ise,0] + 1
t1[ise,2] = t1[ise,0] + nx

iso = ise + 1

t1[iso,0] = t1[ise,2] + 1
t1[iso,1] = t1[iso,0] - 1
t1[iso,2] = t1[iso,0] - nx


print(t1)

print(t - t1)

#%%
t1 = np.zeros((ne,3), dtype=int)
i = np.arange(nx-1).reshape(1,-1) 
j = np.arange(ny-1).reshape(-1,1) 

ise = 2*np.arange(0,int(ne/2))
# ise = ise.flatten()

t1[ise,0] = (i + j*nx).flatten()
t1[ise,1] = t1[ise,0] + 1
t1[ise,2] = t1[ise,0] + nx

iso = ise + 1

t1[iso,0] = t1[ise,2] + 1
t1[iso,1] = t1[iso,0] - 1
t1[iso,2] = t1[iso,0] - nx


print(t1)

print(t - t1)

#%%
nn = nx*ny
p = np.zeros((nn,2))
x = np.linspace(0,xl,nx)
y = np.linspace(0,xl,ny)

X, Y = np.meshgrid(x,y,indexing='ij')

p[:,0] = np.reshape(X,[-1,],order='f')
p[:,1] = np.reshape(Y,[-1,],order='f')