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

mesh_t = np.zeros((ne,3), dtype=int)

i = np.arange(nx-1).reshape(1,-1) 
j = np.arange(ny-1).reshape(-1,1) 

ise = 2*np.arange(0,int(ne/2))
# ise = ise.flatten()

mesh_t[ise,0] = (i + j*nx).flatten()
mesh_t[ise,1] = mesh_t[ise,0] + 1
mesh_t[ise,2] = mesh_t[ise,0] + nx

iso = ise + 1

mesh_t[iso,0] = mesh_t[ise,2] + 1
mesh_t[iso,1] = mesh_t[iso,0] - 1
mesh_t[iso,2] = mesh_t[iso,0] - nx


print(mesh_t)

#%%
nnodes = nx*ny
mesh_p = np.zeros((nnodes,2))
x = np.linspace(0,xl,nx)
y = np.linspace(0,xl,ny)

X, Y = np.meshgrid(x,y,indexing='ij')

mesh_p[:,0] = np.reshape(X,[-1,],order='f')
mesh_p[:,1] = np.reshape(Y,[-1,],order='f')

#%%
nedges = (nx-1)*(ny) + (ny-1)*nx + (nx-1)*(ny-1)

n_p = nnodes + nedges
fem_p = np.zeros((n_p,2))

fem_p[:nnodes,:] = mesh_p 

comb_list = [[0,1],[1,2],[0,2]]

start = nnodes

for list_ in comb_list:
    aa = (mesh_p[mesh_t[:,list_[0]]] + mesh_p[mesh_t[:,list_[1]]])/2.0
    aau, indices = np.unique(aa, axis=0, return_index=True)
    
    end = start+indices.shape[0]
    fem_p[start:end,:] = aa[sorted(indices),:]
    
    start = end

print(fem_p)

#%%
aa = [mesh_t[:,comb_list[0]], mesh_t[:,comb_list[1]], mesh_t[:,comb_list[2]]]
aa = np.array(aa)
aa = np.reshape(aa, [24,2])

#%%
aa = np.sort(aa, axis=1)
aau, indices = np.unique(aa, axis=0, return_index=True)
