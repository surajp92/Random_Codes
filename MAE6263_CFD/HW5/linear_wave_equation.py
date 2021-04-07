#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:11:31 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
a = 250.0
l = 400.0
tm = 1.0
dx = 5.0
dt = 0.02
c = a*dt/dx

n = int(l/dx)
nt = int(tm/dt)

u = np.zeros((n+1,nt+1))

x = np.linspace(0,l,n+1)        
t = np.linspace(0,tm,nt+1)
X,T = np.meshgrid(x,t, indexing='ij')
u0 = 100.0*np.sin(np.pi*(x - 50.0)/60.0)

#%%
k = 0
u[:,k] = u0
u[x <= 50.0,k] = 0.0
u[x >= 110.0,k] = 0.0

i = np.arange(n+1)
for k in range(1,nt+1):
    u[i,k] = u[i,k-1] - c*(u[i,k-1] - u[i-1,k-1])
    
#%%
fig,ax = plt.subplots(1,1,figsize=(7,5))
ax.contourf(T,X,u,60)
plt.show()    