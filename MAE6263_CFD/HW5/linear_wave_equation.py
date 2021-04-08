#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:11:31 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

font = {'size' : 16} 
plt.rc('font', **font)

#%%
a = 250.0
l = 400.0
tm = 1.0
dx = 5.0
m = 3
dt_list = [0.02,0.01,0.005]
u_all = []
t_all = []
x_all = []
nt_list = 1.0/np.array(dt_list)

for i in range(m):
    dt = dt_list[i]
    c = a*dt/dx
    
    n = int(l/dx)
    nt = int(tm/dt)
    
    u = np.zeros((n+1,nt+1))
    
    x = np.linspace(0,l,n+1)        
    t = np.linspace(0,tm,nt+1)
    X,T = np.meshgrid(x,t, indexing='ij')
    u0 = 100.0*np.sin(np.pi*(x - 50.0)/60.0)
    
    k = 0
    u[:,k] = u0
    u[x <= 50.0,k] = 0.0
    u[x >= 110.0,k] = 0.0
    
    i = np.arange(n+1)
    for k in range(1,nt+1):
        u[i,k] = u[i,k-1] - c*(u[i,k-1] - u[i-1,k-1])
    
    x_all.append(x)
    t_all.append(t)
    u_all.append(u)
    
#%%
fig,ax = plt.subplots(1,1,figsize=(7,5))
ax.contourf(T,X,u,60)
plt.show()    

#%%
fig,ax = plt.subplots(1,1,figsize=(10,5))
for i in range(m):
    x = x_all[i]
    u = u_all[i]
    ax.plot(x,u[:,int(nt_list[i]/2)],'o-',lw=2,label=f'c=${1.0/2**i}$')
ax.legend()    
ax.set_title('First-upwind explicit scheme')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
plt.show()    

#%%
u_all = []
t_all = []
x_all = []
nt_list = 1.0/np.array(dt_list)

for i in range(m):
    dt = dt_list[i]
    c = a*dt/dx
    
    n = int(l/dx)
    nt = int(tm/dt)
    
    u = np.zeros((n+1,nt+1))
    
    x = np.linspace(0,l,n+1)        
    t = np.linspace(0,tm,nt+1)
    X,T = np.meshgrid(x,t, indexing='ij')
    u0 = 100.0*np.sin(np.pi*(x - 50.0)/60.0)
    
    k = 0
    u[:,k] = u0
    u[x <= 50.0,k] = 0.0
    u[x >= 110.0,k] = 0.0
    
    i = np.arange(n+1)
    for k in range(1,nt+1):
        u[i,k] = u[i,k-1] - 0.5*c*(u[(i+1)%(n+1),k-1] - u[i-1,k-1]) + \
        0.5*c**2*(u[(i+1)%(n+1),k-1] - 2.0*u[i,k-1] + u[i-1,k-1])
    
    x_all.append(x)
    t_all.append(t)
    u_all.append(u)

fig,ax = plt.subplots(1,1,figsize=(10,5))
for i in range(m):
    x = x_all[i]
    u = u_all[i]
    ax.plot(x,u[:,int(nt_list[i]/2)],'o-',lw=2,label=f'c=${1.0/2**i}$')
ax.legend()    
ax.set_title('Lax-Wendroff explicit scheme')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
plt.show()     