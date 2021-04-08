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
a = 1.0
l = 4.0
tm = 1.8
dx = 0.1
m = 1
dt_list = [0.1]
u_all = []
t_all = []
x_all = []
nt_list = [0,6,12,18]

for i in range(m):
    dt = dt_list[i]
    c = a*dt/dx
    
    n = int(l/dx)
    nt = int(tm/dt)
    
    u = np.zeros((n+1,nt+1))
    
    x = np.linspace(0,l,n+1)        
    t = np.linspace(0,tm,nt+1)
    X,T = np.meshgrid(x,t, indexing='ij')
    
    k = 0
    u[x <= 2.0,k] = 1.0
    u[x >= 2.0,k] = 0.0
    
    u[0,:] = 1.0
    
    i = np.arange(1,n)
    for k in range(1,nt+1):
        u[i,k] = 0.5*(u[i+1,k-1] + u[i-1,k-1]) - (0.5*dt/dx)*(0.5*u[(i+1),k-1]**2 - 0.5*u[i-1,k-1]**2)
    
    x_all.append(x)
    t_all.append(t)
    u_all.append(u)
    
#%%
fig,ax = plt.subplots(1,1,figsize=(7,5))
ax.contourf(T,X,u,60)
plt.show()    

#%%
fig,ax = plt.subplots(1,1,figsize=(10,5))
for i in range(4):
    x = x_all[0]
    u = u_all[0]
    ax.plot(x,u[:,nt_list[i]],'o-',lw=2,label=f'$t={nt_list[i]*dt:0.2f}$')
ax.legend()    
ax.set_title('Lax explicit scheme')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
plt.show()    

#%%
m = 2
dt_list = [0.1,0.05]
u_all = []
t_all = []
x_all = []
nt_list = [0,6,12,18]

for i in range(m):
    dt = dt_list[i]
    c = a*dt/dx
    
    n = int(l/dx)
    nt = int(tm/dt)
    
    u = np.zeros((n+1,nt+1))
    
    x = np.linspace(0,l,n+1)        
    t = np.linspace(0,tm,nt+1)
    X,T = np.meshgrid(x,t, indexing='ij')
    
    k = 0
    u[x <= 2.0,k] = 1.0
    u[x >= 2.0,k] = 0.0
    
    u[0,:] = 1.0
    
    i = np.arange(1,n)
    for k in range(1,nt+1):
        u[i,k] = 0.5*(u[i+1,k-1] + u[i-1,k-1]) - (0.5*dt/dx)*(0.5*u[(i+1),k-1]**2 - 0.5*u[i-1,k-1]**2)
    
    x_all.append(x)
    t_all.append(t)
    u_all.append(u)
    
#%%
fig,ax = plt.subplots(1,1,figsize=(10,5))
for i in range(2):
    x = x_all[i]
    u = u_all[i]
    ax.plot(x,u[:,-1],'o-',lw=2,label=f'$c={dt_list[i]/dx:0.2f}$')
ax.legend()    
ax.set_title('Lax explicit scheme')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
plt.show()   