#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:11:31 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

font = {'size' : 14} 
plt.rc('font', **font)

def tdma(a,b,c,r,s,e):
    
    a_ = np.copy(a)
    b_ = np.copy(b)
    c_ = np.copy(c)
    r_ = np.copy(r)
    
    un = np.zeros((np.shape(r)[0],np.shape(r)[1]))
    
    for i in range(s+1,e+1):
        b_[i,:] = b_[i,:] - a_[i,:]*(c_[i-1,:]/b_[i-1,:])
        r_[i,:] = r_[i,:] - a_[i,:]*(r_[i-1,:]/b_[i-1,:])
        
    un[e,:] = r_[e,:]/b_[e,:]
    
    for i in range(e-1,s-1,-1):
        un[i,:] = (r_[i,:] - c_[i,:]*un[i+1,:])/b_[i,:]
    
    del a_, b_, c_, r_
    
    return un

#%% first explicit upwind
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
    
xe = x - a*(tm/2.0)
ue = 100.0*np.sin(np.pi*(xe - 50.0)/60.0)
ue[xe <= 50.0] = 0.0
ue[xe >= 110.0] = 0.0
  
fig,ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(x,ue,'x--',fillstyle='none',ms=8,lw=2,label='Exact')
for i in range(m):
    x = x_all[i]
    u = u_all[i]
    ax.plot(x,u[:,int(nt_list[i]/2)],'o-',fillstyle='none',ms=8,lw=2,label=f'c=${1.0/2**i}$')

ax.legend()    
ax.set_title('First-upwind explicit scheme')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
plt.show() 
fig.savefig('lw_first_explicit_upwind.png', dpi=300)

#%% first implicit upwind
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
    
    s = 1
    e = n-1
    
    lm = c*np.ones((n+1,1))
    dm = -(1.0+c)*np.ones((n+1,1))
    um = -0.0*np.ones((n+1,1))
    
    i = np.arange(n+1)
    for k in range(1,nt+1):
        r = -u[i,k-1].reshape(-1,1)
        u[i,k] = tdma(lm,dm,um,r,s,e).flatten()
    
    x_all.append(x)
    t_all.append(t)
    u_all.append(u)
    
xe = x - a*(tm/2.0)
ue = 100.0*np.sin(np.pi*(xe - 50.0)/60.0)
ue[xe <= 50.0] = 0.0
ue[xe >= 110.0] = 0.0

fig,ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(x,ue,'x--',fillstyle='none',ms=8,lw=2,label='Exact')
for i in range(m):
    x = x_all[i]
    u = u_all[i]
    ax.plot(x,u[:,int(nt_list[i]/2)],'o-',fillstyle='none',ms=8,lw=2,label=f'c=${1.0/2**i}$')
    
ax.legend()    
ax.set_title('First-upwind implicit scheme')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
plt.show()    
fig.savefig('lw_first_implicit_upwind.png', dpi=300)

#%% Lax-wendroff scheme
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

xe = x - a*(tm/2.0)
ue = 100.0*np.sin(np.pi*(xe - 50.0)/60.0)
ue[xe <= 50.0] = 0.0
ue[xe >= 110.0] = 0.0

fig,ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(x,ue,'x--',fillstyle='none',ms=8,lw=2,label='Exact')
for i in range(m):
    x = x_all[i]
    u = u_all[i]
    ax.plot(x,u[:,int(nt_list[i]/2)],'o-',fillstyle='none',ms=8,
            lw=2,label=f'c=${1.0/2**i}$')
ax.legend()    
ax.set_title('Lax-Wendroff explicit scheme')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
plt.show()     
fig.savefig('lw_lax_wendroff.png', dpi=300)

#%% BTCS
u_all = []
t_all = []
x_all = []

l = 600.0
dx = 5.0
dt_list = [0.02,0.05]
nt_list = 1.0/np.array(dt_list)

for i in range(2):

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
    
    s = 1
    e = n-1
    
    lm = 0.5*c*np.ones((n+1,1))
    dm = -1*np.ones((n+1,1))
    um = -0.5*c*np.ones((n+1,1))
    
    i = np.arange(n+1)
    for k in range(1,nt+1):
        r = -u[i,k-1].reshape(-1,1)
        u[i,k] = tdma(lm,dm,um,r,s,e).flatten()
    
    x_all.append(x)
    t_all.append(t)
    u_all.append(u)


for i in range(2):    
    dt = dt_list[i]
    c = a*dt/dx
    
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(1, 2, 1)
    
    x = x_all[i]
    u = u_all[i]
    t = t_all[i]
    ax.plot(x,u[:,0],'o-',fillstyle='none',ms=8,lw=2,label='$t=0.0$')
    ax.plot(x,u[:,-1],'o-',fillstyle='none',ms=8,lw=2,label='$t=1.0$')
        
    ax.legend()    
    ax.set_title(f'BTCS ($c={c}$)')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u$')
    
    X,T = np.meshgrid(x,t, indexing='ij')
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(X, T, u, cmap='viridis',rstride=1, cstride=1, alpha=1,
                           linewidth=0, antialiased=False)
    
    ax.view_init(30, 120)
    ax.set_xticks([0,200,400,600])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_zlabel('$u$')
    ax.set_title(f'BTCS ($c={c}$)')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.tight_layout()
    plt.show()
    fig.savefig(f'lw_btcs_c={c}.png', dpi=300)
    
