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

#%% Lax explicit shceme
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
fig.savefig('nlw_lax_explicit1.png', dpi=300)

#%% Lax explicit shceme
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
fig.savefig('nlw_lax_explicit2.png', dpi=300)

#%% Lax wendroff shceme
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
        u[i,k] = u[i,k-1] - (0.5*dt/dx)*(0.5*u[(i+1),k-1]**2 - 0.5*u[i-1,k-1]**2) + \
            ((0.5*dt/dx)**2)*((u[i+1,k-1] + u[i,k-1])*(0.5*u[(i+1),k-1]**2 - 0.5*u[i,k-1]**2) - \
                            (u[i,k-1] + u[i-1,k-1])*(0.5*u[(i),k-1]**2 - 0.5*u[i-1,k-1]**2))
    
    x_all.append(x)
    t_all.append(t)
    u_all.append(u)

fig,ax = plt.subplots(1,1,figsize=(10,5))
for i in range(4):
    x = x_all[0]
    u = u_all[0]
    ax.plot(x,u[:,nt_list[i]],'o-',lw=2,label=f'$t={nt_list[i]*dt:0.2f}$')
ax.legend()    
ax.set_title('Lax Wendroff scheme')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
plt.show()    
fig.savefig('nlw_lax_wendroff1.png', dpi=300)

#%% Lax Wendroff shceme
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
        u[i,k] = u[i,k-1] - (0.5*dt/dx)*(0.5*u[(i+1),k-1]**2 - 0.5*u[i-1,k-1]**2) + \
            ((0.5*dt/dx)**2)*((u[i+1,k-1] + u[i,k-1])*(0.5*u[(i+1),k-1]**2 - 0.5*u[i,k-1]**2) - \
                            (u[i,k-1] + u[i-1,k-1])*(0.5*u[(i),k-1]**2 - 0.5*u[i-1,k-1]**2))
    
    x_all.append(x)
    t_all.append(t)
    u_all.append(u)
    
fig,ax = plt.subplots(1,1,figsize=(10,5))
for i in range(2):
    x = x_all[i]
    u = u_all[i]
    ax.plot(x,u[:,-1],'o-',lw=2,label=f'$c={dt_list[i]/dx:0.2f}$')
ax.legend()    
ax.set_title('Lax wendroff scheme')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
plt.show() 
fig.savefig('nlw_lax_wendroff2.png', dpi=300)

#%% Beam and warming implicit method
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
    
    s = 1
    e = n-1
    
    lm = np.zeros((n+1,1))
    dm = np.zeros((n+1,1))
    um = np.zeros((n+1,1))
    rm = np.zeros((n+1,1))
    
    i = np.arange(1,n)
    
    for k in range(1,nt+1):
        lm[i,0] = -(dt/(4.0*dx))*u[i-1,k-1]
        dm[i,0] = 1.0
        um[i,0] = (dt/(4.0*dx))*u[i+1,k-1]
            
        rm[i,0] = u[i,k-1] - (0.5*dt/dx)*(0.5*u[(i+1),k-1]**2 - 0.5*u[i-1,k-1]**2) + \
            (0.25*dt/dx)*u[i+1,k-1]*u[i+1,k-1] - (0.25*dt/dx)*u[i-1,k-1]*u[i-1,k-1]
        
        dm[0,0] = 1.0
        rm[0,0] = 1.0
        dm[n,0] = 1.0
        rm[n,0] = 0.0
                
        u[:,k] = tdma(lm,dm,um,rm,0,n).flatten() #[1:-1,0]
        # u[0,k] = 1.0
    
    x_all.append(x)
    t_all.append(t)
    u_all.append(u)

fig,ax = plt.subplots(1,1,figsize=(10,5))
for i in range(4):
    x = x_all[0]
    u = u_all[0]
    ax.plot(x,u[:,nt_list[i]],'o-',lw=2,label=f'$t={nt_list[i]*dt:0.2f}$')

ax.set_ylim([-0.2,1.65])    
ax.legend()    
ax.set_title('Beam-Warming implicit scheme')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
plt.show() 
fig.savefig('nlw_beam_warming.png', dpi=300)

#%% Beam and warming implicit method with artificial dissipation
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

epsilon = 0.1

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
    
    s = 1
    e = n-1
    
    lm = np.zeros((n+1,1))
    dm = np.zeros((n+1,1))
    um = np.zeros((n+1,1))
    rm = np.zeros((n+1,1))
    
    for k in range(1,nt+1):
        i = np.arange(1,n)
        lm[i,0] = -(dt/(4.0*dx))*u[i-1,k-1]
        dm[i,0] = 1.0
        um[i,0] = (dt/(4.0*dx))*u[i+1,k-1]
            
        rm[i,0] = u[i,k-1] - (0.5*dt/dx)*(0.5*u[(i+1),k-1]**2 - 0.5*u[i-1,k-1]**2) + \
            (0.25*dt/dx)*u[i+1,k-1]*u[i+1,k-1] - (0.25*dt/dx)*u[i-1,k-1]*u[i-1,k-1] 
                
        i = np.arange(2,n-1)
        rm[i,0] = rm[i,0] - epsilon*(u[i+2,k-1] - 4.0*u[i+1,k-1] + 6.0*u[i,k-1] - \
                                     4.0*u[i-1,k-1] + u[i-2,k-1])                
        
        dm[0,0] = 1.0
        rm[0,0] = 1.0
        dm[n,0] = 1.0
        rm[n,0] = 0.0
                
        u[:,k] = tdma(lm,dm,um,rm,0,n).flatten() #[1:-1,0]
        # u[0,k] = 1.0
    
    x_all.append(x)
    t_all.append(t)
    u_all.append(u)

fig,ax = plt.subplots(1,1,figsize=(10,5))
for i in range(4):
    x = x_all[0]
    u = u_all[0]
    ax.plot(x,u[:,nt_list[i]],'o-',lw=2,label=f'$t={nt_list[i]*dt:0.2f}$')

ax.set_ylim([-0.2,1.65])    
ax.legend()    
ax.set_title('Beam-Warming implicit scheme with damping')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
plt.show() 
fig.savefig('nlw_beam_warming_damping.png', dpi=300)

#%% Runge-kutta with damping
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
        ut = np.copy(u[:,k-1])
        
        ut[i] = u[i,k-1] - (dt/(4.0*dx))*(0.25*ut[(i+1)]**2 - 0.25*ut[i-1]**2)
        
        ut[i] = u[i,k-1] - (dt/(3.0*dx))*(0.25*ut[(i+1)]**2 - 0.25*ut[i-1]**2)        

        ut[i] = u[i,k-1] - (dt/(2.0*dx))*(0.25*ut[(i+1)]**2 - 0.25*ut[i-1]**2)

        u[i,k] = u[i,k-1] - (dt/dx)*(0.25*ut[(i+1)]**2 - 0.25*ut[i-1]**2)
        
    
    x_all.append(x)
    t_all.append(t)
    u_all.append(u)

fig,ax = plt.subplots(1,1,figsize=(10,5))
for i in range(4):
    x = x_all[0]
    u = u_all[0]
    ax.plot(x,u[:,nt_list[i]],'o-',lw=2,label=f'$t={nt_list[i]*dt:0.2f}$')

ax.set_ylim([-0.2,1.65])    
ax.legend()    
ax.set_title('Runge-Kutta explicit scheme')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
plt.show() 
fig.savefig('nlw_runge_kutta.png', dpi=300)


#%% Runge-kutta with damping
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
epsilon = 0.1

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
    j = np.arange(2,n-1)
    
    for k in range(1,nt+1):                
        ut = np.copy(u[:,k-1])
        
        ut[i] = u[i,k-1] - (dt/(4.0*dx))*(0.25*ut[(i+1)]**2 - 0.25*ut[i-1]**2)
            
        ut[i] = u[i,k-1] - (dt/(3.0*dx))*(0.25*ut[(i+1)]**2 - 0.25*ut[i-1]**2)     

        ut[i] = u[i,k-1] - (dt/(2.0*dx))*(0.25*ut[(i+1)]**2 - 0.25*ut[i-1]**2)

        u[i,k] = u[i,k-1] - (dt/dx)*(0.25*ut[(i+1)]**2 - 0.25*ut[i-1]**2)
        # u[j,k] = u[j,k] - epsilon*(ut[j+2] - 4.0*ut[j+1] + 6.0*ut[j] - \
        #                              4.0*ut[j-1] + ut[j-2]) 
        u[j,k] = u[j,k] - epsilon*(u[j+2,k-1] - 4.0*u[j+1,k-1] + 6.0*u[j,k-1] - \
                                     4.0*u[j-1,k-1] + u[j-2,k-1]) 
        
    
    x_all.append(x)
    t_all.append(t)
    u_all.append(u)

fig,ax = plt.subplots(1,1,figsize=(10,5))
for i in range(4):
    x = x_all[0]
    u = u_all[0]
    ax.plot(x,u[:,nt_list[i]],'o-',lw=2,label=f'$t={nt_list[i]*dt:0.2f}$')

ax.set_ylim([-0.2,1.65])    
ax.legend()    
ax.set_title('Runge-Kutta explicit scheme  with damping')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
plt.show() 
fig.savefig('nlw_runge_kutta_damping.png', dpi=300)