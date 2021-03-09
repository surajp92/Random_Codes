#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:18:27 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import orth

# These are our constants
N = 40  # Number of variables
F = 8  # Forcing


def L96(x, t):
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return d

def dL96(dx, t, x):
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
    dL = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        dL[i] = (x[(i + 1)%N] - x[i - 2])*dx[i - 1] + x[i-1]*(dx[i] - dx[i-2]) - dx[i] 
    return dL

x0 = F * np.ones(N)  # Initial state (equilibrium)
x0[0] += 0.01  # Add small perturbation to the first variable
t = np.arange(0.0, 5.0, 0.01)
x = odeint(L96, x0, t)

x0 = x[-1,:]

t = np.arange(0.0, 10.0, 0.01)
x = odeint(L96, x0, t)

plt.contourf(x.T, 120)
plt.show()

#%%
dt = 0.01

# The number of iterations to throw away
nTransients = 10
# The number of time steps to integrate over
nIterates = 10000

xs = np.copy(x0)

t = np.linspace(0,nIterates*dt,nIterates+1)
xt = odeint(L96, xs, t)

eye = np.eye(N)
 
e1 = eye[:,0]
t = np.linspace(0,nTransients*dt,nTransients+1)
dxt = odeint(dL96, e1, t, args=(xs,))


#%%
# Estimate the LCEs
# The number of iterations to throw away
nTransients = 100
# The number of iterations to over which to estimate
#  This is really the number of pull-backs
nIterates = 10000
# The number of iterations per pull-back
nItsPerPB = 10

x0 = np.copy(xs)

# Initial tangent vectors
ee = np.eye(N)

for n in range(nTransients):
#    for i in range(nItsPerPB):
#        t1 = np.linspace(0,dt,2)
    t1 = np.linspace(0,nItsPerPB*dt,nItsPerPB)
    x = odeint(L96, x0, t1)
    x0 = x[-1]
    
    for k in range(N):
        e = ee[:,k]
        et = odeint(dL96, e, t1, args=(x0,))
        ee[:,k] = et[-1]
    
    eec = np.copy(ee)
    
    for k in range(3):
        dote = []
        
        # Pull-back: Remove any other vector components from kth component
        for i in range(k):
            dote.append(np.sum(ee[:,i]*ee[:,k]))
        
        for i in range(k):
            ee[:,k] = ee[:,k] - dote[i]*ee[:,i]
        
        # Normalize the tangent vector
        d= np.linalg.norm(ee[:,k])
        ee[:,k] = ee[:,k]/d
        
#%%
# Okay, now we're ready to begin the estimation
LCE = np.zeros(N)

for n in range(nIterates):
#    for i in range(nItsPerPB):
#        t1 = np.linspace(0,dt,2)
    t1 = np.linspace(0,nItsPerPB*dt,nItsPerPB)
    x = odeint(L96, x0, t1)
    x0 = x[-1]
    
    for k in range(3):
        e = ee[:,k]
        et = odeint(dL96, e, t1, args=(x0,))
        ee[:,k] = et[-1]
    
    eec = np.copy(ee)
    
    for k in range(3):
        dote = []
        for i in range(k):
            dote.append(np.sum(ee[:,i]*ee[:,k]))
        
        for i in range(k):
            ee[:,k] = ee[:,k] - dote[i]*ee[:,i]
        
        d= np.linalg.norm(ee[:,k])
        ee[:,k] = ee[:,k]/d
        
        LCE[k] = LCE[k] + np.log(d)
    
    
#%%
# Convert to per-iterate, per-second LCEs and to base-2 logs
IntegrationTime = dt * float(nItsPerPB) * float(nIterates)
LCE = LCE / IntegrationTime
        
#%%
print(f'LCE1 = {round(LCE[0],4)}')
print(f'LCE2 = {round(LCE[1],4)}')
print(f'LCE3 = {round(LCE[2],4)}')