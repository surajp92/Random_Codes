#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 21:07:50 2021

@author: suraj
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import orth


tmax = 30
dt = 0.001

# These are our constants
N = 8  # Number of variables
F = 20  # Forcing


def L96(x, t):
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return d

x0 = F * np.ones(N)  # Initial state (equilibrium)
x0[0] += 0.01  # Add small perturbation to the first variable
t = np.arange(0.0, 5.0, 0.01)
x = odeint(L96, x0, t)

x0 = x[-1,:]

t = np.arange(0.0, 10.0, 0.01)
x = odeint(L96, x0, t)

plt.contourf(x.T, 120)
plt.show()

epsilon_ = 1.0e-8

xn1 = np.copy(x[-1,:])
xn2 = np.copy(x[-1,:])
xn2[2] = xn1[2] + epsilon_

#%%
nt = int(tmax/dt)
t = np.linspace(0,tmax,nt+1)

x1 = odeint(L96, xn1, t)
x2 = odeint(L96, xn2, t)


#%%
d = np.linalg.norm(x1-x2, axis=1)

# plt.plot(d,'-')
plt.semilogy(t,d,'-')
plt.show()

#%%
ts = t[1:5001]
ds = d[1:5001]

slope, intercept = np.polyfit(ts, np.log(ds), 1)
print(slope)


plt.semilogy(t,d,'-')
plt.semilogy(ts,1e-8*np.exp(slope*ts),'--')
plt.show()



