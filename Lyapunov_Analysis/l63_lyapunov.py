#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 17:19:19 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import orth

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

tmax = 10
dt = 0.001

def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

x0 = [1.0, 2.0, 3.0]
t = np.arange(11)
x = odeint(f, x0, t)

xn = x[-1,:]

nt = int(tmax/dt)
t = np.linspace(0,tmax,nt+1)
x = odeint(f, xn, t)


fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(x[:, 0], x[:, 1], x[:, 2])
plt.draw()
plt.show()

#%%
e1, e2, e3 = 0.0, 0.0, 0.0
Jn = np.eye(3)
w = np.eye(3)
J = np.eye(3)

for k in range(nt+1):
    x1, x2, x3 = x[k,0], x[k,1], x[k,2]
    Df = np.array([[-sigma, sigma, 0],[-x3+rho,-1,-x1],[x2,x1,-beta]])
    J = np.eye(3) + Df*dt       
    w = J @ w
    
    e1 = e1 + np.log(np.linalg.norm(w[:,0]))
    e2 = e2 + np.log(np.linalg.norm(w[:,1]))
    e3 = e3 + np.log(np.linalg.norm(w[:,2]))
    
    w = orth(w)
    
    w[:,0] = w[:,0]/np.linalg.norm(w[:,0])
    w[:,1] = w[:,1]/np.linalg.norm(w[:,1])
    w[:,2] = w[:,2]/np.linalg.norm(w[:,2])
    
e1 = e1/tmax
e2 = e2/tmax
e3 = e3/tmax

l1 = np.exp(e1)
l2 = np.exp(e2)
l3 = np.exp(e3)

print(f'[{e1} {e2} {e3}]')
print(f'trace = {e1+e2+e3}')
print(f'[{l1} {l2} {l3}]')