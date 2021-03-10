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

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

tmax = 30
dt = 0.001

def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

x0 = [1.0, 2.0, 3.0]
t = np.arange(11)
x = odeint(f, x0, t)

epsilon_ = 1.0e-8

xn1 = np.copy(x[-1,:])
xn2 = np.copy(x[-1,:])
xn2[2] = xn1[2] + epsilon_

#%%
nt = int(tmax/dt)
t = np.linspace(0,tmax,nt+1)

x1 = odeint(f, xn1, t)
x2 = odeint(f, xn2, t)

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(x1[:, 0], x1[:, 1], x1[:, 2])
ax.plot(x2[:, 0], x2[:, 1], x2[:, 2])
plt.draw()
plt.show()

#%%
d = np.linalg.norm(x1-x2, axis=1)

# plt.plot(d,'-')
plt.semilogy(d,'--')
plt.show()

#%%
logd = np.log(d[t<20][1:])
logt = np.log(t[t<20][1:])
logd2 = np.polyfit(logt, logd, 1)
slope = logd2[1] - logd2[0]
poly1d_fn = np.poly1d(logd2) 

plt.semilogy(t,d)
# plt.semilogy(logd,'--')
plt.semilogy(logt,poly1d_fn(t[t<20][1:]))
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

x = [1,2,3,4]
y = [3,5,7,10] # 10, not 9, so the fit isn't perfect

coef = np.polyfit(x,y,1)
poly1d_fn = np.poly1d(coef) 
# poly1d_fn is now a function which takes in x and returns an estimate for y

plt.plot(x,y, 'yo', x, poly1d_fn(x), '--k')
plt.xlim(0, 5)
plt.ylim(0, 12)
