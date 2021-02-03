#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:12:22 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'Times New Roman',
        'size'   : 18}  

plt.rc('text', usetex=True)
  
plt.rc('font', **font)

import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

def ftcs()

#%%
nu = 0.000217

tmax = 1.08
dt = 0.002
nt = int(tmax/dt)

ny = 40
ymin = 0.0
ymax = 0.04
dy = (ymax-ymin)/ny

Ub = 40.0

y = np.linspace(ymin, ymax, ny+1)
u = np.zeros((ny+1,nt+1))

u[0,:] = Ub

d = nu*dt/(dy**2)

for k in range(1,nt+1):
    # for i in range(1,ny):
    #     u[i,k] = u[i,k-1] + d*(u[i+1,k-1] - 2.0*u[i,k-1] + u[i-1,k-1])
    u[1:-1,k] = u[1:-1,k-1] + d*(u[2:ny+1,k-1] - 2.0*u[1:-1,k-1] + u[0:-2,k-1])

#%%
fig, ax = plt.subplots(1,1,figsize=(10,6)) 
for k in range(nt+1):
    if k%90 == 0:
        ax.plot(u[:,k],y, 'o-', lw=2, label=f'$t = ${k*dt}')
        
ax.legend()
ax.set_ylabel('$y$')
ax.set_xlabel('$u$')
# ax.set_ylim([0.0,0.04])
plt.show()    