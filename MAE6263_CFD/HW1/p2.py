#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:12:22 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

font = {'family' : 'Times New Roman',
        'size'   : 18}  

plt.rc('text', usetex=True)
  
plt.rc('font', **font)

import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
def plot(u,y,filename):
    fig, ax = plt.subplots(1,1,figsize=(10,6)) 
    for k in range(nt+1):
        if k%90 == 0:
            ax.plot(u[:,k],y, 'o-', lw=2, fillstyle='none', ms=8, 
                    label=f'$t = ${k*dt}')
            
    ax.legend()
    ax.set_ylabel('$y$')
    ax.set_xlabel('$u$')
    plt.show()    
    fig.savefig(filename, dpi=300)

#%%
def analytical(y,ymax,Ub,nu,t):
    eta_1 = ymax/(2.0*np.sqrt(nu*t))
    eta = y/(2.0*np.sqrt(nu*t))
    
    ua = np.zeros(y.shape[0])
    for n in range(5):
        ua = ua - Ub*(special.erf(2.0*n*eta_1 + eta) - 
                      special.erf(2.0*(n+1)*eta_1 - eta))
    
    return ua

#%%
nu = 0.000217

tmax = 1.08
dt = 0.002
nt = int(tmax/dt)

ymin = 0.0
ymax = 0.04
dy = (0.001)
ny = int((ymax-ymin)/dy)

Ub = 1.0

y = np.linspace(ymin, ymax, ny+1)

ua = np.zeros((ny+1,nt+1))
ua[0,0] = Ub

d = nu*dt/(dy**2)

for k in range(1,nt+1):
    tt = dt*k
    ua[:,k] = analytical(y,ymax,Ub,nu,tt)
    
plot(ua,y,f'analytical_{dt}.png')
    
#%% FTCS
u = np.zeros((ny+1,nt+1))
u[0,:] = Ub

for k in range(1,nt+1):
    u[1:-1,k] = u[1:-1,k-1] + d*(u[2:ny+1,k-1] - 2.0*u[1:-1,k-1] + u[0:-2,k-1])

plot(u,y,f'ftcs_{dt}.png')
er = (ua-u)*100    
plot(er,y,f'ftcs_error_{dt}.png')

#%% DuFort-Frankel scheme
u = np.zeros((ny+1,nt+1))
u[0,:] = Ub

for k in range(1,nt+1):
    u[1:-1,k] = ((1.0-2.0*d)*u[1:-1,k-2] + 
                 2.0*d*(u[2:ny+1,k-1] + u[0:-2,k-1]))/(1.0+2.0*d)

plot(u,y,f'DuFort_Frankel_{dt}.png')

#%% Laasonen method
u = np.zeros((ny+1,nt+1))
u[0,:] = Ub

a = np.zeros(ny+1)
b = np.zeros(ny+1)
c = np.zeros(ny+1)
r = np.zeros(ny+1)

i = 0
a[i] = 0.0
b[i] = 1.0
c[i] = 0.0
r[i] = Ub

i = ny
a[i] = 0.0    
b[i] = 1.0
c[i] = 0.0
r[i] = 0.0

for i in range(1,ny):
    a[i] = d
    b[i] = -(1.0 + 2.0*d)
    c[i] = d
    if i == 1:
        r[i] = -u[i,0] - a[i]*u[i-1,0]
    elif i == ny-1:
        r[i] = -u[i,0] - c[i]*u[i+1,0]
    else:
        r[i] = -u[i,0]

def tdms(a,b,c,r):
    n = a.shape[0]
    h = np.zeros(n)
    g = np.zeros(n)
    for i in range(1,n):
        h[i] = c[i]/()
        
        





