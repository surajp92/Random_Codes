#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:36:20 2021

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
    fig, ax = plt.subplots(1,1,figsize=(6,5)) 
    for k in range(4):
        ax.plot(u[:,k],y, 'o-', lw=2, fillstyle='none', ms=8, 
                    label=f'$t = ${dt_list[k]}')
            
    ax.legend()
    ax.set_ylabel('$y$')
    ax.set_xlabel('Error')
    plt.show()    
    fig.tight_layout()
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

#%% Laasonen method
def tdms(a,b,c,r):
    b_ = np.copy(b)    
    
    n = a.shape[0]
    x = np.zeros(n)
    for i in range(1,n):
        w = a[i]/b_[i-1]
        b_[i] = b_[i] - w*c[i-1]
        r[i] = r[i] - w*r[i-1]
    
    x[i] = r[i]/b_[i]
    for i in range(n-2,-1,-1):
        x[i] = (r[i] - c[i]*x[i+1])/b_[i]
    
    return x

#%%

ymin = 0.0
ymax = 0.04
dy = (0.001)
ny = int((ymax-ymin)/dy)
tmax = 1.00
nu = 0.000217

u_list = np.zeros((ny+1,4))    
u_error_list = np.zeros((ny+1,4))    
dt_list = np.array([0.005,0.01,0.1,0.2])

for j in range(4):
    dt = dt_list[j]
    nt = int(tmax/dt)
    Ub = 1.0
    
    y = np.linspace(ymin, ymax, ny+1)
    
    ua = np.zeros((ny+1,nt+1))
    ua[0,0] = Ub
    
    d = nu*dt/(dy**2)
    
    for k in range(1,nt+1):
        tt = dt*k
        ua[:,k] = analytical(y,ymax,Ub,nu,tt)
    
    beta = 1.0 # Laasonen method
    
    u = np.zeros((ny+1,nt+1))
    u[0,:] = Ub
    
    a = np.zeros(ny-1)
    b = np.zeros(ny-1)
    c = np.zeros(ny-1)
    r = np.zeros(ny-1)
    
    for i in range(ny-1):
        a[i] = d*beta
        b[i] = -(1.0 + 2.0*d*beta)
        c[i] = d*beta
        
    for k in range(1,nt+1):
        r[:] = -u[1:-1,k-1] - (1.0 -beta)*d*(u[2:ny+1,k-1] - 2.0*u[1:-1,k-1] + u[0:-2,k-1])
        r[0] = r[0] - a[0]*Ub
        r[ny-2] = r[ny-2] 
        
        u[1:-1,k] = tdms(a,b,c,r)
    
    u_list[:,j] = u[:,-1]
    u_error_list[1:-1,j] = (ua[1:-1,-1] - u[1:-1,-1])*100 #/ua[1:-1,-1]
    
plot(u_error_list,y,f'laasonen_error.png')     