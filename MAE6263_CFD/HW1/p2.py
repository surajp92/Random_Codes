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
    fig, ax = plt.subplots(1,1,figsize=(6,5)) 
    for k in range(nt+1):
        if k%90 == 0:
            ax.plot(u[:,k],y, 'o-', lw=2, fillstyle='none', ms=8, 
                    label=f'$t = ${k*dt}')
            
    ax.legend()
    ax.set_ylabel('$y$')
    ax.set_xlabel('$u$')
    plt.show()    
    fig.tight_layout()
    fig.savefig(filename, dpi=300)

def plot_error(error,y,filename):
    label = ['FTCS', 'DF', 'Laasonen', 'CN']
    fig, ax = plt.subplots(1,1,figsize=(6,5)) 
    k = 0
    for er in error:
        ax.plot(er,y, 'o-', lw=2, fillstyle='none', ms=8, 
                    label=label[k])
        k = k + 1
            
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

#%%
nu = 0.000217

tmax = 1.08
dt = 0.00232
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
er_ftcs = (ua-u)*100    
u_ftcs = np.copy(u)
plot(er_ftcs,y,f'ftcs_error_{dt}.png')
    
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
er_ftcs = (ua-u)*100    
u_ftcs = np.copy(u)
plot(er_ftcs,y,f'ftcs_error_{dt}.png')

#%% DuFort-Frankel scheme
u = np.zeros((ny+1,nt+1))
u[0,:] = Ub

for k in range(1,nt+1):
    u[1:-1,k] = ((1.0-2.0*d)*u[1:-1,k-2] + 
                 2.0*d*(u[2:ny+1,k-1] + u[0:-2,k-1]))/(1.0+2.0*d)

plot(u,y,f'DuFort_Frankel_{dt}.png')
er_df = (ua-u)*100    
u_df = np.copy(u)
plot(er_df,y,f'DuFort_Frankel_error_{dt}.png')

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
    
plot(u,y,f'Laasonen_{dt}.png')   
er_laasonen = (ua-u)*100    
u_laasonen = np.copy(u)
plot(er_laasonen,y,f'laasonen_error_{dt}.png') 

#%%
beta = 0.5 # Crank-Nicolson method

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
    
plot(u,y,f'CN_{dt}.png')   
er_cn = (ua-u)*100    
u_cn= np.copy(u)
plot(er_cn,y,f'CN_error_{dt}.png') 
    
#%%
error = [er_ftcs[:,90], er_df[:,90], er_laasonen[:,90], er_cn[:,90]]
plot_error(error,y,'error_0.18.png')

error = [er_ftcs[:,-1], er_df[:,-1], er_laasonen[:,-1], er_cn[:,-1]]
plot_error(error,y,'error_1.08.png')


