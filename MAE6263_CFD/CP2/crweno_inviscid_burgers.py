#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 12:29:08 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

def tdmsv(a,b,c,r,s,e,n):
    gam = np.zeros((e+1,n+1))
    u = np.zeros((e+1,n+1))
    bet = np.zeros((1,n+1))
    
    bet[0,:] = b[s,:]
    u[s,:] = r[s,:]/bet[0,:]
    
    for i in range(s+1,e+1):
        gam[i,:] = c[i-1,:]/bet[0,:]
        bet[0,:] = b[i,:] - a[i,:]*gam[i,:]
        u[i,:] = (r[i,:] - a[i,:]*u[i-1,:])/bet[0,:]
    
    for i in range(e-1,s-1,-1):
        u[i,:] = u[i,:] - gam[i+1,:]*u[i+1,:]
    
    return u
        
#-----------------------------------------------------------------------------#
# Solution to tridigonal system using cyclic Thomas algorithm
#-----------------------------------------------------------------------------#
def ctdmsv(a,b,c,alpha,beta,r,s,e,n):
    bb = np.zeros((e+1,n+1))
    u = np.zeros((e+1,n+1))
    gamma = np.zeros((1,n+1))
    
    gamma[0,:] = -b[s,:]
    bb[s,:] = b[s,:] - gamma[0,:]
    bb[e,:] = b[e,:] - alpha*beta/gamma[0,:]
    
#    for i in range(s+1,e):
#        bb[i] = b[i]
    
    bb[s+1:e,:] = b[s+1:e,:]
    
    x = tdmsv(a,bb,c,r,s,e,n)
    
    u[s,:] = gamma[0,:]
    u[e,:] = alpha[0,:]
    
    z = tdmsv(a,bb,c,u,s,e,n)
    
    fact = (x[s,:] + beta[0,:]*x[e,:]/gamma[0,:])/(1.0 + z[s,:] + beta[0,:]*z[e,:]/gamma[0,:])
    
#    for i in range(s,e+1):
#        x[i] = x[i] - fact*z[i]
    
    x[s:e+1,:] = x[s:e+1,:] - fact*z[s:e+1,:]
        
    return x


def c4dd_p(f,dx,dy,nx,ny,isign):
    
    if isign == 'XX':
        u = np.copy(f)
        h = dx
    if isign == 'YY':
        u = np.copy(f.T)
        h = dy
        temp = nx
        nx = ny
        ny = temp
    
    a = np.zeros((nx,ny+1))
    b = np.zeros((nx,ny+1))
    c = np.zeros((nx,ny+1))
    r = np.zeros((nx,ny+1))

    ii = np.arange(0,nx)
    up = u[ii,:]
    a[ii,:] = 1.0/10.0
    b[ii,:] = 1.0
    c[ii,:] = 1.0/10.0
    r[ii,:] = (6.0/5.0)*(up[ii-1,:] - 2.0*up[ii,:] + up[(ii+1)%nx,:])/(h*h)
    
    start = 0
    end = nx
        
    alpha = np.zeros((1,ny+1))
    beta = np.zeros((1,ny+1))
    
    alpha[0,:] = 1.0/10.0
    beta[0,:] = 1.0/10.0
    
    x = ctdmsv(a,b,c,alpha,beta,r,0,nx-1,ny)
    
    udd = np.zeros((nx+1,ny+1))
    udd[0:nx,:] = x[0:nx,:]
    udd[nx,:] = udd[0,:]
    
    if isign == 'XX':
        fdd = np.copy(udd)
    if isign == 'YY':
        fdd = np.copy(udd.T)
        
    return fdd

#-----------------------------------------------------------------------------#
# Calculate right hand term of the inviscid Burgers equation
# r = -u∂u/∂x
#-----------------------------------------------------------------------------#
def rhs(nx,dx,u,nu):
    uL = np.zeros(nx)
    uR = np.zeros(nx+1)
    r = np.zeros(nx+1)

    uL = crwenoL(nx,u,uL)
    uR = crwenoR(nx,u,uR)

    for i in range(1,nx):
        if (u[i] >= 0.0):
            r[i] = -u[i]*(uL[i] - uL[i-1])/dx
        else:
            r[i] = -u[i]*(uR[i+1] - uR[i])/dx

    #i = 0; periodic
    i = 0
    if (u[i] >= 0.0):
        r[i] = -u[i]*(uL[i] - uL[nx-1])/dx
    else:
        r[i] = -u[i]*(uR[i+1] - uR[nx])/dx
    
    ur = np.reshape(u,[-1,1])
    d2udx2 = c4dd_p(ur,dx,dx,nx,0,'XX')
    
    r = r + nu*d2udx2.flatten()
    
    return r

#-----------------------------------------------------------------------------#
# CRWENO reconstruction for upwind direction (positive; left to right)
# u(i): solution values at finite difference grid nodes i = 1,...,N+1
# f(j): reconstructed values at nodes j <== i+1/2; only use j = 1,2,...,N
#-----------------------------------------------------------------------------#
def crwenoL(n,u,f):
    a = np.zeros((n,1))
    b = np.zeros((n,1))
    c = np.zeros((n,1))
    r = np.zeros((n,1))

    i = 0
    v1 = u[n-2]
    v2 = u[n-1]
    v3 = u[i]
    v4 = u[i+1]
    v5 = u[i+2]

    a1,a2,a3,b1,b2,b3 = crwcL(v1,v2,v3,v4,v5)
    a[i,0] = a1
    b[i,0] = a2
    c[i,0] = a3
    r[i,0] = b1*u[n] + b2*u[i] + b3*u[i+1]

    i = 1
    v1 = u[n-1]
    v2 = u[i-1]
    v3 = u[i]
    v4 = u[i+1]
    v5 = u[i+2]

    a1,a2,a3,b1,b2,b3 = crwcL(v1,v2,v3,v4,v5)
    a[i,0] = a1
    b[i,0] = a2
    c[i,0] = a3
    r[i,0] = b1*u[i-1] + b2*u[i] + b3*u[i+1]

    for i in range(2,n-1):
        v1 = u[i-2]
        v2 = u[i-1]
        v3 = u[i]
        v4 = u[i+1]
        v5 = u[i+2]

        a1,a2,a3,b1,b2,b3 = crwcL(v1,v2,v3,v4,v5)
        a[i,0] = a1
        b[i,0] = a2
        c[i,0] = a3
        r[i,0] = b1*u[i-1] + b2*u[i] + b3*u[i+1]

    i = n-1
    v1 = u[i-2]
    v2 = u[i-1]
    v3 = u[i]
    v4 = u[i+1]
    v5 = u[1]

    a1,a2,a3,b1,b2,b3 = crwcL(v1,v2,v3,v4,v5)
    a[i,0] = a1
    b[i,0] = a2
    c[i,0] = a3
    r[i,0] = b1*u[i-1] + b2*u[i] + b3*u[i+1]

    alpha = c[n-1,:].reshape(1,-1)
    beta = a[0,:].reshape(1,-1)

    f = ctdmsv(a,b,c,alpha,beta,r,0,n-1,0)
    
    return f

#-----------------------------------------------------------------------------#
# CRWENO reconstruction for downwind direction (negative;right to left)
# u(i): solution values at finite difference grid nodes i =1,...,N+1
# f(j): reconstructed values at nodes j <== i-1/2; only use j = 2,...,N+1
#-----------------------------------------------------------------------------#
def crwenoR(n,u,f):
    a = np.zeros((n+1,1))
    b = np.zeros((n+1,1))
    c = np.zeros((n+1,1))
    r = np.zeros((n+1,1))

    i = 1
    v1 = u[n-1]
    v2 = u[i-1]
    v3 = u[i]
    v4 = u[i+1]
    v5 = u[i+2]

    a1,a2,a3,b1,b2,b3 = crwcR(v1,v2,v3,v4,v5)
    a[i,0] = a1
    b[i,0] = a2
    c[i,0] = a3
    r[i,0] = b1*u[i-1] + b2*u[i] + b3*u[i+1]

    for i in range(2,n-1):
        v1 = u[i-2]
        v2 = u[i-1]
        v3 = u[i]
        v4 = u[i+1]
        v5 = u[i+2]

        a1,a2,a3,b1,b2,b3 = crwcR(v1,v2,v3,v4,v5)
        a[i,0] = a1
        b[i,0] = a2
        c[i,0] = a3
        r[i,0] = b1*u[i-1] + b2*u[i] + b3*u[i+1]

    i = n-1
    v1 = u[i-2]
    v2 = u[i-1]
    v3 = u[i]
    v4 = u[i+1]
    v5 = u[1]

    a1,a2,a3,b1,b2,b3 = crwcR(v1,v2,v3,v4,v5)
    a[i,0] = a1
    b[i,0] = a2
    c[i,0] = a3
    r[i,0] = b1*u[i-1] + b2*u[i] + b3*u[i+1]

    i = n
    v1 = u[i-2]
    v2 = u[i-1]
    v3 = u[i]
    v4 = u[1]
    v5 = u[2]

    a1,a2,a3,b1,b2,b3 = crwcR(v1,v2,v3,v4,v5)
    a[i,0] = a1
    b[i,0] = a2
    c[i,0] = a3
    r[i,0] = b1*u[i-1] + b2*u[i] + b3*u[2]

    alpha = c[n,:].reshape(1,-1)
    beta = a[1,:].reshape(1,-1)

    f = ctdmsv(a,b,c,alpha,beta,r,1,n,0)
    
    return f

#---------------------------------------------------------------------------#
#nonlinear weights for upwind direction
#---------------------------------------------------------------------------#
def crwcL(v1,v2,v3,v4,v5):
    eps = 1.0e-6

    s1 = (13.0/12.0)*(v1-2.0*v2+v3)**2 + 0.25*(v1-4.0*v2+3.0*v3)**2
    s2 = (13.0/12.0)*(v2-2.0*v3+v4)**2 + 0.25*(v2-v4)**2
    s3 = (13.0/12.0)*(v3-2.0*v4+v5)**2 + 0.25*(3.0*v3-4.0*v4+v5)**2

    c1 = 2.0e-1/((eps+s1)**2)
    c2 = 5.0e-1/((eps+s2)**2)
    c3 = 3.0e-1/((eps+s3)**2)

    w1 = c1/(c1+c2+c3)
    w2 = c2/(c1+c2+c3)
    w3 = c3/(c1+c2+c3)

    a1 = (2.0*w1 + w2)/3.0
    a2 = (w1 + 2.0*w2 + 2.0*w3)/3.0
    a3 = w3/3.0

    b1 = w1/6.0
    b2 = (5.0*w1 + 5.0*w2 + w3)/6.0
    b3 = (w2 + 5.0*w3)/6.0

    return a1,a2,a3,b1,b2,b3

#---------------------------------------------------------------------------#
#nonlinear weights for downwind direction
#---------------------------------------------------------------------------#
def crwcR(v1,v2,v3,v4,v5):
    eps = 1.0e-6

    s1 = (13.0/12.0)*(v1-2.0*v2+v3)**2 + 0.25*(v1-4.0*v2+3.0*v3)**2
    s2 = (13.0/12.0)*(v2-2.0*v3+v4)**2 + 0.25*(v2-v4)**2
    s3 = (13.0/12.0)*(v3-2.0*v4+v5)**2 + 0.25*(3.0*v3-4.0*v4+v5)**2

    c1 = 3.0e-1/(eps+s1)**2
    c2 = 5.0e-1/(eps+s2)**2
    c3 = 2.0e-1/(eps+s3)**2

    w1 = c1/(c1+c2+c3)
    w2 = c2/(c1+c2+c3)
    w3 = c3/(c1+c2+c3)

    a1 = w1/3.0
    a2 = (w3 + 2.0*w2 + 2.0*w1)/3.0
    a3 = (2.0*w3 + w2)/3.0

    b1 = (w2 + 5.0*w1)/6.0
    b2 = (5.0*w3 + 5.0*w2 + w1)/6.0
    b3 = w3/6.0

    return a1,a2,a3,b1,b2,b3

#%%
nx = 200
ns = 10
dt = 0.0001
tm = 0.25
nu = 1.0e-1

lx = 1.0
dx = lx/nx
nt = int(tm/dt)

freq = int(nt/ns)

u = np.zeros((nx+1, ns+1))
ut = np.zeros(nx+1)

x = np.linspace(0,lx,nx+1)
un = np.sin(2.0*np.pi*x)

k = 0
u[:,k] = un

for j in range(1,nt+1):
    r = rhs(nx, dx, un, nu)
    
    ut[:nx] = un[:nx] + dt*r[:nx]
    ut[nx] = ut[0]
    
    r = rhs(nx, dx, ut, nu)
    ut[:nx] = 0.75*un[:nx] + 0.25*ut[:nx] + 0.25*dt*r[:nx]
    ut[nx] = ut[0]
    
    r = rhs(nx, dx, ut, nu)
    un[:nx] = (1.0/3.0)*un[:nx] + (2.0/3.0)*ut[:nx] + (2.0/3.0)*dt*r[:nx]
    un[nx] = un[0]
    
    if j%freq == 0:
        k = k + 1
        print(k)
        u[:,k] = un
        
#%%
fig, ax = plt.subplots(1,1, figsize=(8,6))

for i in range(5):
    ax.plot(x,u[:,2*i],label=f'k = {2*i}')

ax.legend()
plt.show()  
    
    