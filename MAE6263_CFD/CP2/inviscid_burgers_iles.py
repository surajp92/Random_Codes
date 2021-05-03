#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 12:29:08 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

def initialize(array):  # Initializes in fourier domain
    global nx, nr
    pi = np.pi

    dx = xlength / float(nx)
    dx1 = dx / float(nr)

    nx_new = nx * nr
    xlength_new = dx1 * float(nx_new)
    
    # Note no im heref
    kx = np.array([(2 * pi) * i / xlength_new for i in
                   list(range(0, int(nx_new / 2))) + [0] + list(range(-int(nx_new / 2) + 1, 0))])  
    
    acons = 2.0 / (10.0 ** (5.0)) / (3.0 * (pi ** (0.5)))

    array_hat = np.zeros(nx_new, dtype=np.complex)  # The array is of type complex
    phase = np.zeros(2 * nx_new, dtype='double')

    # np.random.seed(0)
    rand = np.random.uniform(0.0, 1.0)
    phase[0] = np.cos(2.0 * pi * rand)
    phase[1] = 0.0
    phase[nx_new] = np.cos(2.0 * pi * rand)
    phase[nx_new + 1] = 0.0

    k = 3
    for i in range(1, int(nx_new / 2)):
        rand = np.random.uniform(0.0, 1.0)
        phase[k - 1] = np.cos(2.0 * pi * rand)
        phase[k] = np.sin(2.0 * pi * rand)
        phase[2 * nx_new - k + 1] = np.cos(2.0 * pi * rand)
        phase[2 * nx_new - k + 2] = -np.sin(2.0 * pi * rand)
        k = k + 2

    k = 0
    for i in range(0, nx_new):
        espec_ip = np.exp(-(kx[i] / 10.0) ** (2.0))
        espec = acons * (kx[i] ** 4) * espec_ip
        array_hat[i] = nx_new * (np.sqrt(2.0 * espec) * (phase[k] + phase[k + 1]))
        k = k + 2

    temp_array = np.real(np.fft.ifft(array_hat))
    copy_array = np.zeros(nx+1, dtype='double')

    for i in range(0, nx):
        copy_array[i] = temp_array[i * nr]

    # np.copyto(array, np.fft.fft(copy_array))
    
    copy_array[nx] = copy_array[0]
    
    np.copyto(array, copy_array)
    
    del temp_array, copy_array

def spectra_calculation(array_hat):
    # Normalizing data
    global nx
    array_new = np.copy(array_hat / float(nx))
    # Energy Spectrum
    espec = 0.5 * np.absolute(array_new)**2
    # Angle Averaging
    eplot = np.zeros(int(nx / 2), dtype='double')
    for i in range(1, int(nx / 2)):
        eplot[i] = 0.5 * (espec[i] + espec[nx - i])

    return eplot


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
def rhs(nx,dx,u,nu,iles):
    uL = np.zeros(nx)
    uR = np.zeros(nx+1)
    r = np.zeros(nx+1)
    
    if iles == 1:
        uL = wenoL(nx,u,uL)
        uR = wenoR(nx,u,uR)
    elif iles == 2:
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

#-----------------------------------------------------------------------------#
# CRWENO reconstruction for upwind direction (positive; left to right)
# u(i): solution values at finite difference grid nodes i = 1,...,N+1
# f(j): reconstructed values at nodes j <== i+1/2; only use j = 1,2,...,N
#-----------------------------------------------------------------------------#
def wenoL(n,u,f):
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    r = np.zeros(n)
    f = np.zeros(n)

    i = 0
    v1 = u[n-2]
    v2 = u[n-1]
    v3 = u[i]
    v4 = u[i+1]
    v5 = u[i+2]
    f[i] = wcL(v1,v2,v3,v4,v5)

    i = 1
    v1 = u[n-1]
    v2 = u[i-1]
    v3 = u[i]
    v4 = u[i+1]
    v5 = u[i+2]
    f[i] = wcL(v1,v2,v3,v4,v5)

    for i in range(2,n-1):
        v1 = u[i-2]
        v2 = u[i-1]
        v3 = u[i]
        v4 = u[i+1]
        v5 = u[i+2]
        f[i] = wcL(v1,v2,v3,v4,v5)
    
    i = n-1
    v1 = u[i-2]
    v2 = u[i-1]
    v3 = u[i]
    v4 = u[i+1]
    v5 = u[1]
    f[i] = wcL(v1,v2,v3,v4,v5)
    
    return f

#-----------------------------------------------------------------------------#
# CRWENO reconstruction for downwind direction (negative;right to left)
# u(i): solution values at finite difference grid nodes i =1,...,N+1
# f(j): reconstructed values at nodes j <== i-1/2; only use j = 2,...,N+1
#-----------------------------------------------------------------------------#
def wenoR(n,u,f):
    a = np.zeros(n+1)
    b = np.zeros(n+1)
    c = np.zeros(n+1)
    r = np.zeros(n+1)
    f = np.zeros(n+1)

    i = 1
    v1 = u[n-1]
    v2 = u[i-1]
    v3 = u[i]
    v4 = u[i+1]
    v5 = u[i+2]
    f[i] = wcR(v1,v2,v3,v4,v5)
    
    for i in range(2,n-1):
        v1 = u[i-2]
        v2 = u[i-1]
        v3 = u[i]
        v4 = u[i+1]
        v5 = u[i+2]
        f[i] = wcR(v1,v2,v3,v4,v5)
    
    i = n-1
    v1 = u[i-2]
    v2 = u[i-1]
    v3 = u[i]
    v4 = u[i+1]
    v5 = u[1]
    f[i] = wcR(v1,v2,v3,v4,v5)

    i = n
    v1 = u[i-2]
    v2 = u[i-1]
    v3 = u[i]
    v4 = u[1]
    v5 = u[2]
    f[i] = wcR(v1,v2,v3,v4,v5)
    
    return f

#---------------------------------------------------------------------------#
#nonlinear weights for upwind direction
#---------------------------------------------------------------------------#
def wcL(v1,v2,v3,v4,v5):
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

    # candiate stencils
    q1 = v1/3.0 - 7.0/6.0*v2 + 11.0/6.0*v3
    q2 =-v2/6.0 + 5.0/6.0*v3 + v4/3.0
    q3 = v3/3.0 + 5.0/6.0*v4 - v5/6.0

    # reconstructed value at interface
    f = (w1*q1 + w2*q2 + w3*q3)

    return f

#---------------------------------------------------------------------------#
#nonlinear weights for downwind direction
#---------------------------------------------------------------------------#
def wcR(v1,v2,v3,v4,v5):
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

    # candiate stencils
    q1 =-v1/6.0      + 5.0/6.0*v2 + v3/3.0
    q2 = v2/3.0      + 5.0/6.0*v3 - v4/6.0
    q3 = 11.0/6.0*v3 - 7.0/6.0*v4 + v5/3.0

    # reconstructed value at interface
    f = (w1*q1 + w2*q2 + w3*q3)

    return f

#%%
isp = 1  # [0] Sine, [1] exp spectrum 
nse = 16  # number of ensembles
iles = 2 # [1] WENO5, [2] CRWENO5

nx = 1024
nr = 1 #int(512 / nx)

ns = 10
dt = 1e-4
tm = 0.25
nu = 5.0e-4

# xlength = 1.0
xlength = 2.0 * np.pi
dx = xlength/nx
nt = int(tm/dt)

freq = int(nt/ns)

u = np.zeros((nse,nx+1, ns+1))
ut = np.zeros(nx+1)

x = np.linspace(0,xlength,nx+1)

for ne in range(nse):
    if isp == 0:
        un = np.sin(x)
        # un = np.sin(2.0*np.pi*x)
    elif isp == 1:
        un = np.zeros(nx+1)  # The array is of type complex
        initialize(un)
    
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(x,un)
    plt.show()
    
    k = 0
    u[ne,:,k] = un
    
    for j in range(1,nt+1):
        r = rhs(nx, dx, un, nu, iles)
        
        ut[:nx] = un[:nx] + dt*r[:nx]
        ut[nx] = ut[0]
        
        r = rhs(nx, dx, ut, nu, iles)
        ut[:nx] = 0.75*un[:nx] + 0.25*ut[:nx] + 0.25*dt*r[:nx]
        ut[nx] = ut[0]
        
        r = rhs(nx, dx, ut, nu, iles)
        un[:nx] = (1.0/3.0)*un[:nx] + (2.0/3.0)*ut[:nx] + (2.0/3.0)*dt*r[:nx]
        un[nx] = un[0]
        
        if j%freq == 0:
            k = k + 1
            print(k)
            u[ne,:,k] = un
        
#%%
fig, ax = plt.subplots(1,1, figsize=(8,6))

for i in range(5):
    ax.plot(x,u[0,:,2*i],label=f'k = {2*i}')

ax.legend()
plt.show()  


#%%
fig, ax = plt.subplots(1,1, figsize=(8,6))

for i in range(4,5):
    for ne in range(nse):    
        uf = np.fft.fft(u[ne,:,2*i])
        eplot = spectra_calculation(uf)
        if ne == 0:
            eplot1 = eplot
        else:
            eplot1 = eplot1 + eplot
        # ax.loglog(eplot,label=f'n = {ne}')
        
    eplot1 = eplot1/nse
    kw = np.linspace(1,eplot1.shape[0]-1,eplot1.shape[0]-1)
    ax.loglog(kw,eplot1[1:],'b-', lw=2, label=f'k = {2*i}')

kstart = 10
kend = 120

ks = np.linspace(kstart,kend,kend-kstart+1)
escaling = 1e0*ks**(-2)

ax.loglog(ks,escaling,'k--', lw=2, label=f'$k^{-2}$')

ax.set_ylim([1e-6,1e-1])
ax.set_xlim([1,kw[-1]+100])

ax.legend()
plt.show()  

np.savez(f'data_{iles}_{isp}_{nse}_{nx}.npz', 
         x = x, u = u, nx = nx, 
         nse = nse)
    
    