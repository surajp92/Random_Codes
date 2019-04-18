#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:26:40 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

nx =  1024 # spatial grid number
re_start = 100.0
re_final = 1000.0
n_re = 10
tmax = 1.0
lx = 1.0
n_snap = 50 #number of snapshot per each Re 
n_s = (n_snap+1)*n_re #total number of snapshots in paramatric-space

dx = lx/nx
dt_snap = tmax/n_snap

def rhs(n_modes, b_c, b_l, b_nl, a):
    r2 = np.zeros(n_modes)
    r3 = np.zeros(n_modes)
    r = np.zeros(n_modes)
    
    for k in range(n_modes):
        r2[k] = 0.0
        for i in range(n_modes):
            r2[k] = r2[k] + b_l[i,k]*a[i]
    
    for k in range(n_modes):
        r3[k] = 0.0
        for j in range(n_modes):
            for i in range(n_modes):
                r3[k] = r3[k] + b_nl[i,j,k]*a[i]*a[j]
    
    r = b_c + r2 + r3    
    return r
    
    
def rk3(n_modes, b_c, b_l, b_nl, a, dt):
    r = np.zeros(n_modes)
    a1 = np.zeros(n_modes)
    
    #1.stage
    r = rhs(n_modes, b_c, b_l, b_nl, a)
    a1 = a + dt*r
    #2. stage
    r = rhs(n_modes, b_c, b_l, b_nl, a1)
    a1 = 0.75*a + 0.25*a1 + 0.25*dt*r
    #3. stage
    r = rhs(n_modes, b_c, b_l, b_nl, a1)
    a = (1.0/3.0)*a + (2.0/3.0)*a1 + (2.0/3.0)*dt*r
    

def tdma(a, b, c, r, up, s, e):
    for i in range(s+1,e+1):
        b[i] = b[i] - a[i]/b[i-1]*c[i-1]
        r[i] = r[i] - a[i]/b[i-1]*r[i-1]
    
    up[e] = r[e]/b[e]
    
    for i in range(e-1,s-1,-1):
        up[i] = (r[i]-c[i]*up[i+1])/b[i]
    
    

def pade4d(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    ud = np.zeros(n+1)
    
    i = 0
    b[i] = 1.0
    c[i] = 2.0
    r[i] = (-5.0*u[i] + 4.0*u[i+1] + u[i+2])/(2.0*h)
    
    for i in range(1,n):
        a[i] = 1.0
        b[i] = 4.0
        c[i] = 1.0
        r[i] = 3.0*(u[i+1] - u[i-1])/h
    
    i = n
    a[i] = 2.0
    b[i] = 1.0
    r[i] = (-5.0*u[i] + 4.0*u[i-1] + u[i-2])/(-2.0*h)
    
    tdma(a, b, c, r, ud, 0, n)
    return ud
    
def pade4dd(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    udd = np.zeros(n+1)
    
    i = 0
    b[i] = 1.0
    c[i] = 11.0
    r[i] = (13.0*u[i] - 27.0*u[i+1] + 15.0*u[i+2] - u[i+3])/(h*h)
    
    for i in range(1,n):
        a[i] = 0.1
        b[i] = 1.0
        c[i] = 0.1
        r[i] = 1.2*(u[i+1] - 2.0*u[i] + u[i-1])/(h*h)
    
    i = n
    a[i] = 11.0
    b[i] = 1.0
    r[i] = (13.0*u[i] - 27.0*u[i-1] + 15.0*u[i-2] - u[i-3])/(h*h)
    
    tdma(a, b, c, r, udd, 0, n)
    return udd


def uexact(x, t, nu):
    temp = np.exp(1.0/(8.0*nu))
    uexact = (x/(t+1.0))/(1.0+np.sqrt((t+1.0)/temp)*np.exp(x*x/(4.0*nu*(t+1.0))))
    return uexact

# simpsons 1/3rd rule. nx should be even number
def simp1d(nx,dx,g):
    
    nh = int(nx/2)
    th = dx/3.0
    
    s= 0.0
    for i in range(0,nh):
        s = s + th*(g[2*i]+4.0*g[2*i+1]+g[2*i+2])
    return s

#generate snapshot data from exact solution
x = np.zeros(nx+1)
t = np.zeros(n_snap+1)
nu = np.zeros(n_re)
ue = np.zeros((nx+1,n_snap+1,n_re))
for p in range(0,n_re):
    re = re_start + p*(re_final-re_start)/(n_re-1)
    nu[p] = 1.0/re
    for n in range(0,n_snap+1):
        t[n] = dt_snap*n
        for i in range(nx+1):
            x[i]=dx*i
            ue[i,n,p]=uexact(x[i],t[n],nu[p])

#generate snapshot data in paramatric-space
us = np.zeros((nx+1,n_s))
for p in range(0,n_re):
    for n in range(0,n_snap+1):
        for i in range(nx+1):
            us[i,((n_snap+1)*p)+n] = ue[i,n,p]           

#compute mean of snapshots
um = np.zeros(nx+1) 
for n in range(0,n_s):
    um[:] = um[:] + us[:,n] 
um[:]=um[:]/(n_s)

#compute anomaly(fluctuating) part of snapshots
uf = np.zeros((nx+1,n_s))
for n in range(0,n_s):
    for i in range(nx+1):
        uf[i,n] = us[i,n]-um[i]


#generate snapshots time correlation matrix
#c = np.zeros((n_s,n_s))   
#nh = int(nx/2)
#th = dx/3.0     
#for l in range(n_s):
#    for k in range(n_s):
#        for i in range(nh):
#            c[k,l] += th*(uf[2*i,k]*uf[2*i,l]+4.0*uf[2*i+1,k]*uf[2*i+1,l]+uf[2*i+2,k]*uf[2*i+2,l])
            
c = np.transpose(uf).dot(uf)/nx

#solve eigen system
w, v = LA.eig(c)      
w = np.real(w)
v = np.real(v)

n_modes = 5
phi = np.zeros((nx+1,n_modes))
v_reduced = v[:,0:n_modes]
phi = uf.dot(v_reduced)

for k in range(n_modes):
    phi[:,k] = phi[:,k]/np.sqrt(abs(w[k]))

# galerkin projection
b_c = np.zeros(n_modes)
b_l = np.zeros((n_modes,n_modes))
b_nl = np.zeros((n_modes,n_modes, n_modes))
phid = np.zeros((nx+1,n_modes))
phidd = np.zeros((nx+1,n_modes))

re_test = 750.0
nu_test = 1.0/re_test
umdd = np.zeros(nx+1)
umd = np.zeros(nx+1)
umdd = pade4dd(um,dx,nx)
umd = pade4d(um,dx,nx)

for i in range(n_modes):
    phidd[:,i] = pade4dd(phi[:,i],dx,nx)
    phid[:,i] = pade4d(phi[:,i],dx,nx)

#constant term:
for k in range(n_modes):
    b_c[k]= nu_test*np.transpose(phi[:,k]).dot(umdd)/nx - np.transpose(phi[:,k]).dot(um*umd)/nx  
    
for k in range(n_modes):
    for i in range(n_modes):
        b_l[i,k]= nu_test*np.transpose(phi[:,k]).dot(phidd[:,i])/nx - \
                  np.transpose(phi[:,k]).dot(um*phid[:,i] + phi[:,i]*umd)/nx    

for k in range(n_modes):
    for j in range(n_modes):
        for i in range(n_modes):
            b_nl[i,j,k]= - np.transpose(phi[:,k]).dot(phi[:,i]*phid[:,j])/nx   
                  
# galerkin-rom
nt = 1000
dt = tmax/nt
a = np.zeros((n_modes,nt+1))
at = np.zeros((n_modes,nt+1))
ut = np.zeros((nx+1,nt+1))
r = np.zeros(n_modes)
a1 = np.zeros(n_modes)
t = np.linspace(0,tmax,nt+1)    
#initial condition:
for n in range(nt+1):
    for i in range(nx+1):
        ut[i,n]=uexact(x[i],t[n],nu_test)-um[i]

at = np.transpose(phi).dot(ut)/nx
a[:,0]=at[:,0]

for k in range(1,nt+1):
    #1.stage
    r = rhs(n_modes, b_c, b_l, b_nl, a[:,k-1])
    a1 = a[:,k-1] + dt*r
    #2. stage
    r = rhs(n_modes, b_c, b_l, b_nl, a1)
    a1 = 0.75*a[:,k-1] + 0.25*a1 + 0.25*dt*r
    #3. stage
    r = rhs(n_modes, b_c, b_l, b_nl, a1)
    a[:,k] = (1.0/3.0)*a[:,k-1] + (2.0/3.0)*a1 + (2.0/3.0)*dt*r

plt.figure()
for i in range(5):    
    plt.plot(t,a[i,:],'b-')
    plt.plot(t,at[i,:],'r-')
    
delta = np.transpose(phi).dot(phi)/nx

plt.figure()
plt.plot(x,phi[:,9],'b-')
  
plt.figure()
plt.plot(w,'b-')
plt.yscale('log')
plt.xscale('log')
plt.xlim([0,100])

#compute RIC (relative inportance index)
ric = np.zeros(n_s)
for k in range(n_s):
    temp = 0.0
    for i in range(k+1):
        temp = temp + w[i]
    ric[k] = temp/sum(w)*100
    
plt.figure()
plt.plot(ric,'b-')
plt.xlim([0,10])
   
plt.figure()
for n in range(n_snap+1):
    plt.plot(x, uf[:,n], 'b-')
    
    
            
plt.figure()
plt.plot(x,um,'b-')

plt.figure()
for n in range(n_snap+1):
    plt.plot(x, ue[:,n,3], 'b-')
    



    
plt.figure()
for n in range(n_snap+1):
    plt.plot(x, uf[:,n], 'b-')

    
plt.figure()
plt.plot(x,um,'b-')



    
n = 64
x_left = 2.0
x_right = 6.0

h = (x_right - x_left)/n

x = np.zeros(n+1)
u = np.zeros(n+1)
upe = np.zeros(n+1)
uppe = np.zeros(n+1)

for j in range(0, n+1):
    x[j] = x_left + j*h
    u[j] = np.sin(x[j])
    upe[j] = np.cos(x[j])  
    uppe[j] = -np.sin(x[j])    
    
ud4 = np.zeros(n+1)
udd4 = np.zeros(n+1)

ud4 = pade4d(u, h, n)
udd4 = pade4dd(u, h, n)

plt.figure()
plt.plot(x, ud4, 'bo')
plt.plot(x, upe, 'r-')

plt.figure()
plt.plot(x, udd4, 'bo')
plt.plot(x, uppe, 'r-')