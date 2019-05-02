#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:49:05 2019

@author: Suraj Pawar
"""
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

n = 99
h = 0.01
x = np.linspace(0,1,n+2)
x = x.reshape(n+2,1)
control = 0.1

f = np.sin(np.pi*x)
theta_d = np.cos(np.pi*x)

# construct matrix Q
a = np.identity(n)
b = np.zeros((n,n))
c = control*np.identity(n)
q = np.hstack((a,b))
q = np.vstack((q,np.hstack((b,c))))

# construct matrix c = [a, b]
a = np.identity(n)
b = -1.0*np.identity(n)
a = -1.0*(2.0/(h*h) + 1.0/h)*a

for i in range(1,n-1):
    a[i,i-1] = 1.0/(h*h)
    a[i,i+1] = 1.0/(h*h)+1.0/h

a[0,1] = 1.0/(h*h)+1.0/h
a[n-1,n-2] = 1.0/(h*h)

c = np.hstack((a,b))

#analytical solution z = inv(q)*c'*(c*inv(q)*c')(f-A*theta_d)
qinv = inv(q)
k = np.dot(qinv,c.T)
l = np.dot(c, qinv)
l = np.dot(l, c.T)
m = f[1:n+1] - np.dot(a, theta_d[1:n+1])

z = np.dot(k,inv(l))
zstar = np.dot(z,m)

theta = np.zeros(n+2)
t = zstar[0:n] + theta_d[1:n+1]
t = t.reshape(n)
theta[1:n+1] = t # addd solution except for boundary points
u = zstar[n:]

plt.figure()
plt.plot(u)

plt.figure()
plt.plot(theta)