#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:49:05 2019

@author: Suraj Pawar
"""
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy import linalg as LA

#%% 
n = 99
h = 0.01
xx = np.linspace(0,1,n+2)
xx = xx.reshape(n+2,1)
control = 0.0001

#%%
f = np.sin(np.pi*xx)
theta_d = np.cos(np.pi*xx)

# construct matrix Q
a = np.identity(n)
b = np.zeros((n,n))
c = control*np.identity(n)
q = np.hstack((a,b))
q = np.vstack((q,np.hstack((b,c))))

#%%
# construct matrix c = [a, b]
a = -1.0*(2.0/(h*h) + 1.0/h)*np.identity(n)
b = -1.0*np.identity(n)

for i in range(1,n-1):
    a[i,i-1] = 1.0/(h*h)
    a[i,i+1] = 1.0/(h*h)+1.0/h

a[0,1] = 1.0/(h*h)+1.0/h
a[n-1,n-2] = 1.0/(h*h)

c = np.hstack((a,b))

#%%
#analytical solution z = inv(q)*c'*(c*inv(q)*c')(f-A*theta_d)
qinv = inv(q)
k = np.dot(qinv,c.T)
l = np.dot(np.dot(c, qinv), c.T)
m = f[1:n+1] - np.dot(a, theta_d[1:n+1])

xstar = np.dot(np.dot(k,inv(l)),m)

theta_a = np.zeros(n+2)
theta_hat = xstar[0:n]
u_a = xstar[n:]

t = theta_hat + theta_d[1:n+1]
t = t.reshape(n)
theta_a[1:n+1] = t # addd solution except for boundary points
theta_a = theta_a.reshape(n+2,1)

#%%
fig, ax1 = plt.subplots()
ax1.plot(u_a, label ='U', color = 'blue', linewidth=3, marker='o')
plt.xlim(0,100)
ax1.legend()

fig, ax2 = plt.subplots()
ax2.plot(theta_a, label ='Theta', color = 'red', linewidth=3, marker='o')
plt.xlim(0,100)
ax2.legend()

#%%
plt.figure()
plt.plot(theta_d)
plt.plot(theta_hat)
plt.plot(theta_a)
#%%
x_p = 10*np.ones(2*n)
x_p= x_p.reshape(2*n,1)
alpha = 0.0001
beta = 0.00001
gamma_p = 10*np.ones(n)
gamma_p = gamma_p.reshape(n,1)
x_n = x_p.copy
gamma_n = gamma_p
rms = 10.0
k = 0
iteration_history = np.empty(shape=[0, 2])
#%%
#while k<60:
while rms>1e-6: 
    x_n = x_p - alpha*(np.dot(q, x_p) - np.dot(c.T, gamma_p))
    gamma_p = gamma_p + beta*(m - np.dot(c,x_n))
    rms = LA.norm(x_n - x_p)
#    print(rms)
    k = k+1
    x_p = x_n
#    iteration_history = np.append(iteration_history, [[k, rms]], axis=0)
#    if k>50:
#        grad_f = np.dot(q, x_n)
#        alpha = (np.dot(grad_f.T, grad_f))/(np.dot(grad_f.T, np.dot(q, grad_f)))
#        print(k, " ", alpha)

#%%
#w = 99
#while rms>1e-6:
#    x_n = x_p - alpha*(np.dot(q, x_p) - np.dot(c.T, gamma_p))
#    gamma_p = gamma_p + beta*(f[1:n+1] - np.dot(a, theta_d[1:n+1]) - np.dot(c,x_n))
#    rms = LA.norm(x_n - x_p)
##    print(rms)
#    k = k+1
#    for j in range(99,198):
#        if x_n[j]<-1.0:
#            x_n[j] = -1.0
#        elif x_n[j]>1.0:
#            x_n[j] = 1.0
#    x_p = x_n

#%%
theta_n = np.zeros((n+2,1))
theta_n[1:n+1] = x_n[0:n] + theta_d[1:n+1]
theta_n = theta_n.reshape(n+2,1)
u_n = x_n[n:]

#%%
fig, ax2 = plt.subplots()
ax2.plot(xx[1:100], u_a, '-r', label ='Analytical', linewidth=2, marker = 'o')
ax2.plot(xx[1:100], u_n, label ='Numerical', linewidth=3, color = 'blue')
plt.xlim(0,1)
plt.ylabel('Control input '+ r'$U(x)$')
plt.xlabel('x')
ax2.legend()
plt.tight_layout()
fig.savefig('contorl_10_u.eps')

fig, ax2 = plt.subplots()
ax2.plot(xx, theta_a, '-r', label ='Analytical', linewidth=2, marker = 'o')
ax2.plot(xx, theta_n, label ='Numerical', color = 'blue', linewidth=3)
plt.xlim(0,1)
plt.ylabel('Temperature '+r'$\theta(x)$', labelpad=15)
plt.xlabel('x')
ax2.legend()
plt.tight_layout()
fig.savefig('contorl_10_theta.eps')

#%%
obj_f_a = 0.5*(LA.norm(theta_a-theta_d)**2) + 0.5*control*(LA.norm(u_a)**2)
obj_f_n = 0.5*(LA.norm(theta_n-theta_d)**2) + 0.5*control*(LA.norm(u_n)**2)

list = [control, obj_f_a, obj_f_n]
with open('objectiv_function.csv', 'a') as f:
    f.write("\n")
    for item in list:
        f.write("%s\t" % item)
