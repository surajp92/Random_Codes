#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 10:58:10 2019

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

font = {'family' : 'Times New Roman',
        'size'   : 12}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
data = np.loadtxt('crash.txt')
m,n = data.shape

mask = np.linspace(0,m-1,m)%2

training_set = data[mask==0]
validation_set = data[mask==1]

L = 20

phi_train = np.zeros((training_set.shape[0],L))
phi_val = np.zeros((validation_set.shape[0],L))
phi_data = np.zeros((data.shape[0],L))

error_train = np.zeros((L,1))
error_val = np.zeros((L,1))

for i in range(L):
    phi_train[:,i] = training_set[:,0]**i
    phi_val[:,i] = validation_set[:,0]**i
    phi_data[:,i] = data[:,0]**i

#%%
for l in range(L):
    a = np.dot(phi_train[:,:l+1].T,phi_train[:,:l+1])
    b = np.dot(phi_train[:,:l+1].T,training_set[:,1])
    w = np.linalg.solve(a, b)
    
    that_train = np.dot(phi_train[:,:l+1],w)
    that_val = np.dot(phi_val[:,:l+1],w)
    
    error_train[l] = LA.norm(training_set[:,1] - that_train)
    error_val[l] = LA.norm(validation_set[:,1] - that_val)

#%%
l_optimal = np.argmin(error_val) 
a = np.dot(phi_train[:,:l_optimal+1].T,phi_train[:,:l_optimal+1])
b = np.dot(phi_train[:,:l_optimal+1].T,training_set[:,1])

w_optimal = np.linalg.solve(a, b)

pred = np.dot(phi_data[:,:l_optimal+1],w_optimal)

#%%
fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10,6))

ax[0].plot(np.linspace(1,L,L),error_train,'k',lw='2',label='Training set')
ax[0].plot(np.linspace(1,L,L),error_val,'r',lw='2',label='Validation set')
ax[0].set_xticks(np.linspace(1,L,L))
ax[0].legend(loc=0)

ax[1].plot(data[:,0],pred,'b',lw=2,label='Prediction')
ax[1].scatter(training_set[:,0],training_set[:,1],color='g',label='Training set')
ax[1].scatter(validation_set[:,0],validation_set[:,1],color='darkorange',label='Validation set')
ax[1].legend(loc=4)

plt.show()
