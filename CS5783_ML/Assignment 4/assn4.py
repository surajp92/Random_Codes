#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:17:01 2019

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.optimize import minimize

font = {'family' : 'Times New Roman',
        'size'   : 12}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
# Define the kernel function
def kernel_se(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

def kernel_e(a, b, param):
    absdist = np.abs(a - b.reshape(-1,))
    return np.exp(-1.0 * (1/param) * absdist)

#%%
data = np.loadtxt('crash.txt')
m,n = data.shape

mindata = np.min(data,axis=0)
data = data - mindata
maxdata = np.max(data,axis=0)
data = data/maxdata

mask = np.linspace(0,m-1,m)%2

train_set = data[mask == 0]
test_set = data[mask == 1]

param = 0.15

xtrain, ytrain = train_set[:,0].reshape(-1,1), train_set[:,1].reshape(-1,1)
xtest, ytest = test_set[:,0].reshape(-1,1), test_set[:,1].reshape(-1,1)

K_ss = kernel_se(xtest, xtest, param)

# Apply the kernel function to our training points
K = kernel_se(xtrain, xtrain, param)
L = np.linalg.cholesky(K + 0.00002*np.eye(len(xtrain)))

# Compute the mean at our test points.
K_s = kernel_se(xtrain, xtest, param)
Lk = np.linalg.solve(L, K_s)
mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((xtest.shape[0],))

# Compute the standard deviation so we can plot it
s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
stdv = np.sqrt(s2)


plt.plot(xtest,ytest,label='True')
plt.plot(xtest,mu,label='GP')
#plt.gca().fill_between(xtest.flat, mu-1*stdv, mu+1*stdv, color="#dddddd")
plt.show()

#%%

















