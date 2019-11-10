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

from numpy.random import seed
seed(1)

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

test_images_file = open('t10k-images-idx3-ubyte','rb')
test_images = test_images_file.read()
test_images_file.close()
test_images = bytearray(test_images)
test_images = test_images[16:]

test_labels_file = open('t10k-labels-idx1-ubyte','rb')
test_labels = test_labels_file.read()
test_labels_file.close()
test_labels = bytearray(test_labels)
test_labels = test_labels[8:]

test_images = np.array(test_images)
test_images = test_images.reshape(10000,-1)

test_labels = np.array(test_labels)
test_labels = test_labels.reshape(10000,-1)

#test_images[test_images[:,:]>=1] = 1
#test_images[test_images[:,:]<1] = 0

#%%
K = 10
#centroid = np.random.randint(np.max(test_images),size=(10,784))
centroid = np.random.rand(10,784)*255
centroid_temp = np.zeros((10,784))

dist_mat = np.zeros((test_images.shape[0],10))

residual = 1

n = 0

#%%
while residual > 0.001:
    for k in range(K):
        b = test_images - centroid[k,:]
        dist_mat[:,k] = LA.norm(b,axis=1)**2
    
    ind = np.argmin(dist_mat,axis=1)

    for k in range(K):
        temp = test_images[ind == k]
        if temp.size != 0:
            centroid_temp[k,:] = np.average(temp,axis=0)
    
    residual = LA.norm(centroid - centroid_temp)
    centroid[:,:] = centroid_temp[:,:]
    print(n, " res = ", residual)
    n = n+1

#%%
J = 0
for k in range(K):
    temp = test_images[ind == k]
    b = temp - centroid[k,:]
    J = J + LA.norm(b)**2
    
unique, counts = np.unique(ind, return_counts=True)
unique1, counts1 = np.unique(test_labels, return_counts=True)

comp = test_labels == ind.reshape(-1,1)
u,c = np.unique(comp, return_counts=True)

#%%
dist = LA.norm(centroid[0,:]-test_images[0,:], 2)**2











