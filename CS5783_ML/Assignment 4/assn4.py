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


#%%
# Define the kernel function
def kernel_se(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

def kernel_e(a, b, param):
    absdist = np.abs(a - b.reshape(-1,))
    return np.exp(-1.0 * (1/param) * absdist)

def gp_kernel_se(xtrain,ytrain,xtest,ytest,param):
    K_ss = kernel_se(xtest, xtest, param)
        
    # Apply the kernel function to our training points
    K = kernel_se(xtrain, xtrain, param)
    L = np.linalg.cholesky(K + (0.09569**2)*np.eye(len(xtrain)))
    
    # Compute the mean at our test points.
    K_s = kernel_se(xtrain, xtest, param)
    Lk = np.linalg.solve(L, K_s)
    mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((xtest.shape[0],))
    
    # Compute the standard deviation so we can plot it
    s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
    stdv = np.sqrt(s2)
    
    return mu, stdv

def gp_kernel_e(xtrain,ytrain,xtest,ytest,param):
    K_ss = kernel_e(xtest, xtest, param)
        
    # Apply the kernel function to our training points
    K = kernel_e(xtrain, xtrain, param)
    L = np.linalg.cholesky(K + (0.09569**2)*np.eye(len(xtrain)))
    
    # Compute the mean at our test points.
    K_s = kernel_e(xtrain, xtest, param)
    Lk = np.linalg.solve(L, K_s)
    mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((xtest.shape[0],))
    
    # Compute the standard deviation so we can plot it
    s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
    stdv = np.sqrt(s2)
    
    return mu, stdv
    
#%%
def problem1():
    data = np.loadtxt('crash.txt')
    m,n = data.shape
    
    
    mindata = np.min(data,axis=0)
    data = data - mindata
    maxdata = np.max(data,axis=0)
    data = data/maxdata
    
    folds = 5
    cv_results_se = np.zeros((101,6))
    cv_results_e = np.zeros((101,6))
    params = np.linspace(0.01,0.5,101)
    foldsize = int(data.shape[0]/folds)
    
    random_indices = np.arange(data.shape[0])
    np.random.shuffle(random_indices)
    
    for j in range(params.shape[0]):
        param = params[j]
        cv_results_se[j,0] = param
        
        for i in range(5):
            randind_train = np.vstack((random_indices[:i*foldsize].reshape(-1,1),
                                       random_indices[(i+1)*foldsize:].reshape(-1,1)))
            
            randind_test = random_indices[i*foldsize:(i+1)*foldsize]
            
            xtrain = data[randind_train.reshape(-1,),0].reshape(-1,1)
            ytrain = data[randind_train.reshape(-1,),1].reshape(-1,1)
            
            xtest = data[randind_test.reshape(-1,),0].reshape(-1,1)
            ytest = data[randind_test.reshape(-1,),1].reshape(-1,1)
            
            mu_se,stdev_se = gp_kernel_se(xtrain,ytrain,xtest,ytest,param)
            mse = np.mean((np.square(mu_se - ytest.reshape(-1,))))
            cv_results_se[j,i+1] = mse
            
            mu_e,stdev_e = gp_kernel_e(xtrain,ytrain,xtest,ytest,param)
            mse = np.mean((np.square(mu_e - ytest.reshape(-1,))))
            cv_results_e[j,i+1] = mse
    
    mean_mse_se = np.mean(cv_results_se[:,1:],axis=1)        
    opt_param_se = params[np.argmin(mean_mse_se)]
    
    mean_mse_e = np.mean(cv_results_e[:,1:],axis=1)        
    opt_param_e = params[np.argmin(mean_mse_e)]
    
    
    xtrain, ytrain = data[:,0].reshape(-1,1), data[:,1].reshape(-1,1)
    xtest = np.linspace(0,1,101)
    xtest = xtest.reshape(-1,1)
    
    mu_se,stdev_se = gp_kernel_se(xtrain,ytrain,xtest,ytest,opt_param_se)
    mu_e,stdev_e = gp_kernel_e(xtrain,ytrain,xtest,ytest,opt_param_e)
    
    print('Squeared exponential')
    print('Optimal sigma = ', format(opt_param_se, '.3f'))
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,4))
    ax.plot(data[:,0],data[:,1],marker='o',label='True data')
    ax.plot(xtest,mu_se,marker='o',label='Gaussian process')
    ax.legend()
    ##plt.gca().fill_between(xtest.flat, mu-1*stdv, mu+1*stdv, color="#dddddd")
    plt.show()
    
    print('Exponential')
    print('Optimal sigma = ', format(opt_param_e, '.3f'))
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,4))
    ax.plot(data[:,0],data[:,1],marker='o',label='True data')
    ax.plot(xtest,mu_e,marker='o',label='Gaussian process')
    ax.legend()
    ##plt.gca().fill_between(xtest.flat, mu-1*stdv, mu+1*stdv, color="#dddddd")
    plt.show()

problem1()

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

K = 10
# part 1
centroid = np.random.rand(10,784)*255

# part 2

#part 3
centroid = np.zeros((K,test_images.shape[1]))
for k in range(K):
    centroid[k,:] = test_images[test_labels.reshape(-1,)==k][0]

# part 4


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
    print('.',end =" ")
    #print(n, " res = ", residual)
    n = n+1

print(n, " Iterations")

J = 0
for k in range(K):
    temp = test_images[ind == k]
    b = temp - centroid[k,:]
    J = J + LA.norm(b)**2

fig, axs = plt.subplots(nrows=2,ncols=5,figsize=(10,4))
ax = axs.flat

for k in range(K):
    ax[k].imshow(centroid[k].reshape(28,28))
    ax[k].set_xticks([])
    ax[k].set_yticks([])

fig.tight_layout()
plt.show()

#%%    
unique, counts = np.unique(ind, return_counts=True)
unique1, counts1 = np.unique(test_labels, return_counts=True)

comp = test_labels == ind.reshape(-1,1)
u,c = np.unique(comp, return_counts=True)

dist = LA.norm(centroid[0,:]-test_images[0,:], 2)**2


#%% problem 3









