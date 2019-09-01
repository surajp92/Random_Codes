# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 22:59:21 2019

@author: suraj
"""

import numpy as np
import matplotlib.pyplot  as plt
from numpy.random import seed
seed(1)
from sklearn.datasets import make_spd_matrix
from sklearn.neighbors import KDTree

plt.rcParams.update({'font.size': 12})

#%%
n1 = 1000
n2 = 1000
split = 0.5
nsamples = 5


for i in range(nsamples):
    cov = make_spd_matrix(2,2)
    #cov = np.array([[np.random.randint(1,10),0],[0,np.random.randint(1,200)]])
    mean = np.random.randint(-2,3)*np.random.randn(1,2).flatten()
    d = np.random.multivariate_normal(mean, cov, n1)
    if i == 0:
        data1 = d
    else:
        data1 = np.vstack((data1, d))
    
label1 = np.zeros((n1*nsamples,1))

for i in range(nsamples):
    cov = make_spd_matrix(2,2)
    #cov = np.array([[np.random.randint(5,20),0],[0,np.random.randint(1,100)]])
    mean = np.random.randint(-2,3)*np.random.randn(1,2).flatten()
    d = np.random.multivariate_normal(mean, cov, n1)
    if i == 0:
        data2 = d
    else:
        data2 = np.vstack((data2, d))

label2 = np.ones((n2*nsamples,1))

data = np.vstack((data1,data2))
labels = np.vstack((label1,label2))

plt.scatter(data1[:,0], data1[:,1])
plt.scatter(data2[:,0], data2[:,1])
plt.show()

#%%
indices = np.full(data.shape[0],True)
indices[:int(data.shape[0]*split)] = False
np.random.shuffle(indices)

xtrain, ytrain = data[indices==True], labels[indices==True]
xtest, ytest = data[indices==False], labels[indices==False]

#%%
def linear_classifier(xtrain, ytrain):
    xtxi = np.linalg.inv(np.dot(xtrain.T, xtrain))
    beta = np.dot(np.dot(xtxi,xtrain.T),ytrain)
    
    return beta

beta = linear_classifier(xtrain, ytrain)

#%%
ypred = np.zeros((ytest.shape))

for i in range(xtest.shape[0]):
    xt = xtest[i].reshape(1,xtest.shape[1])
    if np.dot(xt,beta) < 0.5:
        ypred[i] = 0
    else:
        ypred[i] = 1

#%%
def mask_results(xtrain, ytrain, xtest, ytest, ypred):
    train_c0 = xtrain[ytrain[:,0]==0]
    train_c1 = xtrain[ytrain[:,0]==1]
    test_c0 = xtest[(ytest[:,0]==0) & (ypred[:,0]==0)]
    test_c1 = xtest[(ytest[:,0]==1) & (ypred[:,0]==1)]
    test_ic0 = xtest[(ytest[:,0]==0) & (ypred[:,0]==1)]
    test_ic1 = xtest[(ytest[:,0]==1) & (ypred[:,0]==0)]
    
    return train_c0, train_c1, test_c0, test_c1, test_ic0, test_ic1 

train_c0, train_c1, test_c0, test_c1, test_ic0, test_ic1 = mask_results(xtrain, ytrain, xtest, ytest, ypred)

#%%
def plot_results(train_c0, train_c1, test_c0, test_c1, test_ic0, test_ic1):
    fig, ax = plt.subplots(figsize=(10,8))
    
    ax.scatter(train_c0[:,0],train_c0[:,1],marker='x',color='r',label='Training: class 0')
    ax.scatter(train_c1[:,0],train_c1[:,1],marker='x',color='b',label='Training: class 1')
    
    ax.scatter(test_c0[:,0],test_c0[:,1],marker='x',color='g',label='Correctly classified class 0')
    plt.scatter(test_c1[:,0],test_c1[:,1],marker='x',color='m',label='Correctly classified class 1')
    
    ax.scatter(test_ic0[:,0],test_ic0[:,1],marker='D',color='k',label='Incorrectly classified class 0')
    ax.scatter(test_ic1[:,0],test_ic1[:,1],marker='D',color='darkorange',label='Incorrectly classified class 1')
    
    ax.set_xlabel(r'$x_1$',fontsize=18)
    ax.set_ylabel(r'$x_2$',fontsize=18)
    ax.legend()
    plt.show()

plot_results(train_c0, train_c1, test_c0, test_c1, test_ic0, test_ic1)

#%%
tree = KDTree(xtrain, leaf_size=2)

ypred_kdtree = np.zeros((ytest.shape))

for i in range(xtest.shape[0]):
    xt = xtest[i].reshape(1,xtest.shape[1])
    nearest_dist, nearest_ind = tree.query(xt, k=1)
    ypred_kdtree[i] = ytrain[nearest_ind[0]]

train_c0_kdt, train_c1_kdt, test_c0_kdt, test_c1_kdt, test_ic0_kdt, test_ic1_kdt = mask_results(xtrain, 
                                                                                                ytrain, 
                                                                                                xtest, 
                                                                                                ytest, 
                                                                                                ypred_kdtree)

plot_results(train_c0_kdt, train_c1_kdt, test_c0_kdt, test_c1_kdt, test_ic0_kdt, test_ic1_kdt)