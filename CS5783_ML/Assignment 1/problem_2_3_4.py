# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 22:59:21 2019

@author: suraj
"""

import numpy as np
import matplotlib.pyplot  as plt
from numpy.random import seed
seed(123)
from sklearn.neighbors import KDTree
from sklearn.datasets import make_spd_matrix

plt.rcParams.update({'font.size': 12})

#%%
def mask_results(xtrain, ytrain, xtest, ytest, ypred):
    train_c0 = xtrain[ytrain[:,0]==0]
    train_c1 = xtrain[ytrain[:,0]==1]
    test_c0 = xtest[(ytest[:,0]==0) & (ypred[:,0]==0)]
    test_c1 = xtest[(ytest[:,0]==1) & (ypred[:,0]==1)]
    test_ic0 = xtest[(ytest[:,0]==0) & (ypred[:,0]==1)]
    test_ic1 = xtest[(ytest[:,0]==1) & (ypred[:,0]==0)]
    
    return train_c0, train_c1, test_c0, test_c1, test_ic0, test_ic1 

#%%
def plot_results(train_c0, train_c1, test_c0, test_c1, test_ic0, test_ic1,name):
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
    fig.savefig(name)
    
#%%
n1 = 5000
n2 = 5000
split = 0.5

mean1 = np.random.randn(1,2).flatten()
cov1 = make_spd_matrix(2,2)
data1 = np.random.multivariate_normal(mean1, cov1, n1)
label1 = np.zeros((n1,1))

mean2 = np.random.randn(1,2).flatten()
cov2 = make_spd_matrix(2,2)
data2 = np.random.multivariate_normal(mean2, cov2, n2)
label2 = np.ones((n2,1))

data = np.vstack((data1,data2))
labels = np.vstack((label1,label2))

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

ypred = np.zeros((ytest.shape))
ypred = np.dot(xtest,beta)
ypred = np.array(ypred > 0.5, dtype=int)

train_c0, train_c1, test_c0, test_c1, test_ic0, test_ic1 = mask_results(xtrain, ytrain, xtest, ytest, ypred)

lin_accuracy = (test_c0.shape[0] + test_c1.shape[0])/(ytest.shape[0])

plot_results(train_c0, train_c1, test_c0, test_c1, test_ic0, test_ic1,'linear_classifier.pdf')

#%%
tree = KDTree(xtrain, leaf_size=2)

nearest_dist, nearest_ind = tree.query(xtest, k=1)
ypred_kdtree = ytest[nearest_ind[:,0]]

train_c0_kdt, train_c1_kdt, test_c0_kdt, test_c1_kdt, test_ic0_kdt, test_ic1_kdt = mask_results(xtrain, 
                                                                                                ytrain, 
                                                                                                xtest, 
                                                                                                ytest, 
                                                                                                ypred_kdtree)

kdtree_accuracy = (test_c0_kdt.shape[0] + test_c1_kdt.shape[0])/(ytest.shape[0])

plot_results(train_c0_kdt, train_c1_kdt, test_c0_kdt, test_c1_kdt, test_ic0_kdt, test_ic1_kdt,'kdtree_classifier.pdf')

#%%
print('Linear classifier accuracy = ', round(lin_accuracy*100,2), '%')
print('kD tree classifier accuracy = ', round(kdtree_accuracy*100,2), '%')