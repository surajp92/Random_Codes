# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 11:54:55 2019

@author: suraj
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.stats import multivariate_normal as mvn

#%%
train_images, train_labels, test_images, test_labels = get_data()

#%%
# problem 1
cat, counts = np.unique(train_labels, return_counts=True)
theta_kd = np.zeros((cat.shape[0],train_images.shape[1]))
prior = np.zeros(cat.shape[0])
for c in cat:
    current_x = train_images[train_labels[:,0] == c]
    theta_kd[c,:] = np.sum(current_x,axis=0,keepdims=True)/counts[c]+1e-3
    prior[c] = len(train_labels[train_labels[:,0] == c])/len(train_labels)
    
n,d = test_images.shape    
k = len(counts)
p = np.zeros((n,k))

for j in range(n):
    for c in cat:
        mask1 = test_images[j,:] == 1
        prob1 = test_images[j][mask1]*theta_kd[c][mask1]
        mask2 = test_images[j,:] == 0
        prob0 = 1-theta_kd[c][mask2]
        p[j,c] = np.sum(np.log(prob1)) + np.sum(np.log(prob0)) + np.log(prior[c])

cl = np.argmax(p,axis=1)
z = test_labels[test_labels[:,0] == cl[:]]

    
#%%
# problem 2
mean_kd = np.zeros((cat.shape[0],train_images.shape[1]))
var_kd = np.zeros((cat.shape[0],train_images.shape[1]))
for c in cat:
    current_x = train_images[train_labels[:,0] == c]
    mean_kd[c,:] = np.mean(current_x,axis=0,keepdims=True)
    var_kd[c,:] = np.var(current_x,axis=0,keepdims=True)+1e-3

n,d = test_images.shape
k = len(counts)
p = np.zeros((n,k))
for c in cat:
    p[:,c] = mvn.logpdf(test_images,mean=mean_kd[c,:], cov=var_kd[c,:])

cl = np.argmax(p,axis=1)
z = test_labels[test_labels[:,0] == cl[:]]
    