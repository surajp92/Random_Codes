# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 11:54:55 2019

@author: suraj
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.stats import multivariate_normal as mvn
from numpy import linalg as LA
from sklearn.model_selection import KFold
import random
from numpy.random import seed
seed(222)

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

cl1 = np.argmax(p,axis=1)
z1 = test_labels[test_labels[:,0] == cl1[:]]

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

cl2 = np.argmax(p,axis=1)
z2 = test_labels[test_labels[:,0] == cl2[:]]


#%%
def problem3():
    test_index_set = np.arange(600).reshape(5,-1)
    train_index_set = np.zeros((5,480))
    train_index_set[0,:] = np.arange(120,600)
    train_index_set[1,:] = np.hstack((np.arange(120),np.arange(240,600)))
    train_index_set[2,:] = np.hstack((np.arange(240),np.arange(360,600)))
    train_index_set[3,:] = np.hstack((np.arange(360),np.arange(480,600)))
    train_index_set[4,:] = np.arange(480)
    train_index_set = train_index_set.astype(int)
    
    train_images_new, train_label_new, test_data, test_label = get_data_problem3()
    
    rand_mask = np.array(random.sample(range(600), 600))
    train_images_shf =train_images_new[rand_mask]
    train_labels_shf =train_label_new[rand_mask]
    
    accuracy = np.zeros((5,5))
    l = 0
    Klist = np.array([1,3,5,7,9])
    for K in Klist:
        p = 0
        for j in range(5):
            train_data = train_images_shf[train_index_set[j,:]]
            train_label = train_labels_shf[train_index_set[j,:]]
            val_data = train_images_shf[test_index_set[j,:]]
            val_label = train_labels_shf[test_index_set[j,:]]
            
            pred_label = np.zeros((val_data.shape[0],1))
            
            for i in range(val_data.shape[0]):
                distance = LA.norm(train_data-val_data[i,:], axis=1, keepdims=True)    
                indices = np.argsort(distance,axis=0)
                nn_indices = indices[:K].flatten()
                data_labels = train_label[nn_indices]
                unique, counts = np.unique(data_labels, return_counts=True)
                pred_label[i] = unique[np.argmax(counts)]
            
            accuracy[l,p] = len(val_label[pred_label==val_label])/len(val_label)
            p = p+1
        l = l+1
    
    mean_performance = np.mean(accuracy,axis=1,keepdims=True)    
    K_model = Klist[np.argmax(mean_performance,axis=0)]
    max_accuracy = np.max(mean_performance)
    
    print("Best K = ", K_model)
    print("Mean validation accuracy = ", max_accuracy)
    
    test_pred_label = np.zeros((test_data.shape[0],1))
    for i in range(test_data.shape[0]):
        distance = LA.norm(train_images_shf-test_data[i,:], axis=1, keepdims=True)    
        indices = np.argsort(distance,axis=0)
        nn_indices = indices[:K_model[0]].flatten()
        data_labels = train_labels_shf[nn_indices]
        unique, counts = np.unique(data_labels, return_counts=True)
        test_pred_label[i] = unique[np.argmax(counts)]
    
    test_accuracy = len(test_label[test_pred_label==test_label])/len(test_label)
    print("Test accuracy = ", test_accuracy)
    
    plot_mask = test_pred_label[:] == test_label[:]
    c1, ic1 = np.argmax(plot_mask[:50]), np.argmin(plot_mask[:50])
    c2, ic2 = np.argmax(plot_mask[50:100])+50, np.argmin(plot_mask[50:100])+50
    c7, ic7 = np.argmax(plot_mask[100:])+100, np.argmin(plot_mask[100:])+100
    
    plot_problem3(test_data, c1, ic1, c2, ic2, c7, ic7)

problem3()


#%%














