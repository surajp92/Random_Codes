# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 22:59:21 2019

@author: suraj
"""

import numpy as np
import matplotlib.pyplot  as plt
from numpy.random import seed
seed(1)
import pprint

#%%
mean1 = np.array([0.0,0.0])
cov1 = np.array([[1,0],[0,100]])
data1 = np.random.multivariate_normal(mean1, cov1, 5000)

mean2 = np.array([4.0,30.0])
cov2 = np.array([[1,0],[0,50]])
data2 = np.random.multivariate_normal(mean2, cov2, 5000)

#%%
xtrain = np.vstack((data1,data2))
indices = np.random.randint(0,xtrain.shape[0],xtrain.shape[0])
xtrain = xtrain[indices]

#%%
plt.scatter(data1[:,0],data1[:,1],marker='x',color='r')
plt.scatter(data2[:,0],data2[:,1],marker='x',color='b')
#plt.axis('equal')
plt.show()
