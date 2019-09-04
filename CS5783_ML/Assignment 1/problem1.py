#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:22:34 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
seed(1)

plt.rcParams.update({'font.size': 12})

#%%
def l2_distane(data1, data2):
        x1, y1 = data1[0], data1[1]
        x2, y2 = data2[0], data2[1]
        
        dx = x2 - x1
        dy = y2-y1
        
        return np.sqrt(dx**2 + dy**2)

#%%    
def build_kdtree(train_data, depth = 0):
    k = train_data.shape[1]
    n = (train_data.shape[0])
   
    if n <= 0:
        return None
    
    axis = depth % k
     
    train_data = train_data[train_data[:,axis].argsort()]
        
    return {
        'data': np.array(train_data[n//2,:]),
        'left': build_kdtree(train_data[:n//2,:], depth+1),
        'right': build_kdtree(train_data[n//2+1:,:], depth+1)
        }

#%%        
def nearer_neighbour(test_data, point1, point2):
    print('2')
    if point1 is None:
        return point2
    
    if point2 is None:
        return point1
    
    d1 = l2_distane(test_data,point1)
    d2 = l2_distane(test_data,point2)
    
    if d1 < d2:
        return point1
    else:
        return point2

#%%
def kdtree_nearest_neighbour(kdtree, test_data, depth = 0):   
    k = test_data.shape[0]
    
    if kdtree == None:
        return None
    
    axis = depth % k
    
    next_branch = None
    opposite_branch = None
    
    if test_data[axis] < kdtree['data'][axis]:
        next_branch = kdtree['left']
        opposite_branch = kdtree['right']
    else:
        next_branch = kdtree['right']
        opposite_branch = kdtree['left']
    
    nearest = nearer_neighbour(test_data,
                           kdtree_nearest_neighbour(next_branch, test_data, depth + 1),
                           kdtree['data'])
    print(nearest)
    
    if l2_distane(test_data, nearest) > abs(test_data[axis] - kdtree['data'][axis]):
        print('1')
        print(nearest)
        nearest = nearer_neighbour(test_data,
                           kdtree_nearest_neighbour(opposite_branch, test_data, depth + 1),
                           nearest)
    
    return nearest  

#%%
train_data = np.random.randn(50,2)
test_data = np.array([0.5,0.5])

#%%
kdtree = build_kdtree(train_data)
nearest_neighbour = kdtree_nearest_neighbour(kdtree,test_data)
nearest_indice = np.argwhere(train_data == nearest_neighbour)[0,0]
print(nearest_neighbour)
print([nearest_indice])

#%%
fig, ax = plt.subplots(figsize=(10,8))
    
ax.scatter(train_data[:,0],train_data[:,1],marker='x', s=100, color='r',label='Training data')
ax.scatter(test_data[0],test_data[1],marker='v',s=100, color='g',label='Test data')
ax.scatter(nearest_neighbour[0],nearest_neighbour[1],marker='s',s=100, color='b',label='Nearest neighbour')

ax.set_xlabel(r'$x_1$',fontsize=18)
ax.set_ylabel(r'$x_2$',fontsize=18)
ax.legend()
plt.show()
fig.savefig('kdtree.pdf')

