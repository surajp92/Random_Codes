# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 20:59:29 2019

@author: suraj
"""

import numpy as np
import matplotlib as plt
from numpy.random import seed
seed(1)
import pprint

from sklearn.neighbors import KDTree

#%%
def distane_squared(data1, data2):
    x1, y1 = data1[0], data1[1]
    x2, y2 = data2[0], data2[1]
    
    dx = x2 - x1
    dy = y2-y1
    
    return np.sqrt(dx**2 + dy**2)

a = np.array([3,4])
b = np.array([4,5])

d = distane_squared(a, b)
print("distacnce = ", d)

#%%
def find_nearest_neighbour(train_data, test_data):
    nearest_neighbour = None
    nearest_distance = None
    
    for data in train_data:
        current_distance = distane_squared(data, test_data)
        
        if nearest_distance == None or current_distance < nearest_distance:
            nearest_distance = current_distance
            nearest_neighbour = data

    return nearest_neighbour, nearest_distance

#%%
k = 2
def build_kdtree_median(train_data, depth = 0):
    
    n = (train_data.shape[0])
   
    if n <= 0:
        return None
    
    axis = depth%k
     
    train_data = train_data[train_data[:,axis].argsort()]
        
    return {
            'data': np.array(train_data[n//2,:]),
            'left': build_kdtree_median(train_data[:n//2,:], depth+1),
            'right': build_kdtree_median(train_data[n//2+1:,:], depth+1)
            }
    
def build_kdtree_mean(train_data, depth = 0):
    n = (train_data.shape[0])
   
    if n <= 1:
        return train_data
    
    axis = depth%k
     
    data_mean = np.mean(train_data[:,axis])
        
    return {
            'data': np.array(data_mean),
            'left': build_kdtree_mean(train_data[train_data[:,axis] < data_mean], depth+1),
            'right': build_kdtree_mean(train_data[train_data[:,axis] > data_mean], depth+1)
            }
                      
#%%
def kdtree_nearest_neighbour(kdtree, test_data, depth = 0):
    k = 2
    axis = depth % k
           
    if test_data[:,axis] < kdtree['data']:
        next_branch = kdtree['left']
    else:
        next_branch = kdtree['right']
    
    if type(next_branch) != dict:
        print((next_branch[0].shape))
        return next_branch[0]
    else:
        return kdtree_nearest_neighbour(next_branch, test_data, depth + 1)   

def kdtree_nearest_neighbour_median(kdtree, test_data, nearest=None, depth = 0):
    k = 2

    if kdtree == None:
        return nearest
    
    axis = depth % k
    
    next_best = None
    next_branch = None
    
    if nearest.any() == None or distane_squared(test_data,nearest) > distane_squared(test_data,kdtree['data']):
        next_best = kdtree['data']
    else:
        next_best = nearest
        
    if test_data[axis] < kdtree['data'][axis]:
        next_branch = kdtree['left']
    else:
        next_branch = kdtree['right']
        
    return kdtree_nearest_neighbour_median(next_branch, test_data, next_best, depth + 1)   

#%%
def closer_distance(test_data, p1, p2):
    
    if p1 is None:
        return p2
    
    if p2 is None:
        return p1
    
    d1 = distane_squared(test_data,p1)
    d2 = distane_squared(test_data,p2)
    
    if d1 < d2:
        return p1
    else:
        return p2

def kdtree_nearest_neighbour_advanced(kdtree, test_data, depth = 0):
    k = 2

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
    
    nearest = closer_distance(test_data,
                           kdtree_nearest_neighbour_advanced(next_branch, test_data, depth + 1),
                           kdtree['data'])
    
    if distane_squared(test_data, nearest) > abs(test_data[axis] - kdtree['data'][axis]):
        nearest = closer_distance(test_data,
                           kdtree_nearest_neighbour_advanced(opposite_branch, test_data, depth + 1),
                           nearest)
    
    return nearest

       
#%%
train_data = np.random.randn(10,2)
test_data = np.array([0.5,0.5])

#%%
nearest_neighbour, nearest_distance = find_nearest_neighbour(train_data, test_data)
print(nearest_neighbour)

#%%
kdtree_median = (build_kdtree_median(train_data))

#%%
nearest = np.array([None,None])
nn_simple = kdtree_nearest_neighbour_median(kdtree_median, test_data, nearest)
print(nn_simple)

#%%
nn_advanced = kdtree_nearest_neighbour_advanced(kdtree_median, test_data)
print(nn_advanced)
nn_advanced_ind = np.argwhere(train_data == nn_advanced)[0,0]
print(nn_advanced_ind)

#%%
tree = KDTree(train_data, leaf_size=2)
nearest_dist, nearest_ind = tree.query(test_data.reshape((1,2)), k=1)
print(train_data[nearest_ind[0]])
print(nearest_ind[0])




    
