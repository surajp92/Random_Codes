#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:22:34 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib as plt
from numpy.random import seed
seed(1)
import pprint

from sklearn.neighbors import KDTree

#%%
class KDTree1:
    def __init__(self, matrix,depth=0):
        self.train_data = matrix
        self.k = matrix.shape[1]
        
        axis = depth % self.k
        
        n = (matrix.shape[0])
        
        if n <= 0:
            return None
        
        matrix = matrix[matrix[:,axis].argsort()]
        
        depth = depth + 1
        self.data = matrix[n//2,:]
        self.left = KDTree1(matrix[:n//2,:], depth)
        self.right = KDTree1(matrix[n//2+1:,:], depth)
           
    def l2_distane(data1, data2):
        x1, y1 = data1[0], data1[1]
        x2, y2 = data2[0], data2[1]
        
        dx = x2 - x1
        dy = y2-y1
        
        return np.sqrt(dx**2 + dy**2)
        
    def nearer_neighbour(self,test_data, point1, point2):
    
        if point1 is None:
            return point2
        
        if point2 is None:
            return point1
        
        d1 = self.l2_distane(test_data,point1)
        d2 = self.l2_distane(test_data,point2)
        
        if d1 < d2:
            return point1
        else:
            return point2
    
    def kdtree_nearest_neighbour(self, test_data, depth = 0):   
        
        print(self.data)
        axis = depth % self.k
        
        next_branch = None
        opposite_branch = None
        
        if test_data[axis] < self.data[axis]:
            next_branch =self.left
            opposite_branch = self.right
        else:
            next_branch = self.right
            opposite_branch = self.left
        
        if next_branch is None or opposite_branch is None:
            return None
               
        nearest = self.nearer_neighbour(test_data,
                               next_branch.kdtree_nearest_neighbour(test_data, depth + 1),
                               self.data)
        pprint.pprint(nearest)
        
        if self.l2_distane(test_data, nearest) > abs(test_data[axis] - self.data[axis]):
            
            nearest = self.nearer_neighbour(test_data,
                               opposite_branch.kdtree_nearest_neighbour(test_data, depth + 1),
                               nearest)
        
        return nearest

#%%
train_data = np.random.randn(10,2)
test_data = np.array([0.5,0.5])

obj = KDTree1(train_data)

#%%

nearest_neighbour = obj.kdtree_nearest_neighbour(test_data)

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
kdt = build_kdtree(train_data)

#%%
def l2_distane(data1, data2):
        x1, y1 = data1[0], data1[1]
        x2, y2 = data2[0], data2[1]
        
        dx = x2 - x1
        dy = y2-y1
        
        return np.sqrt(dx**2 + dy**2)
        
def nearer_neighbour(test_data, point1, point2):

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

def kdtree_nearest_neighbour(kdtree, test_data, depth = 0):   
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
    
    nearest = nearer_neighbour(test_data,
                           kdtree_nearest_neighbour(next_branch, test_data, depth + 1),
                           kdtree['data'])
    pprint.pprint(nearest)
    
    if l2_distane(test_data, nearest) > abs(test_data[axis] - kdtree['data'][axis]):
        nearest = nearer_neighbour(test_data,
                           kdtree_nearest_neighbour(opposite_branch, test_data, depth + 1),
                           nearest)
    
    return nearest     

#%%
nearest_neighbour = kdtree_nearest_neighbour(kdtree,test_data)

#%%
tree = KDTree(train_data, leaf_size=2)
nearest_dist, nearest_ind = tree.query(test_data.reshape((1,2)), k=1)
print(train_data[nearest_ind[0]])
print(nearest_ind[0])