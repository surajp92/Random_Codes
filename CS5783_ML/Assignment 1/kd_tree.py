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
        print(data, " ", current_distance)
        
        if nearest_distance == None or current_distance < nearest_distance:
            nearest_distance = current_distance
            nearest_neighbour = data

    return nearest_neighbour, nearest_distance

train_data = np.random.randn(5,2)
test_data = np.array([0.5,0.5])

find_nearest_neighbour(train_data, test_data)

#%%
k = 3
train_data = np.random.randn(5,3)

def build_kdtree(train_data, depth = 0):
    n = (train_data.shape[0])
   
    if n <= 2:
        return train_data
    
    axis = depth%k
    print(axis, depth)
     
    data_mean = np.mean(train_data[:,axis])
        
    return {
            'data': np.array(data_mean),
            'left': build_kdtree(train_data[train_data[:,axis] < data_mean], depth+1),
            'right': build_kdtree(train_data[train_data[:,axis] > data_mean], depth+1)
            }

s_data = (build_kdtree(train_data))
pprint.pprint(s_data)
                      

#%%
    
