#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 17:00:52 2021

@author: suraj
"""

import numpy as np

A = np.array([[1,1,1,1,1],[0,-1,1,2,3],[0,1,1,4,9],[0,-1,1,8,27],[0,1,1,16,81]])

B = np.array([0,0,2,0,0])

X = np.linalg.inv(A) @ B
print(np.sum(X))

#%%
A1 = np.array([[1,-1,1,2],[-2,1,1,4],[3,-1,1,8],[-4,1,1,16]])

B1 = np.array([0,2,0,0])

X1 = np.linalg.inv(A1) @ B1
print(np.sum(X1))

#%%
A2 = np.array([[1,4,9],[1,8,27],[1,16,81]])

B2 = np.array([2,0,0])

X2 = np.linalg.inv(A2) @ B2
print(np.sum(X2))