#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 10:52:51 2021

@author: suraj
"""

import numpy as np

#%%
# 5th order boundary condition first derivative
A = np.array([[1,1,1,1,1],
              [0,1,2**1,3**1,4**1],
              [0,1,2**2,3**2,4**2],
              [0,1,2**3,3**3,4**3],
              [0,1,2**4,3**4,4**4]])

Ai = np.linalg.inv(A)
print(Ai)

f = 24
one = np.array([[0],[1],[0],[0],[0]])
alpha = np.array([[0],[1],[2],[3],[4]])

print(f*(Ai @ one))
print(f*(Ai @ alpha))

#%%
# 3rd order boundary condition second derivative
A = np.array([[1,1,1,1],
              [0,1,2**1,3**1],
              [0,1,2**2,3**2],
              [0,1,2**3,3**3]])

Ai = np.linalg.inv(A)
print(Ai)

f = 6
print(f*Ai)

one = np.array([[0],[0],[2],[0]])
alpha = np.array([[0],[0],[2],[6]])

print(f*(Ai @ one))
print(f*(Ai @ alpha))

#%%
# 3rd order boundary condition second derivative
A = np.array([[1,1,1,1,1],
              [0,1,2**1,3**1,4**1],
              [0,1,2**2,3**2,4**2],
              [0,1,2**3,3**3,4**3],
              [0,1,2**4,3**4,4**4]])

Ai = np.linalg.inv(A)
print(Ai)

f = 12
print(f*Ai)

one = np.array([[0],[0],[2],[0],[0]])
alpha = np.array([[0],[0],[2],[6],[12]])

print(f*(Ai @ one))
print(f*(Ai @ alpha))

#%%
# 5th order boundary condition second derivative
A = np.array([[1,1,1,1,1,1],
              [0,1,2**1,3**1,4**1,5**1],
              [0,1,2**2,3**2,4**2,5**2],
              [0,1,2**3,3**3,4**3,5**3],
              [0,1,2**4,3**4,4**4,5**4],
              [0,1,2**5,3**5,4**5,5**5]])

Ai = np.linalg.inv(A)
print(Ai)

f = 12
print(f*Ai)

#%%
one = np.array([[0],[0],[2],[0],[0],[0]])
alpha = np.array([[0],[0],[2],[6],[12],[20]])

print(f*(Ai @ one))
print(f*(Ai @ alpha))

#%%
pow_6 = np.array([0,1,2,3,4,5])**6
lhs = pow_6*f*(Ai @ one).flatten()
rhs = pow_6*f*(Ai @ alpha).flatten()

alpha_c = np.sum(lhs)/(f*30 - np.sum(rhs))