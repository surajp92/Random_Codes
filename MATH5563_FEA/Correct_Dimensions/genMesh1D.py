#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 10:33:48 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
'''
Usage: form finite element global DoF of a uniform mesh of

INPUTS:
mesh --- a struct data contains very rich mesh information
pd --- the degree of polynomial used to form FE spaces.
                 Possible values: 1,2,3,4,5...

OUTPUTS:
fem --- a struct data contains the following fields:
fem.p --- x- and y- coordinates of each vertice corresponds to
                    a global degree of freedom.
         fem.t --- indices of global degrees of freedom in each element
         fem.pd --- degree of polynomial used to form FE spaces.

'''

class genMesh1D:
  def __init__(self, domain, n):
      xmin = domain[0]
      xmax = domain[1]
      
      self.p = np.linspace(xmin, xmax, n+1)
      self.p = np.reshape(self.p, [-1,1])
      
      self.t = np.zeros((n,2), dtype='int32')
      self.t[:,0] = np.arange(0,n,1, dtype='int32')
      self.t[:,1] = np.arange(1,n+1,1, dtype='int32')
      
if __name__ == "__main__":
    domain = [0,1]
    n = 5      
    mesh = genMesh1D(domain, n)
    
    print('mesh.p = \n', mesh.p)      
    print('mesh.t = \n', mesh.t)      
    
    