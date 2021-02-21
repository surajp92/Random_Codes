#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 10:33:48 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from genMesh1D import *

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

class genFEM1D:
  def __init__(self, mesh, pd):
      
      nt = np.shape(mesh.t)[0]
      
      if pd == 1:
          self.p = mesh.p
          self.t = mesh.t
          self.pd = pd
      else:
          self.p = np.zeros((pd*nt + 1, 1))
          self.t = np.zeros((nt,pd+1), dtype='int32')
          self.pd = pd
          
          self.p[np.arange(0,pd*nt+1,pd)] = mesh.p
          
          for j in range(1,pd):              
              self.p[np.arange(j,(nt-1)*pd+j+1,pd)] = \
                  (pd-j)/pd*mesh.p[0:-1] + (j)/pd*mesh.p[1:]
          
          temp = np.arange(0,(nt-1)*pd+1,pd)
          
          for j in range(pd+1):
              self.t[:,j] = temp
              temp = temp + 1
          

if __name__ == "__main__":
    domain = [0,1]
    n = 5      
    pd = 1
    mesh = genMesh1D(domain, n)
    fem = genFEM1D(mesh, pd)
    
    print('fem.p = \n ', fem.p)      
    print('fem.t = \n ', fem.t)      
    
    