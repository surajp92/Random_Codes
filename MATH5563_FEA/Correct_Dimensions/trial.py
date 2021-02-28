#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:02:33 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

def fun(x,f):
    return f(x)
    
f = lambda x: np.sin(2.0*np.pi*x)    
x = np.linspace(0,1,101)

u = fun(x,f) 

plt.plot(x,u)
plt.show()