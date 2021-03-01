#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 17:32:57 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

class pdeEx1:
    def __init__(self):
        self.sin = lambda x: np.sin(2.0*np.pi*x)
        # excact solution
        self.exactu = lambda x: x*np.exp(x) + 1.0

        # right hand side
        self.f = lambda x : np.exp(x)*(-4.0 - x + x**3 - (2.0 + x)*np.cos(x)
                             + (1.0 + x)*np.sin(x)) + x**2 + 1.0
        
        # Dirichlet boundary condition
        self.gD = lambda x : self.exactu(x)
        
        # Neumann boundary condition
        self.gN = lambda x : self.a(x)*self.Du(x)
        
        # Derivative of the exact solution
        self.Du = lambda x : (x + 1.0)*np.exp(x)
        
        # Diffusion coefficient function
        self.a = lambda x : 2.0 + np.cos(x)
        
        # Reacation coefficient function
        self.c = lambda x : 1.0 + x**2
        
        
if __name__ == "__main__":
    domain = [0,1]
    n = 101      

    x = np.linspace(domain[0],domain[1],101)
    x = np.reshape(x,[-1,1])
    pde = pdeEx1()
    
    def trial(fun):
        x = np.linspace(0,1,11)
        plt.plot(x,fun(x), 's--')
        plt.show()
    
    trial(pde.gN)
    
    plt.plot(x, pde.gN(x))
    
