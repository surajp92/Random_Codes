#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 11:02:32 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
def bas1DP1(x,vert,ibas,dind):
    x1 = vert[0]
    x2 = vert[1]
    
    if dind == 0:
        if ibas == 1:
            f = (x - x2)/(x1 - x2)
        elif ibas == 2:
            f = (x - x1)/(x2 - x1)
    
    elif dind == 1:
        if ibas == 1:
            f = (1/(x1 - x2))*np.ones(np.shape(x)[0])
        elif ibas == 2:
            f = (1/(x2 - x1))*np.ones(np.shape(x)[0])
    
    return f

def bas1DP2(x,vert,ibas,dind):
    h = vert[1] - vert[0]
    x1 = vert[0]
    x2 = vert[0] + 0.5*h
    x3 = vert[1]
    
    if dind == 0:
        if ibas == 1:
            f = ((x - x2)*(x - x3))/((x1 - x2)*(x1 - x3))
        elif ibas == 2:
            f = ((x - x1)*(x - x3))/((x2 - x1)*(x2 - x3))
        elif ibas == 3:
            f = ((x - x1)*(x - x2))/((x3 - x1)*(x3 - x2))
    
    elif dind == 1:
         # to do
        next
       
    return f

def bas1DPN(x,vert,ibas,dind,n):
    xl = np.linspace(vert[0],vert[1],n+1)
    f = np.ones(np.shape(x)[0])
    xp = xl[ibas-1]
    
    for k in range(np.shape(xl)[0]):
        if k == ibas - 1:
            f = f
        else:
            f = f*(x - xl[k])/(xp - xl[k])
    
    return f

if __name__ == "__main__":
    vert = [0,1]
    dind = 0
    ibas = 3
    pd = 4
    
    x = np.linspace(0,1,101)
    f = bas1DP2(x,vert,ibas,dind)
    
    for ibas in range(1,pd+2):
        fg = bas1DPN(x,vert,ibas,dind,pd)
        plt.plot(x,fg,'-',lw=3,label=f'$L_{ibas}(x)$')
    
    plt.xlim([0,1])
    plt.ylim([-0.6,1.2])
    plt.grid()
    plt.legend()
    plt.show()