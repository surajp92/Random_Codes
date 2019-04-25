#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:47:34 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt

at = np.loadtxt(open('at.csv', "rb"), delimiter=",", skiprows=0)
agp = np.loadtxt(open('agp.csv', "rb"), delimiter=",", skiprows=0)
aml = np.loadtxt(open('aml.csv', "rb"), delimiter=",", skiprows=0)
m,n = at.shape

#%% plotting
font = {'family' : 'Times New Roman',
        'size'   : 14}	
plt.rc('font', **font)

nrows = int((n-1)/2)
k = 1
fig, axs = plt.subplots(nrows, 2, figsize=(11,8))#, constrained_layout=True)
for i in range(nrows):
    for j in range(2):
        axs[i,j].plot(at[:,1],at[:,k+1], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
        axs[i,j].plot(agp[:,1],agp[:,k+1], color='blue', linestyle='-.', label=r'$y_'+str(i+1)+'$'+' (GP)', zorder=5) 
        axs[i,j].plot(aml[:,0],aml[:,k], color='red', linestyle='--', label=r'$y_'+str(i+1)+'$'+' (GP)', zorder=5)
        axs[i,j].set_xlim([0, at[m-1,1]])
        axs[i,j].set_ylabel('$a_'+'{'+(str(k)+'}'+'$'), fontsize = 14)
        axs[i,j].set_xlabel('Time', fontsize = 14)
        k = int(k+1)

fig.tight_layout() 

fig.subplots_adjust(bottom=0.1)

line_labels = ["True", "ROM-G", "ML"]#, "ML-Train", "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 13} ) #bbox_to_anchor=(0.5, 0.0), borderaxespad=0.1, 

fig.savefig('gp.eps')#, bbox_inches = 'tight', pad_inches = 0.01)