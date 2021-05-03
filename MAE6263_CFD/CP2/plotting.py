#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 08:43:10 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

def spectra_calculation(nx,array_hat):
    # Normalizing data
    array_new = np.copy(array_hat / float(nx))
    # Energy Spectrum
    espec = 0.5 * np.absolute(array_new)**2
    # Angle Averaging
    eplot = np.zeros(int(nx / 2), dtype='double')
    for i in range(1, int(nx / 2)):
        eplot[i] = 0.5 * (espec[i] + espec[nx - i])

    return eplot

fig, ax = plt.subplots(1,2, figsize=(14,6))

isp = 1
nse = 16
nx_list = [1024, 512, 256]
titles = ['WENO', 'CRWENO']

for j in range(2):
    iles = j + 1
    for nx in nx_list:
        data = np.load(f'data_{iles}_{isp}_{nse}_{nx}.npz')
        u = data['u']
        
        
        for ne in range(nse):    
            uf = np.fft.fft(u[ne,:,-1])
            eplot = spectra_calculation(nx,uf)
            if ne == 0:
                eplot1 = eplot
            else:
                eplot1 = eplot1 + eplot
            # ax.loglog(eplot,label=f'n = {ne}')
            
        eplot1 = eplot1/nse
        kw = np.linspace(1,eplot1.shape[0]-1,eplot1.shape[0]-1)
        ax[j].loglog(kw,eplot1[1:], lw=2, label=f'$N = {nx}$')
    
    kstart = 20
    kend = 120
    
    ks = np.linspace(kstart,kend,kend-kstart+1)
    escaling = 2*ks**(-2)
    
    for j in range(2):
        ax[j].loglog(ks,escaling,'k--', lw=2,)
        ax[j].set_ylim([1e-7,1e-1])
    # ax.set_xlim([1,kw[-1]+100])
        ax[j].legend()
        ax[j].set_title(titles[j])
        ax[j].set_xlabel('$k$')
        ax[j].set_ylabel('$E(k)$')
        
plt.show() 
fig.tight_layout()
fig.savefig('numerical_scheme.png', dpi=300)