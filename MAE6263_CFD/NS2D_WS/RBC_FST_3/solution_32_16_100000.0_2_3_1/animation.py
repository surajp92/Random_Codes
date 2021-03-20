#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:31:45 2021

@author: suraj
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

font = {'family' : 'Times New Roman',
        'size'   : 16}    
plt.rc('font', **font)


#'weight' : 'bold'

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


lx = 2.0
ly = 1.0

nx = 32
ny = 16

x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
ns = 50

w = np.zeros((ns, nx+1, ny+1))

for i in range(50):
    data = np.load(f'ws_{i}_32_16_100000.0_2_3_1.npz')
    w[i,:,:] = data['w']

#%%

fig, ax = plt.subplots()

ims = []
for i in range(50):
    im = ax.imshow(w[i,:,:].T, extent=[0, 2, 0, 1], origin='lower',
            interpolation='bicubic',cmap='RdBu_r', alpha=1.0, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                repeat_delay=1000)
plt.show()
ani.save("movie.mp4")
    
#%%    
    
fig = plt.figure(figsize=(14,7))

plt.xticks([])
plt.yticks([])
    
def animate(i): 
    X, Y = np.meshgrid(x, y, indexing='ij')
    aa = w[i,:,:]
    cont = plt.pcolormesh(X,Y,aa,cmap='jet')
    return cont  
    
anim = animation.FuncAnimation(fig, animate, frames=50)
fig.tight_layout()
anim.save('animation.mp4')
