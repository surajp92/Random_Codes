#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:29:42 2020

@author: suraj
"""

import random
random.seed(10)

import numpy as np
np.random.seed(10)
from numpy import linalg as LA

import time as tm

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.animation as animation

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from keras.regularizers import l2
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

font = {'family' : 'Times New Roman',
        'size'   : 16}    
plt.rc('font', **font)

#'weight' : 'bold'

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

import h5py
from tqdm import tqdm as tqdm

#%%
f = h5py.File('./sst_weekly.mat','r') # can be downloaded from https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu?usp=sharing
lat = np.array(f['lat'])
lon = np.array(f['lon'])
sst = np.array(f['sst'])
time = np.array(f['time'])

sst2 = np.zeros((len(sst[:,0]),len(lat[0,:]),len(lon[0,:])))
for i in tqdm(range(len(sst[:,0]))):
    sst2[i,:,:] = np.flipud((sst[i,:].reshape(len(lat[0,:]),len(lon[0,:]),order='F')))

#%%
lon1 = np.hstack((np.flip(-lon[0,:180]),lon[0,:180]))

x,y = np.meshgrid(lat,lon1,indexing='ij')

#%%    
fig,axs = plt.subplots(1,1, figsize=(12,6))

current_cmap = plt.cm.get_cmap('RdYlBu')
current_cmap.set_bad(color='black',alpha=0.8)

aa = np.hstack((sst2[0,:,180:],sst2[0,:,:180]))

#cs = axs.imshow(sst2[0,:,:],cmap='RdYlBu')
cs = axs.contourf(y,x,aa,120,cmap='RdYlBu')

#axs.grid()
fig.colorbar(cs, ax=axs, orientation='vertical',shrink=0.8)

    
fig.tight_layout()
plt.show()    
