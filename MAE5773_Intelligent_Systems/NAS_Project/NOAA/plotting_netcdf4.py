#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:32:51 2020

@author: suraj
"""

from mpl_toolkits.basemap import Basemap
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

#%%
import h5py
from tqdm import tqdm as tqdm

f = h5py.File('./sst_weekly.mat','r') # can be downloaded from https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu?usp=sharing
lat = np.array(f['lat'])
lon = np.array(f['lon'])
sst = np.array(f['sst'])
time = np.array(f['time'])

sst2 = np.zeros((len(sst[:,0]),len(lat[0,:]),len(lon[0,:])))
for i in tqdm(range(len(sst[:,0]))):
    sst2[i,:,:] = np.flipud((sst[i,:].reshape(len(lat[0,:]),len(lon[0,:]),order='F')))
    # sst2[i,:,:] = np.flipud((sst[i,:].reshape(len(lon[0,:]),len(lat[0,:]))))
    
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

#%%
fig = plt.figure(figsize=(14,7))

m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
# draw parallels and meridians.

m.drawparallels(np.arange(-90.,91.,30.))
m.drawmeridians(np.arange(-180.,181.,60.))

x, y = m(*np.meshgrid(lon1,lat))
m.pcolormesh(x,y,aa,shading='flat',cmap='jet')

# x, y = m(*np.meshgrid(lon,lat))
# m.pcolormesh(x,y,sst2[0,:,:],shading='flat',cmap=plt.cm.jet)

m.drawmapboundary(fill_color='#FFFFFF')
m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])

m.colorbar(location='right')

fig.tight_layout()
plt.title("NOAA SST")
plt.show()
fig.savefig('noaa_sst.png',dpi=300)

#%%
fig = plt.figure(figsize=(14,7))

m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
# draw parallels and meridians.

m.drawparallels(np.arange(-90.,91.,30.))
m.drawmeridians(np.arange(-180.,181.,60.))

m.drawmapboundary(fill_color='#FFFFFF')
m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])

m.colorbar(location='right')

plt.title("NOAA SST")

#%%
fig = plt.figure(figsize=(14,7))

plt.xticks([])
plt.yticks([])
    
def animate(i): 
    x, y = m(*np.meshgrid(lon1,lat))
    aa = np.hstack((sst2[5*i,:,180:],sst2[5*i,:,:180]))
    cont = plt.pcolormesh(x,y,aa,shading='flat',cmap='jet')
    return cont  
    
anim = animation.FuncAnimation(fig, animate, frames=50)
fig.tight_layout()
anim.save('animation.mp4')

