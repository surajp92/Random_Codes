#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 14:16:04 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(1)
import pyfftw
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import os
from numba import jit
from scipy.fftpack import dst, idst

from scipy.ndimage import gaussian_filter
import yaml

from poisson import *
from rhs import *
from time_integration import *
from utils import *

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


#%%
# second-order bc
def bc(nx,ny,w,s):
    
    w[:,0] = -(2.0/dy**2)*(s[:,1]) # bottom wall
    w[:,ny] = -(2.0/dy**2)*(s[:,ny-1]) # top wall
    
    # w[0,:] = -(2.0/dx**2)*(s[1,:]) # left wall
    # w[nx,:] = -(2.0/dx**2)*(s[nx-1,:]) # right wall
    
    return w

# B.C. for vorticity (third-order)
def bc3(nx,ny,w,s):
    
    # bottom wall
    w[:,0] = (1.0/(18*dy**2))*(85.0*s[:,0] - 108.0*s[:,1] + \
                               27.0*s[:,2] - 4.0*s[:,3]) 
    # top wall
    w[:,ny] = (1.0/(18*dy**2))*(85.0*s[:,ny] - 108.0*s[:,ny-1] + \
                                27.0*s[:,ny-2] - 4.0*s[:,ny-3]) 
    
    # # left wall
    # w[0,:] = (1.0/(18*dx**2))*(85.0*s[0,:] - 108.0*s[1,:] + \
    #                            27.0*s[2,:] - 4.0*s[3,:]) 
    # # right wall
    # w[nx,:] = (1.0/(18*dx**2))*(85.0*s[nx,:] - 108.0*s[nx-1,:] + \
    #                             27.0*s[nx-2,:] - 4.0*s[nx-3,:]) 
    
    return w

#%% 
# read input file
with open(r'rbc_parameters.yaml') as file:
    input_data = yaml.load(file, Loader=yaml.FullLoader)
#    input_data = yaml.load(file)
    
file.close()

nx = input_data['nx']
ny = input_data['ny']
ra = float(input_data['ra'])
pr = input_data['pr']
nt = input_data['nt']
lx = input_data['lx']
ly = input_data['ly']
wc = input_data['wc']
wh = input_data['wh']
isolver = input_data['isolver']
icompact = input_data['icompact']
ip = input_data['ip']
its = input_data['its']
eps = float(input_data['eps'])
freq = input_data['freq']
nsmovie = input_data['nsmovie']
nlevel = input_data['nlevel']
pmax = input_data['pmax']
v1 = input_data['v1']
v2 = input_data['v2']
v3 = input_data['v3']
tolerance = float(input_data['tolerance'])
cfl = input_data['cfl']
sigma = input_data['sigma']

re = np.sqrt(ra/pr)

if ip == 1:
    directory = f'RBC_FST_{isolver}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    directory = os.path.join(directory, f'solution_{nx}_{ny}_{ra}_{its}_{isolver}_{icompact}')
    if not os.path.exists(directory):
        os.makedirs(directory)

elif ip == 2:
    directory = f'MG_{isolver}'
    if not os.path.exists(directory):
        os.makedirs(directory)        

#%% 
pi = np.pi

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)

time = 0.0

x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

dtc = np.min((dx,dy))
dtv = 0.25*re*np.min((dx**2, dy**2))
dt = sigma*np.min((dtc, dtv))

#%%
w = np.zeros((nx+1,ny+1)) 
s = np.zeros((nx+1,ny+1))
th = np.zeros((nx+1,ny+1))

w = 1.0e-3*np.sin(np.pi*X)*np.sin(np.pi*Y) #np.random.randn(nx-1,ny-1)

th = 1.0 - Y
th[:,0] = wh
th[:,ny] = wc

#%% 
w0 = np.copy(w)
s0 = np.copy(s)
th0 = np.copy(th)

kc, rw, rs, rth, time = [np.zeros(nt+1) for i in range(5)]
ene, ens, dis, NuH, NuC, NuMean = [np.zeros(nt+1) for i in range(6)]
tprobe = np.zeros((nt+1,5))

tave = 0.0
tmax = 100.0

dt_movie = (tmax - tave)/nsmovie
t_movie = tave
current_time = 0.0
km = 0
   
clock_time_init = tm.time()

for k in range(1,nt+1):
    w0 = np.copy(w)
    s0 = np.copy(s)
    th0 = np.copy(th)
    
    if its == 1:    
        w,s,th = euler(nx,ny,dx,dy,dt,re,pr,w,s,th,input_data,bc,bc3)
    elif its == 2:
        w,s,th = rk3(nx,ny,dx,dy,dt,re,pr,w,s,th,input_data,bc,bc3)
        
    kc[k] = k
    rw[k] = np.linalg.norm(w - w0)/np.sqrt(np.size(w))
    rs[k] = np.linalg.norm(s - s0)/np.sqrt(np.size(s))
    rth[k] = np.linalg.norm(th - th0)/np.sqrt(np.size(th))
    time[k] = time[k-1] + dt
    
    current_time = current_time + dt
    
    dt, ene[k], ens[k], dis[k], NuH[k], NuC[k], NuMean[k], tprobe[k,:] = \
    compute_history(nx,ny,dx,dy,x,y,re,pr,s,w,th,input_data)
    
    if k % freq == 0:
        print('%0.5i %0.3e %0.3e %0.3e %0.3e' % (kc[k], time[k], rw[k], rs[k], rth[k]))

    if rw[k] <= eps and rs[k] <= eps and rth[k] <= eps:
        break
    
    if current_time >= tmax:
        break
    
    if current_time > t_movie:
        filename = os.path.join(directory, f'ws_{km}_{nx}_{ny}_{ra}_{its}_{isolver}_{icompact}.npz')
        np.savez(filename,
                 w = w, s = s,th=th) 
        t_movie = t_movie + dt_movie
        km = km + 1
    
kc = kc[:k+1]
rw = rw[:k+1]
rs = rs[:k+1]
rth = rth[:k+1]
time = time[:k+1]

ene = ene[:k+1]
ens = ens[:k+1]
dis = dis[:k+1]
NuH = NuH[:k+1]
NuC = NuC[:k+1]
NuMean = NuMean[:k+1]
tprobe = tprobe[:k+1,:]
    
total_clock_time = tm.time() - clock_time_init
print('Total clock time=', total_clock_time)

#%%
fout = os.path.join(directory, f'cpu_time_{nx}_{ny}_{ra}_{its}_{isolver}_{icompact}.txt')
input_data['cpu_time'] = total_clock_time

fo = open(fout, "w")
for k, v in input_data.items():
    fo.write(str(k) + ' --- '+ str(v) + '\n')
fo.close()

filename = os.path.join(directory, f'residual_{nx}_{ny}_{ra}_{its}_{isolver}_{icompact}.png')
plot_residual_history(kc,rw,rs,rth,filename)

filename = os.path.join(directory, f'turb_params_{nx}_{ny}_{ra}_{its}_{isolver}_{icompact}.png')
plot_turbulent_parameters(time,ene,ens,dis,NuMean,filename)

filename = os.path.join(directory, f'probe_temperature_{nx}_{ny}_{ra}_{its}_{isolver}_{icompact}.png')
plot_probe_temperature(time, NuH, NuC, tprobe,filename)

ramax = 1.0e6
filename = os.path.join(directory, f'contour_{nx}_{ny}_{ra}_{its}_{isolver}_{icompact}.png')
plot_contour(X,Y,w,s,th,ra,ramax,filename)

filename = os.path.join(directory, f'solution_{nx}_{ny}_{ra}_{its}_{isolver}_{icompact}.npz')
np.savez(filename,
         X = X, Y = Y,
         w = w, s = s, 
         kc = kc, rw = rw, rs = rs)  

