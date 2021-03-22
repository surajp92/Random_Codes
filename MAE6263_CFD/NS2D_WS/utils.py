#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:51:22 2021

@author: suraj
"""
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt 
from rhs_schemes.compact_schemes_first_order_derivative import *

def compute_history(nx,ny,dx,dy,x,y,re,pr,s,w,th,input_data):
    
    cfl = input_data['cfl']
    sigma = input_data['sigma']
    lx = input_data['lx']
    ly = input_data['ly']
    ra = float(input_data['ra'])
    
    if input_data['icompact'] == 0 or input_data['icompact'] == 1:
        sx = c4d_p(s,dx,dy,nx,ny,'X')
        sy = c4d(s,dx,dy,nx,ny,'Y')
    elif input_data['icompact'] == 2:
        sx = c6d_p(s,dx,dy,nx,ny,'X') 
        sy = c6d_b5_d(s,dx,dy,nx,ny,'Y')  
    
    u = sy
    v = -sx
    
    umax = np.max(np.abs(u))
    vmax = np.max(np.abs(v))
    dt_cfl = min(cfl*dx/umax, cfl*dy/vmax)
    
    alpha  = 1.0/(dx**2) + 1.0/(dy**2) 
    dif_in = min(re,re*pr)
    dt_del = sigma*dif_in/alpha
    
    dt = min(dt_cfl,dt_del)
    
    # compute total energy
    en = 0.5*(u**2 + v**2)
    ene = simps(simps(en, y), x)/(lx*ly)
    
    # compute total enstrophy
    en = 0.5*(w**2)
    ens = simps(simps(en, y), x)/(lx*ly)
    
    # dissipation 
    dis = (2.0/re)*ens
    
    #  Nusselt number across walls
    thy = c4d(th,dx,dy,nx,ny,'Y')
    NuH = simps(thy[:,0], x)/lx # hot wall (lower)
    NuC = simps(thy[:,ny], x)/lx # cold wall (top)
    
    # mean Nusselt number
    vth = v*th
    vth_average = simps(simps(vth, y), x)/(lx*ly)
    Nu_mean = 1.0 + np.sqrt(ra*pr)*vth_average
    
    tprobe = np.zeros(5)
    tprobe[0] = th[int(nx/8), int(ny/8)]
    tprobe[1] = th[int(nx/4), int(3*ny/4)]
    tprobe[2] = th[int(nx/2), int(ny/2)]
    tprobe[3] = th[int(3*nx/4), int(ny/4)]
    tprobe[4] = th[int(7*nx/8), int(7*ny/8)]
    
    return dt, ene, ens, dis, NuH, NuC, Nu_mean, tprobe

def plot_residual_history(kc,rw,rs,rth,filename):
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.semilogy(kc, rw, label='$\omega$')
    ax.semilogy(kc, rs, label='$\psi$')
    ax.semilogy(kc, rth, label='$\Theta$')
    ax.legend()
    # plt.show()
    fig.savefig(filename, bbox_inches = 'tight', pad_inches = 0.1, dpi = 200)

def plot_turbulent_parameters(time,ene,ens,dis,NuMean,filename):
    fig, ax = plt.subplots(2,2,figsize=(8,6))
    axs = ax.flat
    axs[0].plot(time, ene, label='Energy')
    axs[1].plot(time, ens, label='Enstrophy')
    axs[2].plot(time, dis, label='Dissipation')
    axs[3].plot(time, NuMean, label='Mean Nusselt number')
    
    for i in range(4):
        axs[i].legend()

    # plt.show()
    fig.savefig(filename, bbox_inches = 'tight', pad_inches = 0.1, dpi = 200)    
    
def plot_probe_temperature(time, NuH, NuC, tprobe,filename):
    fig, ax = plt.subplots(1,2,figsize=(12,6))

    ax[0].plot(time, NuH, label='Hot wall Nusselt number')
    ax[0].plot(time, NuC, label='Cold wall Nusselt number')
    
    for i in range(4):
        ax[1].plot(time, tprobe[:,i], label=f'$\Theta_{i+1}$')
    
    for i in range(2):
        ax[i].legend()
    
    # plt.show()
    fig.savefig(filename, bbox_inches = 'tight', pad_inches = 0.1, dpi = 200)

def plot_contour(X,Y,w,s,th,ra,ra_max,filename):
    fig, axs = plt.subplots(3,1,figsize=(10,15))
    if ra < ra_max:
        cs = axs[0].contour(X,Y,w,20,colors='black')
    cs = axs[0].imshow(w.T,extent=[0, 2, 0, 1], origin='lower',
            interpolation='bicubic',cmap='RdBu_r', alpha=1.0,)
            # vmin=-20, vmax=20)
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[0], shrink=0.8, orientation='vertical')
    axs[0].set_aspect('equal')
    
    if ra < ra_max:
        cs = axs[1].contour(X,Y,s,20,colors='black')
    cs = axs[1].imshow(s.T,extent=[0, 2, 0, 1], origin='lower',
            interpolation='bicubic',cmap='RdBu_r', alpha=1.0)
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[1], shrink=0.8, orientation='vertical')
    axs[1].set_aspect('equal')
    
    if ra < ra_max:
        cs = axs[2].contour(X,Y,th,20,vmin=-0, vmax=1,colors='black')
    cs = axs[2].imshow(th.T,extent=[0, 2, 0, 1], origin='lower',
            interpolation='bicubic',cmap='RdBu_r', alpha=1.0)
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[2], shrink=0.8, orientation='vertical')
    axs[2].set_aspect('equal')
    
    # plt.show()
    fig.tight_layout()
    fig.savefig(filename, bbox_inches = 'tight', pad_inches = 0.1, dpi = 200)