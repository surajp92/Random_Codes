#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 10:40:29 2021

@author: suraj
"""
import numpy as np
import matplotlib.pyplot as plt
import yaml

with open(r'ldc_parameters.yaml') as file:
    input_data = yaml.load(file, Loader=yaml.FullLoader)
    
file.close()

re = input_data['re']
nx = input_data['nx']
ny = input_data['ny']
lx = input_data['lx']
ly = input_data['ly']
U = input_data['U']
kmax = input_data['kmax']
iimax = input_data['iimax']
eps = float(input_data['eps'])

dx = lx/nx
dy = ly/ny

x = np.linspace(0,lx,nx+1)
y = np.linspace(0,ly,ny+1)

dtc = np.min((dx,dy))
dtv = 0.25*re*np.min((dx**2, dy**2))
sigma = 0.5
dt = sigma*np.min((dtc, dtv))

u = np.zeros((nx+1, ny+2))
v = np.zeros((nx+2, ny+1))
p = np.zeros((nx+2, ny+2))

u0 = np.zeros((nx, ny))
v0 = np.zeros((nx, ny))
p0 = np.zeros((nx, ny))

kc = [] 
ru = [] 
rv = [] 
rp = [] 

for k in range(kmax):
    u0[:,:] = np.copy(u[1:nx+1,1:ny+1])
    v0[:,:] = np.copy(v[1:nx+1,1:ny+1])
    p0[:,:] = np.copy(p[1:nx+1,1:ny+1])
    
    u[:,0] = -u[:,1] # bottom wall
    u[:,ny+1] = 2.0*U - u[:,ny] # top wall
    
    v[0,:] = -v[1,:] # left wall
    v[nx+1,:] = -v[nx,:] # right wall
    
    # x-momentum equation
    
    c3 = (1.0/re)*((u[2:nx+1,1:ny+1] - 2.0*u[1:nx,1:ny+1] + u[0:nx-1,1:ny+1])/(dx**2) + \
                   (u[1:nx,2:ny+2] - 2.0*u[1:nx,1:ny+1] + u[1:nx,0:ny])/(dy**2))
    
    c1 = (0.25/dx)*((u[2:nx+1,1:ny+1] + u[1:nx,1:ny+1])**2 - \
                    (u[0:nx-1,1:ny+1] + u[1:nx,1:ny+1])**2)
    
    c2 = (0.25/dy)*((u[1:nx,2:ny+2] + u[1:nx,1:ny+1])*(v[2:nx+1,1:ny+1] + v[1:nx,1:ny+1]) - \
                    (u[1:nx,0:ny] + u[1:nx,1:ny+1])*(v[2:nx+1,0:ny] + v[1:nx,0:ny]))
        
    u[1:nx,1:ny+1] = u[1:nx,1:ny+1] + dt*(-c1 -c2 + c3)
    
    # y-momentum equation
    
    c3 = (1.0/re)*((v[2:nx+2,1:ny] - 2.0*v[1:nx+1,1:ny] + v[0:nx,1:ny])/(dx**2) + \
                   (v[1:nx+1,2:ny+1] - 2.0*v[1:nx+1,1:ny] + v[1:nx+1,0:ny-1])/(dy**2))
    
    c2 = (0.25/dy)*((v[1:nx+1,2:ny+1] + v[1:nx+1,1:ny])**2 - \
                    (v[1:nx+1,0:ny-1] + v[1:nx+1,1:ny])**2)
    
    c1 = (0.25/dx)*((u[1:nx+1,2:ny+1] + u[1:nx+1,1:ny])*(v[2:nx+2,1:ny] + v[1:nx+1,1:ny]) - \
                    (u[0:nx,2:ny+1] + u[0:nx,1:ny])*(v[0:nx,1:ny] + v[1:nx+1,1:ny]))
    
    v[1:nx+1,1:ny] = v[1:nx+1,1:ny] + dt*(-c1 -c2 + c3)
    
    u[:,0] = -u[:,1] # bottom wall
    u[:,ny+1] = 2.0*U - u[:,ny] # top wall
    
    v[0,:] = -v[1,:] # left wall
    v[nx+1,:] = -v[nx,:] # right wall
    
    # compute pressure : Poisson equation
    
    p[1:nx+1,0] = p[1:nx+1,1]
    p[1:nx+1,ny+1] = p[1:nx+1,ny]
    
    p[0,0:ny+2] = p[1,0:ny+2]
    p[nx+1,0:ny+2] = p[nx,0:ny+2]
    
    a = -2.0/(dx**2) - 2.0/(dy**2)
    omega = 1.0
    
    for i in range(iimax):
        
        f = (1.0/dt)*((u[1:nx+1,1:ny+1] - u[0:nx,1:ny+1])/dx + \
                      (v[1:nx+1,1:ny+1] - v[1:nx+1,0:ny])/dy)
        
        r = f[:,:] - (p[2:nx+2,1:ny+1] - 2.0*p[1:nx+1,1:ny+1] + p[0:nx,1:ny+1])/(dx**2) - \
                     (p[1:nx+1,2:ny+2] - 2.0*p[1:nx+1,1:ny+1] + p[1:nx+1,0:ny])/(dy**2)
                     
        p[1:nx+1,1:ny+1] = p[1:nx+1,1:ny+1] + (omega/a)*r[:,:] 
        
    
    u[1:nx,1:ny+1] = u[1:nx,1:ny+1] - dt*(p[2:nx+1,1:ny+1] - p[1:nx,1:ny+1])/dx
    
    v[1:nx+1,1:ny] = v[1:nx+1,1:ny] - dt*(p[1:nx+1,2:ny+1] - p[1:nx+1,1:ny])/dy
    
    kc.append(k)
    ru.append(np.linalg.norm(u[1:nx+1,1:ny+1] - u0)/np.sqrt(nx*ny))
    rv.append(np.linalg.norm(v[1:nx+1,1:ny+1] - v0)/np.sqrt(nx*ny))
    rp.append(np.linalg.norm(p[1:nx+1,1:ny+1] - p0)/np.sqrt(nx*ny))
    
    print('%0.3i %0.3e %0.3e %0.3e' % (kc[k], ru[k], rv[k], rp[k]))
    
    if ru[k] <= eps and rv[k] <= eps and rp[k] <= eps:
        break
    

#%%
plt.semilogy(kc, ru)
plt.show()

#%%
X,Y = np.meshgrid(x,y)

plt.contourf(X,Y,u[:,1:ny+2].T, 60, cmap='jet', vmin=-0.2, vmax=1.5)
plt.colorbar()
plt.show()    

#%%
plt.plot(y,v[int(ny/2),:],'ro-')
plt.show()            
            
            
            
            
            
            
            
            
            
            