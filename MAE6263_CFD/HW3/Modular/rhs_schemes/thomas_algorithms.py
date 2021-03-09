#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:24:39 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt


def tdma(a,b,c,r,s,e):
    
    a_ = np.copy(a)
    b_ = np.copy(b)
    c_ = np.copy(c)
    r_ = np.copy(r)
    
    un = np.zeros((np.shape(r)[0],np.shape(r)[1]))
    
    for i in range(s+1,e+1):
        b_[i,:] = b_[i,:] - a_[i,:]*(c_[i-1,:]/b_[i-1,:])
        r_[i,:] = r_[i,:] - a_[i,:]*(r_[i-1,:]/b_[i-1,:])
        
    un[e,:] = r_[e,:]/b_[e,:]
    
    for i in range(e-1,s-1,-1):
        un[i,:] = (r_[i,:] - c_[i,:]*un[i+1,:])/b_[i,:]
    
    del a_, b_, c_, r_
    
    return un


if __name__ == "__main__":
    
    A = np.array([[10,2,0,0],[3,10,4,0],[0,1,7,5],[0,0,3,4]],dtype=float)   

    a = np.array([0.,3.,1,3]) 
    b = np.array([10.,10.,7.,4.])
    c = np.array([2.,4.,5.,0])
    d = np.array([3,4,5,6.])
    
    a = np.reshape(a,[-1,1])
    b = np.reshape(b,[-1,1])
    c = np.reshape(c,[-1,1])
    d = np.reshape(d,[-1,1])
    
    print(tdma(a, b, c, d, 0, 3) - np.linalg.solve(A, d))
    
    # print(np.linalg.solve(A, d))
    
    # xl = -1.0
    # xr = 1.0
    # dx = 0.025
    # nx = int((xr - xl)/dx)
    
    # dt = 0.0025
    # tmax = 1.0
    # nt = int(tmax/dt)
    
    # ny = nx
    
    # x = np.linspace(xl,xr,nx+1)
    # xx = (np.ones((ny+1,1))*x).T
    
    # t = np.linspace(0,tmax,nt+1)
    # u = np.zeros((nt+1,nx+1,ny+1))
    # a = np.zeros((nx+1,ny+1))
    # b = np.zeros((nx+1,ny+1))
    # c = np.zeros((nx+1,ny+1))
    # r = np.zeros((nx+1,ny+1))
    
    # k = 0
    # u[k,:,:] = -np.sin(np.pi*xx)
    
    # exact = lambda x,t : -np.exp(-t)*np.sin(np.pi*x)
    
    # ue = exact(xx,tmax)
    
    # start = 1
    # end = nx-1
    
    # i = 0
    # a[i,:],b[i,:],c[i,:],r[i,:] = 0.0, 1.0, 0.0, exact(x[i],0) 
    
    # i = nx
    # a[i,:],b[i,:],c[i,:],r[i,:] = 0.0, 1.0, 0.0, exact(x[i],0) 
    
    # alpha = 1.0/(np.pi**2)
    
    # ii = np.arange(1,nx)
    
    # a[ii,:] = 12.0/(dx*dx) - 2.0/(alpha*dt)
    # b[ii,:] = -24.0/(dx*dx) - 20.0/(alpha*dt)
    # c[ii,:] = 12.0/(dx*dx) - 2.0/(alpha*dt)
    
    # for k in range(1,nt+1):
    #     i = 0
    #     r[i,:] = exact(x[i],t[k]) 
              
    #     r[ii,:] = -(2.0/(alpha*dt))*(u[k-1,ii+1,:] + 10.0*u[k-1,ii,:] + u[k-1,ii-1,:]) - \
    #              (12.0/(dx**2))*(u[k-1,ii+1,:] - 2.0*u[k-1,ii,:] + u[k-1,ii-1,:])
        
    #     i = nx
    #     r[i,:] = exact(x[i],t[k]) 
        
    #     u[k,:,:] = tdma(a,b,c,r,start,end)
    
    # uerror = u[-1,:,:] - ue
    # l2 = np.linalg.norm(uerror)/np.sqrt(np.size(uerror))
    
    # print(f'L2 norm of the error = {l2}')
    
    # plt.plot(x,ue)
    # plt.plot(x,u[-1,:],'--')
    # plt.show()
    
    # plt.plot(x,np.abs((ue-u[-1,:])), 'bo-')
    # plt.show()
        
        
            
    
    