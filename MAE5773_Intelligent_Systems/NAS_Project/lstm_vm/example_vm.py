# -*- coding: utf-8 -*-
"""Example.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EzTMRhEofQY_VjsbNJGL1kCKVSy_24LG

## Load toy data and create evaluation function
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import simps
import pyfftw

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import pandas as pd
import time as clck
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import keras.backend as K
K.set_floatx('float64')

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

#%% Define Functions

###############################################################################
#POD Routines
###############################################################################         
def POD(u,R): #Basis Construction
    n,ns = u.shape
    U,S,Vh = LA.svd(u, full_matrices=False)
    Phi = U[:,:R]  
    L = S**2
    #compute RIC (relative inportance index)
    RIC = sum(L[:R])/sum(L)*100   
    return Phi,L,RIC

def PODproj(u,Phi): #Projection
    a = np.dot(u.T,Phi)  # u = Phi * a.T
    return a

def PODrec(a,Phi): #Reconstruction    
    u = np.dot(Phi,a.T)    
    return u


###############################################################################
#Interpolation Routines
###############################################################################  
# Grassmann Interpolation
def GrassInt(Phi,pref,p,pTest):
    # Phi is input basis [training]
    # pref is the reference basis [arbitrarty] for Grassmann interpolation
    # p is the set of training parameters
    # pTest is the testing parameter
    
    nx,nr,nc = Phi.shape
    Phi0 = Phi[:,:,pref] 
    Phi0H = Phi0.T 

    print('Calculating Gammas...')
    Gamma = np.zeros((nx,nr,nc))
    for i in range(nc):
        templ = ( np.identity(nx) - Phi0.dot(Phi0H) )
        tempr = LA.inv( Phi0H.dot(Phi[:,:,i]) )
        temp = LA.multi_dot([templ,Phi[:,:,i],tempr])
               
        U, S, Vh = LA.svd(temp, full_matrices=False)
        S = np.diag(S)
        Gamma[:,:,i] = LA.multi_dot([U,np.arctan(S),Vh])
    
    print('Interpolating ...')
    alpha = np.ones(nc)
    GammaL = np.zeros((nx,nr))
    #% Lagrange Interpolation
    for i in range(nc):
        for j in range(nc):
            if (j != i) :
                alpha[i] = alpha[i]*(pTest-p[j])/(p[i]-p[j])
    for i in range(nc):
        GammaL = GammaL + alpha[i] * Gamma[:,:,i]
            
    U, S, Vh = LA.svd(GammaL, full_matrices=False)
    PhiL = LA.multi_dot([ Phi0 , Vh.T ,np.diag(np.cos(S)) ]) + \
           LA.multi_dot([ U , np.diag(np.sin(S)) ])
    PhiL = PhiL.dot(Vh)
    return PhiL

###############################################################################
#LSTM Routines
############################################################################### 
def create_training_data_lstm(training_set, m, n, lookback):
    ytrain = [training_set[i,1:] for i in range(lookback,m)]
    ytrain = np.array(ytrain)    
    xtrain = np.zeros((m-lookback,lookback,n))
    for i in range(m-lookback):
        a = training_set[i,:]
        for j in range(1,lookback):
            a = np.vstack((a,training_set[i+j,:]))
        xtrain[i,:,:] = a
    return xtrain , ytrain

def rhs(nr, b_l, b_nl, a): # Right Handside of Galerkin Projection
    r2, r3, r = [np.zeros(nr) for _ in range(3)]
    
    for k in range(nr):
        r2[k] = 0.0
        for i in range(nr):
            r2[k] = r2[k] + b_l[i,k]*a[i]
    
    for k in range(nr):
        r3[k] = 0.0
        for j in range(nr):
            for i in range(nr):
                r3[k] = r3[k] + b_nl[i,j,k]*a[i]*a[j]
    
    r = r2 + r3    
    return r

def plot_3d_surface(x,t,field):
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(t, x)
    
    surf = ax.plot_surface(Y, X, field, cmap=plt.cm.viridis,
                           linewidth=1, antialiased=False,rstride=1,
                            cstride=1)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    fig.tight_layout()
    plt.show()
    fig.savefig('3d.pdf')

#%% fast poisson solver using second-order central difference scheme
def fpsi(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    aa = -2.0/(dx*dx) - 2.0/(dy*dy)
    bb = 2.0/(dx*dx)
    cc = 2.0/(dy*dy)
    hx = 2.0*np.pi/np.float64(nx)
    hy = 2.0*np.pi/np.float64(ny)
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[:] = hx*np.float64(np.arange(0, nx))

    ky[:] = hy*np.float64(np.arange(0, ny))
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(np.cos(kx), np.cos(ky), indexing='ij')
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
    
    data[:,:] = np.vectorize(complex)(f,0.0)

    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    e = fft_object(data)
    #e = pyfftw.interfaces.scipy_fftpack.fft2(data)
    
    e[0,0] = 0.0
    
    data1[:,:] = e[:,:]/(aa + bb*kx[:,:] + cc*ky[:,:])

    ut = np.real(fft_object_inv(data1))
        
    return ut

#%%
def nonlinear_term(nx,ny,dx,dy,wf,sf):
    '''
    this function returns -(Jacobian)
    
    '''
    w = np.zeros((nx+3,ny+3))
    
    w[1:nx+1,1:ny+1] = wf
    
    # periodic
    w[:,ny+1] = w[:,1]
    w[nx+1,:] = w[1,:]
    w[nx+1,ny+1] = w[1,1]
    
    # ghost points
    w[:,0] = w[:,ny]
    w[:,ny+2] = w[:,2]
    w[0,:] = w[nx,:]
    w[nx+2,:] = w[2,:]
    
    s = np.zeros((nx+3,ny+3))
    
    s[1:nx+1,1:ny+1] = sf
    
    # periodic
    s[:,ny+1] = s[:,1]
    s[nx+1,:] = s[1,:]
    s[nx+1,ny+1] = s[1,1]
    
    # ghost points
    s[:,0] = s[:,ny]
    s[:,ny+2] = s[:,2]
    s[0,:] = s[nx,:]
    s[nx+2,:] = s[2,:]
    
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    
    f = np.zeros((nx+1,ny+1))
    
    #Arakawa
    j1 = gg*( (w[2:nx+3,1:ny+2]-w[0:nx+1,1:ny+2])*(s[1:nx+2,2:ny+3]-s[1:nx+2,0:ny+1]) \
             -(w[1:nx+2,2:ny+3]-w[1:nx+2,0:ny+1])*(s[2:nx+3,1:ny+2]-s[0:nx+1,1:ny+2]))

    j2 = gg*( w[2:nx+3,1:ny+2]*(s[2:nx+3,2:ny+3]-s[2:nx+3,0:ny+1]) \
            - w[0:nx+1,1:ny+2]*(s[0:nx+1,2:ny+3]-s[0:nx+1,0:ny+1]) \
            - w[1:nx+2,2:ny+3]*(s[2:nx+3,2:ny+3]-s[0:nx+1,2:ny+3]) \
            + w[1:nx+2,0:ny+1]*(s[2:nx+3,0:ny+1]-s[0:nx+1,0:ny+1]))

    j3 = gg*( w[2:nx+3,2:ny+3]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
            - w[0:nx+1,0:ny+1]*(s[0:nx+1,1:ny+2]-s[1:nx+2,0:ny+1]) \
            - w[0:nx+1,2:ny+3]*(s[1:nx+2,2:ny+3]-s[0:nx+1,1:ny+2]) \
            + w[2:nx+3,0:ny+1]*(s[2:nx+3,1:ny+2]-s[1:nx+2,0:ny+1]) )

    f = -(j1+j2+j3)*hh
                  
    return f[1:nx+1,1:ny+1]

def linear_term(nx,ny,dx,dy,re,f):
    w = np.zeros((nx+3,ny+3))
    
    w[1:nx+1,1:ny+1] = f
    
    # periodic
    w[:,ny+1] = w[:,1]
    w[nx+1,:] = w[1,:]
    w[nx+1,ny+1] = w[1,1]
    
    # ghost points
    w[:,0] = w[:,ny]
    w[:,ny+2] = w[:,2]
    w[0,:] = w[nx,:]
    w[nx+2,:] = w[2,:]
    
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    
    f = np.zeros((nx+1,ny+1))
    
    lap = aa*(w[2:nx+3,1:ny+2]-2.0*w[1:nx+2,1:ny+2]+w[0:nx+1,1:ny+2]) \
        + bb*(w[1:nx+2,2:ny+3]-2.0*w[1:nx+2,1:ny+2]+w[1:nx+2,0:ny+1])
    
    f = lap/re
            
    return f[1:nx+1,1:ny+1]

def pbc(w):
    f = np.zeros((nx+1,ny+1))
    f[:nx,:ny] = w
    f[:,ny] = f[:,0]
    f[nx,:] = f[0,:]
    
    return f

#%% Main program:
# Inputs
nx =  128  #spatial grid number
ny = 128
nc = 4     #number of control parameters (nu)
ns = 200    #number of snapshot per each Parameter 
nr = 8      #number of modes
Re_start = 200.0
Re_final = 800.0
Re  = np.linspace(Re_start, Re_final, nc) #control Reynolds
nu = 1/Re   #control dissipation
lx = 2.0*np.pi
ly = 2.0*np.pi
dx = lx/nx
dy = ly/ny
dt = 1e-1
tm = 20.0
lookback = 6 #Number of lookbacks
noise = 0.0

ReTest = 500

#%% Data generation for training
x = np.linspace(0, lx, nx+1)
y = np.linspace(0, ly, ny+1)
t = np.linspace(0, tm, ns+1)

um = np.zeros(((nx)*(ny), ns+1, nc))
up = np.zeros(((nx)*(ny), ns+1, nc))
uo = np.zeros(((nx)*(ny), ns+1, nc))

for p in range(0,nc):
    for n in range(0,ns+1):
        file_input = "./snapshots/Re_"+str(int(Re[p]))+"/w/w_"+str(int(n))+ ".csv"
        w = np.genfromtxt(file_input, delimiter=',')
        
        w1 = w[1:nx+1,1:ny+1]
        
        um[:,n,p] = np.reshape(w1,(nx)*(ny)) #snapshots from unperturbed solution
        up[:,n,p] = noise*um[:,n,p] #perturbation (unknown physics)
        uo[:,n,p] = um[:,n,p] + up[:,n,p] #snapshots from observed solution

#plot_3d_surface(x,t,uo[:,:,-1])

#%% POD basis computation
PHIw = np.zeros(((nx)*(ny),nr,nc))
PHIs = np.zeros(((nx)*(ny),nr,nc))        
       
L = np.zeros((ns+1,nc)) #Eigenvalues      
RIC = np.zeros((nc))    #Relative information content

print('Computing POD basis for vorticity ...')
for p in range(0,nc):
    u = uo[:,:,p]
    PHIw[:,:,p], L[:,p], RIC[p]  = POD(u, nr) 

#%%    
print('Computing POD basis for streamfunction ...')
for p in range(0,nc):
    for i in range(nr):
        phi_w = np.reshape(PHIw[:,i,p],[nx,ny])
        phi_s = fpsi(nx, ny, dx, dy, -phi_w)
        PHIs[:,i,p] = np.reshape(phi_s,(nx)*(ny))

#%% Calculating true POD coefficients (observed)
at = np.zeros((ns+1,nr,nc))
print('Computing true POD coefficients...')
for p in range(nc):
    at[:,:,p] = PODproj(uo[:,:,p],PHIw[:,:,p])

print('Reconstructing with true coefficients')
w = PODrec(at[:,:,1],PHIw[:,:,1])

w = w[:,-1]
w = np.reshape(w,[nx,ny])

fig, ax = plt.subplots(1,1,sharey=True,figsize=(5,4))
cs = ax.contourf(x[:nx],y[:ny],w.T, 120, cmap = 'jet')
ax.set_aspect(1.0)

fig.colorbar(cs,orientation='vertical')
fig.tight_layout() 
plt.show()
fig.savefig("reconstructed.png", bbox_inches = 'tight')

# Stacking data
a = np.zeros((ns+1,nr+1,nc))
for p in range(nc):
    a[:,0,p] = nu[p]*np.ones(ns+1)
    a[:,1:,p] = at[:,:,p]
   

# Create training data for LSTM
for p in range(nc):
    xt, yt = create_training_data_lstm(a[:,:,p], ns+1, nr+1, lookback)
    if p == 0:
        xtrain = xt
        ytrain = yt
    else:
        xtrain = np.vstack((xtrain,xt))
        ytrain = np.vstack((ytrain,yt))

data = xtrain # modified GP as the input data
labels = ytrain
   
#%%
# Scaling data
p,q,r = data.shape
data2d = data.reshape(p*q,r)

scalerIn = MinMaxScaler(feature_range=(-1,1))
scalerIn = scalerIn.fit(data2d)
data2d = scalerIn.transform(data2d)
data = data2d.reshape(p,q,r)

scalerOut = MinMaxScaler(feature_range=(-1,1))
scalerOut = scalerOut.fit(labels)
labels = scalerOut.transform(labels)

X, y = data, labels

#%%
def coeff_determination(y_true, y_pred):
    SS_res =  np.sum(np.square( y_true-y_pred ))
    SS_tot = np.sum(np.square( y_true - np.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + 1.0e-7) )

def evaluate_model(learner, X, y, num_folds):
    r2 = 0
    for train_ind, val_ind in KFold(n_splits=num_folds).split(X, y):
        learner.fit(X[train_ind, :], y[train_ind], epochs=20, batch_size=64)
        r2 += coeff_determination(learner.predict(X[val_ind, :]), y[val_ind])

    print("R2:", r2/num_folds)

def lstm_model(n_layers=2,n_cells=40,act_func=1,initializer=1,optimizer=1):
    lookback = 6
    n = 9
    
    act_func_dict = {1:'tanh',2:'relu',}
    initializer_dict = {1:'uniform',2:'glorot_normal',3:'random_normal'}
    optimizer_dict = {1:'adam',2:'rmsprop',3:'SGD'}
    
    model = Sequential()
    
    for k in range(n_layers-1):
        model.add(LSTM(n_cells, input_shape=(lookback, n), return_sequences=True, activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer]))
        
    model.add(LSTM(n_cells, input_shape=(lookback, n), activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer]))
    model.add(Dense(n-1,activation='linear'))
    
    model.compile(loss='mean_squared_error', optimizer=optimizer_dict[optimizer])
#    model.summary()
    
    return model

#learner = lstm_model(n_layers=2,n_cells=40) 
#evaluate_model(learner, X, y, 3)

#%%
from param import ContinuousParam, CategoricalParam, ConstantParam
from genetic_hyperopt import GeneticHyperopt

optimizer = GeneticHyperopt(lstm_model, X, y, mean_squared_error, maximize=False)

n_layers_param = ContinuousParam("n_layers", 4, 2, min_limit=2, max_limit=6, is_int=True)
n_cells_param = ContinuousParam("n_cells", 40, 10, min_limit=20, max_limit=60, is_int=True)
act_func_param = ContinuousParam("act_func", 1, 1, min_limit=1, max_limit=2, is_int=True)
initializer_param = ContinuousParam("initializer", 2, 1, min_limit=1, max_limit=3, is_int=True)
optimizer_param = ContinuousParam("optimizer", 2, 1, min_limit=1, max_limit=3, is_int=True)

optimizer.add_param(n_layers_param)
optimizer.add_param(n_cells_param)
optimizer.add_param(act_func_param)
optimizer.add_param(initializer_param)
optimizer.add_param(optimizer_param)

best_params, best_score, plotting_stats, best_param_dict = optimizer.evolve()

np.savez('results_vm1.npz',plotting_stats=plotting_stats,best_param_dict=best_param_dict)

#%%
labels = [i+1 for i in range(plotting_stats.shape[0])]
all_data = [plotting_stats[i,1:] for i in range(plotting_stats.shape[0])]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

# rectangular box plot
bplot1 = axes.boxplot(all_data,
                      vert=True,  # vertical box alignment
                      patch_artist=False,  # fill with color
                      labels=labels, # will be used to label x-ticks
                      showmeans=True)  

axes.set_title('Rectangular box plot')

axes.yaxis.grid(True)
axes.set_xlabel('Three separate samples')
axes.set_ylabel('Observed values')

fig.tight_layout()
plt.show()
fig.savefig('boxplot.png')
