# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:41:10 2019

@author: Shady
"""
#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import simps

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

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


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
    ytrain = [training_set[i,:] for i in range(lookback,m)]
    ytrain = np.array(ytrain)    
    xtrain = np.zeros((m-lookback,lookback,n))
    for i in range(m-lookback):
        a = training_set[i,:]
        for j in range(1,lookback):
            a = np.vstack((a,training_set[i+j,:]))
        xtrain[i,:,:] = a
    return xtrain , ytrain

###############################################################################
# Burgers Routines
###############################################################################
def uexact(x, t, nu):  #Exact Solution [Sirisup]
    t0 = np.exp(1.0/(8.0*nu))
    uexact = (x/(t+1.0))/(1.0+np.sqrt((t+1.0)/t0)*np.exp(x*x/(4.0*nu*(t+1.0))))
    return uexact


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

###############################################################################
# Numerical Routines
###############################################################################
# Thomas algorithm for solving tridiagonal systems:    
def tdma(a, b, c, r, up, s, e):
    for i in range(s+1,e+1):
        b[i] = b[i] - a[i]/b[i-1]*c[i-1]
        r[i] = r[i] - a[i]/b[i-1]*r[i-1]   
    up[e] = r[e]/b[e]   
    for i in range(e-1,s-1,-1):
        up[i] = (r[i]-c[i]*up[i+1])/b[i]

# Computing first derivatives using the fourth order compact scheme:  
def pade4d(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    ud = np.zeros(n+1)
    i = 0
    b[i] = 1.0
    c[i] = 2.0
    r[i] = (-5.0*u[i] + 4.0*u[i+1] + u[i+2])/(2.0*h)
    for i in range(1,n):
        a[i] = 1.0
        b[i] = 4.0
        c[i] = 1.0
        r[i] = 3.0*(u[i+1] - u[i-1])/h
    i = n
    a[i] = 2.0
    b[i] = 1.0
    r[i] = (-5.0*u[i] + 4.0*u[i-1] + u[i-2])/(-2.0*h)
    tdma(a, b, c, r, ud, 0, n)
    return ud
    
# Computing second derivatives using the foruth order compact scheme:  
def pade4dd(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    udd = np.zeros(n+1)
    i = 0
    b[i] = 1.0
    c[i] = 11.0
    r[i] = (13.0*u[i] - 27.0*u[i+1] + 15.0*u[i+2] - u[i+3])/(h*h)
    for i in range(1,n):
        a[i] = 0.1
        b[i] = 1.0
        c[i] = 0.1
        r[i] = 1.2*(u[i+1] - 2.0*u[i] + u[i-1])/(h*h)
    i = n
    a[i] = 11.0
    b[i] = 1.0
    r[i] = (13.0*u[i] - 27.0*u[i-1] + 15.0*u[i-2] - u[i-3])/(h*h)
    
    tdma(a, b, c, r, udd, 0, n)
    return udd


#%% Main program:
    
# Inputs
nx =  1024  #spatial grid number
nc = 4     #number of control parameters (nu)
ns = 100    #number of snapshot per each Parameter 
nr = 8      #number of modes
Re_start = 100.0
Re_final = 1000.0
Re  = np.linspace(Re_start, Re_final, nc) #control Reynolds
nu = 1/Re   #control dissipation
lx = 1.0
dx = lx/nx
tm = 1.5
dt = tm/ns

#%% Data generation for training
x = np.linspace(0, lx, nx+1)
t = np.linspace(0, tm, ns+1)
um = np.zeros((nx+1, ns+1, nc))
up = np.zeros((nx+1, ns+1, nc))
up1 = np.zeros((nx+1, ns+1, nc))
uo = np.zeros((nx+1, ns+1, nc))

for p in range(0,nc):
    for n in range(0,ns+1):
        t[n] = dt*n
        for i in range(nx+1):
            x[i]=dx*i
            um[i,n,p]=uexact(x[i],t[n],nu[p]) #snapshots from unperturbed solution
            up[i,n,p]=0.1*um[i,n,p] #perturbation (unknown physics)
            #up[i,n,p]=0.1*np.sin(np.pi*x[i])*np.exp(nu[p]*t[n])/(2+np.cos(np.pi*x[i])*np.exp(nu[p]*t[n]))
            #up[i,n,p]= 0.1*(np.sin(np.pi*x[i]))*t[n]**2*x[i]
            #up[i,n,p]= 0.1*(np.sin(np.pi*x[i]))*np.exp(x[i]**2*t[n])
            uo[i,n,p]=um[i,n,p]+up[i,n,p] #snapshots from observed solution
            #uo[i,n,p]=uexact(x[i],t[n],1.2*nu[p]) # perturbation by using different parameter

#%% POD basis computation
PHI = np.zeros((nx+1,nr,nc))        
       
L = np.zeros((ns+1,nc)) #Eigenvalues      
RIC = np.zeros((nc))    #Relative information content

print('Computing POD basis...')
for p in range(0,nc):
    u = uo[:,:,p]
    PHI[:,:,p], L[:,p], RIC[p]  = POD(u, nr) 

#%% Calculating true POD coefficients (observed)
at = np.zeros((nc,ns+1,nr))
print('Computing true POD coefficients...')
for p in range(nc):
    at[p,:,:] = PODproj(uo[:,:,p],PHI[:,:,p])

#%% Calculating true POD coefficients (exact no perturbation)
PHIm = np.zeros((nx+1,nr,nc))        
Lm = np.zeros((ns+1,nc)) #Eigenvalues      
RICm = np.zeros((nc))    #Relative information content

print('Computing POD basis...')
for p in range(0,nc):
    u = um[:,:,p]
    PHIm[:,:,p], Lm[:,p], RICm[p]  = POD(u, nr) 
    
atm = np.zeros((nc,ns+1,nr))
print('Computing true POD coefficients...')
for p in range(nc):
    atm[p,:,:] = PODproj(um[:,:,p],PHIm[:,:,p])

#%% Galerkin projection [Fully Intrusive]

###############################
# Galerkin projection with nr
###############################
b_l = np.zeros((nr,nr,nc))
b_nl = np.zeros((nr,nr,nr,nc))
PHId = np.zeros((nx+1,nr,nc))
PHIdd = np.zeros((nx+1,nr,nc))

for p in range(nc):
    for i in range(nr):
        PHIdd[:,i,p] = pade4dd(PHI[:,i,p],dx,nx)
        PHId[:,i,p] = pade4d(PHI[:,i,p],dx,nx)

# linear term   
for p in range(nc):
    for k in range(nr):
        for i in range(nr):
            b_l[i,k,p] = nu[p]*np.dot(PHIdd[:,i,p].T , PHI[:,k,p]) 
                   
# nonlinear term 
for p in range(nc):
    for k in range(nr):
        for j in range(nr):
            for i in range(nr):
                temp = PHI[:,i,p]*PHId[:,j,p]
                b_nl[i,j,k,p] = - np.dot( temp.T, PHI[:,k,p] ) 

# solving ROM by Adams-Bashforth scheme          
aGP = np.zeros((nc,ns+1,nr))
for p in range(nc):
    aGP[p,0,:] = at[p,0,:nr]
    aGP[p,1,:] = at[p,1,:nr]
    aGP[p,2,:] = at[p,2,:nr]
    for k in range(3,ns+1):
        r1 = rhs(nr, b_l[:,:,p], b_nl[:,:,:,p], aGP[p,k-1,:])
        r2 = rhs(nr, b_l[:,:,p], b_nl[:,:,:,p], aGP[p,k-2,:])
        r3 = rhs(nr, b_l[:,:,p], b_nl[:,:,:,p], aGP[p,k-3,:])
        temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
        aGP[p,k,:] = aGP[p,k-1,:] + dt*temp 

#%%
def plot_data(t,at,aGP,atm):
    fig, ax = plt.subplots(nrows=5,ncols=2,figsize=(12,8))
    ax = ax.flat
    nrs = at.shape[1]
    
    for i in range(nrs):
        ax[i].plot(t,at[:,i],'k',label=r'True Values')
        ax[i].plot(t,atm[:,i],'b',label=r'Exact Values')
        ax[i].plot(t,aGP[:,i],'r',label=r'True Values')
        ax[i].set_xlabel(r'$t$',fontsize=14)
        ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
    
    fig.tight_layout()
    #fig.subplots_adjust(bottom=0.15)
    
    line_labels = ["Perturbation","No perturbation", "GP"]#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
    plt.show()
    fig.savefig('modes2.pdf')

plot_data(t,at[-1,:,:],aGP[-1,:,:],atm[-1,:,:])        

#%%
model = um[:,-1,-1]
observed = uo[:,-1,-1]
 
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(6,5))
ax.plot(x,model,'k',label=r'Model')
ax.plot(x,observed,'r',label=r'Observed')
ax.legend()
plt.show()

#%%
res_proj = at - aGP

#%% LSTM using 1 parameter + nr modes as input and nr modes as ouput
# Removing old models
if os.path.isfile('burgers_integrator_LSTM.hd5'):
    os.remove('burgers_integrator_LSTM.hd5')
    
# Stacking data
a = np.zeros((nc,ns+1,nr+1))
for p in range(nc):
    a[p,:,0] = nu[p]*np.ones(ns+1)
    a[p,:,1:] = res_proj[p,:,:]
   

# Create training data for LSTM
lookback = 3 #Number of lookbacks

# use xtrain from here
for p in range(nc):
    xt, yt = create_training_data_lstm(at[p,:,:], ns+1, nr, lookback)
    if p == 0:
        xtrain = xt
        ytrain = yt
    else:
        xtrain = np.vstack((xtrain,xt))
        ytrain = np.vstack((ytrain,yt))

data = xtrain

# use ytrain from here
for p in range(nc):
    xt, yt = create_training_data_lstm(res_proj[p,:,:], ns+1, nr, lookback)
    if p == 0:
        xtrain = xt
        ytrain = yt
    else:
        xtrain = np.vstack((xtrain,xt))
        ytrain = np.vstack((ytrain,yt))

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

xtrain, xvalid, ytrain, yvalid = train_test_split(data, labels, test_size=0.2 , shuffle= True)

#%%

m,n = ytrain.shape # m is number of training samples, n is number of output features [i.e., n=nr]

# create the LSTM architecture
model = Sequential()
#model.add(Dropout(0.2))
model.add(LSTM(60, input_shape=(lookback, n), return_sequences=True, activation='tanh'))
#model.add(LSTM(20,  input_shape=(lookback, n), return_sequences=True, activation='tanh'))
#model.add(LSTM(80, input_shape=(lookback, n+1), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
#model.add(LSTM(40, input_shape=(lookback, n+1), return_sequences=True, activation='relu', kernel_initializer='uniform'))
#model.add(LSTM(40, input_shape=(lookback, n+1), return_sequences=True, activation='relu', kernel_initializer='uniform'))
model.add(LSTM(60, input_shape=(lookback, n), activation='tanh'))
model.add(Dense(n))

# compile model
#model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

# run the model
history = model.fit(xtrain, ytrain, epochs=1000, batch_size=32, validation_data= (xvalid,yvalid))

# evaluate the model
scores = model.evaluate(xtrain, ytrain, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
epochs = range(1, len(loss) + 1)
plt.semilogy(epochs, loss, 'b', label='Training loss')
plt.semilogy(epochs, val_loss, 'r', label='Validationloss')
plt.title('Training and validation loss')
plt.legend()
filename = 'loss.png'
plt.savefig(filename, dpi = 400)
plt.show()

# Save the model
filename = 'burgers_integrator_LSTM.hd5'
model.save(filename)

#%% Testing
# Data generation for testing
x = np.linspace(0, lx, nx+1)
t = np.linspace(0, tm, ns+1)
uTest = np.zeros((nx+1, ns+1))
upTest = np.zeros((nx+1, ns+1))
uoTest = np.zeros((nx+1, ns+1))
ReTest = 500
nuTest = 1/ReTest
for n in range(ns+1):
   t[n] = dt*n
   for i in range(nx+1):
       x[i]=dx*i
       uTest[i,n]=uexact(x[i],t[n],nuTest) #snapshots from exact solution
       upTest[i,n]=0.1*uTest[i,n]
       uoTest[i,n] = uTest[i,n] + upTest[i,n]

#% POD basis computation     
print('Computing testing POD basis...')
PHItrue, Ltrue, RICtrue  = POD(uoTest, nr) 
        
#% Calculating true POD coefficients
print('Computing testing POD coefficients...')
aTrue = PODproj(uoTest,PHItrue)

#%% Basis Interpolation
pref = 2 #Reference case in [0:nRe]
PHItest = GrassInt(PHI,pref,nu,nuTest)
aTest = PODproj(uoTest,PHItest)

#%%
b_l = np.zeros((nr,nr))
b_nl = np.zeros((nr,nr,nr))
PHId = np.zeros((nx+1,nr))
PHIdd = np.zeros((nx+1,nr))

for i in range(nr):
    PHIdd[:,i] = pade4dd(PHItest[:,i],dx,nx)
    PHId[:,i] = pade4d(PHItest[:,i],dx,nx)

# linear term   
for k in range(nr):
    for i in range(nr):
        b_l[i,k] = nuTest*np.dot(PHIdd[:,i].T , PHItest[:,k]) 
                   
# nonlinear term 
for k in range(nr):
    for j in range(nr):
        for i in range(nr):
            temp = PHItest[:,i]*PHId[:,j]
            b_nl[i,j,k] = - np.dot( temp.T, PHItest[:,k] ) 

# solving ROM by Adams-Bashforth scheme          
aGPtest = np.zeros((ns+1,nr))

aGPtest[0,:] = aTest[0,:nr]
aGPtest[1,:] = aTest[1,:nr]
aGPtest[2,:] = aTest[2,:nr]

for k in range(3,ns+1):
    r1 = rhs(nr, b_l[:,:], b_nl[:,:,:], aGPtest[k-1,:])
    r2 = rhs(nr, b_l[:,:], b_nl[:,:,:], aGPtest[k-2,:])
    r3 = rhs(nr, b_l[:,:], b_nl[:,:,:], aGPtest[k-3,:])
    temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
    aGPtest[k,:] = aGPtest[k-1,:] + dt*temp 
        
#%%
def plot_data(x,phiTrue,phiTest,phi):
    fig, ax = plt.subplots(nrows=5,ncols=2,figsize=(12,8))
    ax = ax.flat
    nrs = phiTrue.shape[1]
    
    for i in range(nrs):
        ax[i].plot(x,phiTrue[:,i],'k',label=r'True Values')
        ax[i].plot(x,phiTest[:,i],'b--',label=r'Exact Values')
        ax[i].plot(x,phi[:,i],'r-.',label=r'Exact Values')
        ax[i].set_xlabel(r'$t$',fontsize=14)
        ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
    
    fig.tight_layout()
    #fig.subplots_adjust(bottom=0.15)
    
    line_labels = ["True","Grasman","Train"]#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
    plt.show()
    fig.savefig('grasman.pdf')

plot_data(x,PHItrue,PHItest,PHI[:,:,1]) 
#plot_data(x,PHItest,PHItest2,PHI[:,:,1]) 

#%% LSTM [Fully Nonintrusive]
# testing
testing_set = aTest
m,n = testing_set.shape
xtest = np.zeros((1,lookback,nr))
rLSTM = np.zeros((ns+1,nr))
aGPml = np.zeros((ns+1,nr))

#%%
# Initializing
for i in range(lookback):
    temp = testing_set[i,:]
    temp = temp.reshape(1,-1)
    xtest[0,i,:]  = temp
    rLSTM[i, :] = testing_set[i,:] - aGPtest[i,:] 
    aGPml[i,:] = testing_set[i,:]

#%%
# Prediction
for i in range(lookback,ns+1):
    xtest_sc = scalerIn.transform(xtest[0])
    xtest_sc = xtest_sc.reshape(1,lookback,nr)
    ytest_sc = model.predict(xtest_sc)
    ytest = scalerOut.inverse_transform(ytest_sc)
    rLSTM[i, :] = ytest
        
    for k in range(lookback-1):
        xtest[0,k,:] = xtest[0,k+1,:]
    xtest[0,lookback-1,:] = ytest + aGPtest[i,:]
    aGPml[i,:] = ytest + aGPtest[i,:]

#%%
# solving ROM by Adams-Bashforth scheme       
'''
aCtest = np.zeros((ns+1,nr))
aCtest[0,:] = 0.0
aCtest[1,:] = 0.0
aCtest[2,:] = 0.0
for k in range(3,ns+1):
    r1 = rhs(nr, b_l[:,:], b_nl[:,:,:], aTest[k-1,:])
    r2 = rhs(nr, b_l[:,:], b_nl[:,:,:], aTest[k-2,:])
    r3 = rhs(nr, b_l[:,:], b_nl[:,:,:], aTest[k-3,:])
    temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
    aCtest[k,:] = aTest[k,:] - (aTest[k-1,:] + dt*temp)
''' 

#%%
#aGPml = aGPtest + rLSTM

def plot_data(x,aTrue,aTest,aGPml):
    fig, ax = plt.subplots(nrows=5,ncols=2,figsize=(12,8))
    ax = ax.flat
    nrs = aTrue.shape[1]
    
    for i in range(nrs):
        ax[i].plot(x,aTrue[:,i],'k',label=r'True Values')
        ax[i].plot(x,aTest[:,i],'b--',label=r'Exact Values')
        ax[i].plot(x,aGPml[:,i],'r-.',label=r'Exact Values')
        ax[i].set_xlabel(r'$t$',fontsize=14)
        ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
    
    fig.tight_layout()
    #fig.subplots_adjust(bottom=0.15)
    
    line_labels = ["Grasman","GP","GP-ML"]#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
    plt.show()
    fig.savefig('hybrid_res.pdf')

plot_data(t,aTest,aGPtest, aGPml) 


 
#%% Galerkin projection [Fully Intrusive]

###############################
# Galerkin projection with nr
###############################
b_l = np.zeros((nr,nr))
b_nl = np.zeros((nr,nr,nr))
PHId = np.zeros((nx+1,nr))
PHIdd = np.zeros((nx+1,nr))

for i in range(nr):
    PHIdd[:,i] = pade4dd(PHItest[:,i],dx,nx)
    PHId[:,i] = pade4d(PHItest[:,i],dx,nx)

# linear term   
for k in range(nr):
    for i in range(nr):
        b_l[i,k] = nuTest*np.dot(PHIdd[:,i].T , PHItest[:,k]) 
                   
# nonlinear term 
for k in range(nr):
    for j in range(nr):
        for i in range(nr):
            temp = PHItest[:,i]*PHId[:,j]
            b_nl[i,j,k] = - np.dot( temp.T, PHItest[:,k] ) 

# solving ROM             
aGP = np.zeros((ns+1,nr))
aGP[0,:] = aTest[0,:nr]
aGP[1,:] = aTest[1,:nr]
aGP[2,:] = aTest[2,:nr]
for k in range(3,ns+1):
    r1 = rhs(nr, b_l, b_nl, aGP[k-1,:])
    r2 = rhs(nr, b_l, b_nl, aGP[k-2,:])
    r3 = rhs(nr, b_l, b_nl, aGP[k-3,:])
    temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
    aGP[k,:] = aGP[k-1,:] + dt*temp 

###############################
# Galerkin projection with nrs
###############################
b_l = np.zeros((nrs,nrs))
b_nl = np.zeros((nrs,nrs,nrs))
PHId = np.zeros((nx+1,nrs))
PHIdd = np.zeros((nx+1,nrs))

for i in range(nrs):
    PHIdd[:,i] = pade4dd(PHItest[:,i],dx,nx)
    PHId[:,i] = pade4d(PHItest[:,i],dx,nx)

# linear term   
for k in range(nrs):
    for i in range(nrs):
        b_l[i,k] = nuTest*np.dot(PHIdd[:,i].T , PHItest[:,k]) 
                   
# nonlinear term 
for k in range(nrs):
    for j in range(nrs):
        for i in range(nrs):
            temp = PHItest[:,i]*PHId[:,j]
            b_nl[i,j,k] = - np.dot( temp.T, PHItest[:,k] ) 

# solving ROM             
aGPs = np.zeros((ns+1,nrs))
aGPs[0,:] = aTest[0,:nrs]
aGPs[1,:] = aTest[1,:nrs]
aGPs[2,:] = aTest[2,:nrs]
for k in range(3,ns+1):
    r1 = rhs(nrs, b_l, b_nl, aGPs[k-1,:])
    r2 = rhs(nrs, b_l, b_nl, aGPs[k-2,:])
    r3 = rhs(nrs, b_l, b_nl, aGPs[k-3,:])
    temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
    aGPs[k,:] = aGPs[k-1,:] + dt*temp           

#%% Galerkin Projection + LSTM [Hybrid]
aHyb = np.hstack(( aGP , aLSTM[:,nr:] ))
    

##%%
#uPOD = PODrec(at[:,:2*nr,0],PHIt[:,:2*nr,0])
#uGP = PODrec(aGP[:,:,0],PHI[:,:,0])
#uhybrid = PODrec(aGP[:,:,0],PHI[:,:,0]) + PODrec(at[:,nr:2*nr,0],PHIt[:,nr:2*nr,0])
#uex  = ue[:,-1,0]



