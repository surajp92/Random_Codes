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

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

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
    p0 = p[pref]

    mask = np.logical_not( np.arange(0,nc) == pref )
    Phii = Phi[:,:,mask]
    pi = p[mask]

    
    print('Calculating Gammas...')
    Gamma = np.zeros((nx,nr,nc-1))
    for i in range(nc-1):
        templ = ( np.identity(nx) - Phi0.dot(Phi0H) )
        tempr = LA.inv( Phi0H.dot(Phii[:,:,i]) )
        temp = LA.multi_dot([templ,Phii[:,:,i],tempr])
               
        U, S, Vh = LA.svd(temp, full_matrices=False)
        S = np.diag(S)
        Gamma[:,:,i] = LA.multi_dot([U,np.arctan(S),Vh])
    
    print('Interpolating ...')
    alpha = np.ones(nc-1)
    GammaL = np.zeros((nx,nr))
    #% Lagrange Interpolation
    for i in range(nc-1):
        for j in range(nc-1):
            if (j != i) :
                alpha[i] = alpha[i]*(pTest-pi[j])/(pi[i]-pi[j])
    for i in range(nc-1):
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
nr = 10      #number of modes
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
ue = np.zeros((nx+1, ns+1, nc))

for p in range(0,nc):
    for n in range(0,ns+1):
        t[n] = dt*n
        for i in range(nx+1):
            x[i]=dx*i
            ue[i,n,p]=uexact(x[i],t[n],nu[p]) #snapshots from exact solution


#%% POD basis computation
nrs = 2*nr #number of modes for 'super-resolution' [Hybridization]
PHIs = np.zeros((nx+1,nrs,nc)) 
PHI = np.zeros((nx+1,nr,nc))        
       
L = np.zeros((ns+1,nc)) #Eigenvalues      
RIC = np.zeros((nc))    #Relative information content

print('Computing POD basis...')
for p in range(0,nc):
    u = ue[:,:,p]
    PHIs[:,:,p], L[:,p], RIC[p]  = POD(u, nrs) 
    PHI[:,:,p] = PHIs[:,:nr,p]
        
#%% Calculating true POD coefficients
ats = np.zeros((nc,ns+1,nrs))
at = np.zeros((nc,ns+1,nr))
print('Computing true POD coefficients...')
for p in range(nc):
    ats[p,:,:] = PODproj(ue[:,:,p],PHIs[:,:,p])
    at[p,:,:] = ats[p,:,:nr]
    
#%% LSTM using 1 parameter + 8 modes as input and 8 modes as ouput
# Removing old models
if os.path.isfile('burgers_integrator_LSTM.hd5'):
    os.remove('burgers_integrator_LSTM.hd5')
    
# Stacking data
a = np.zeros((nc,ns+1,nrs+1))
for p in range(nc):
    a[p,:,0] = nu[p]*np.ones(ns+1)
    a[p,:,1:] = ats[p,:,:]
   

# Create training data for LSTM
lookback = 3 #Number of lookbacks

for p in range(nc):
    xt, yt = create_training_data_lstm(a[p,:,:], ns+1, nrs+1, lookback)
    if p == 0:
        xtrain = xt
        ytrain = yt
    else:
        xtrain = np.vstack((xtrain,xt))
        ytrain = np.vstack((ytrain,yt))


# Scaling data
temp = a.reshape(-1,nrs+1)
scalerIn = MinMaxScaler(feature_range=(-1,1))
scalerIn = scalerIn.fit(temp[:-1,:])
scalerOut = MinMaxScaler(feature_range=(-1,1))
scalerOut = scalerOut.fit(temp[lookback:,1:])

m,n = ytrain.shape # m is number of training samples, n is number of output features [i.e., n=nr]

#for i in range(m):
    

# Shuffling data
perm = np.random.permutation(m)
xtrain = xtrain[perm,:,:]
ytrain = ytrain[perm,:]
          
# create the LSTM architecture
model = Sequential()
#model.add(Dropout(0.2))
model.add(LSTM(80, input_shape=(lookback, n+1), return_sequences=True, activation='tanh',kernel_initializer='glorot_normal'))
model.add(LSTM(80,  input_shape=(lookback, n+1), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
model.add(LSTM(80, input_shape=(lookback, n+1), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
#model.add(LSTM(40, input_shape=(lookback, n+1), return_sequences=True, activation='relu', kernel_initializer='uniform'))
#model.add(LSTM(40, input_shape=(lookback, n+1), return_sequences=True, activation='relu', kernel_initializer='uniform'))
model.add(LSTM(80, input_shape=(lookback, n+1), activation='tanh', kernel_initializer='uniform'))
model.add(Dense(n))

# compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# run the model
history = model.fit(xtrain, ytrain, epochs=1000, batch_size=64, validation_split=0.2)

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
ReTest = 500
nuTest = 1/ReTest
for n in range(ns+1):
   t[n] = dt*n
   for i in range(nx+1):
       x[i]=dx*i
       uTest[i,n]=uexact(x[i],t[n],nuTest) #snapshots from exact solution

#% POD basis computation     
print('Computing testing POD basis...')
PHItrue, Ltrue, RICtrue  = POD(uTest, nrs) 
        
#% Calculating true POD coefficients
print('Computing testing POD coefficients...')
aTrue = PODproj(uTest,PHItrue)


#%% Basis Interpolation
pref = 2 #Reference case in [0:nRe]
PHItest = GrassInt(PHIs,pref,nu,nuTest)
aTest = PODproj(uTest,PHItest)

#%% LSTM [Fully Nonintrusive]

# testing
testing_set = np.hstack(( nuTest*np.ones((ns+1,1)), aTest  ))
m,n = testing_set.shape
xtest = np.zeros((1,lookback,nrs+1))
aLSTM = np.zeros((ns+1,nrs))

# Initializing
for i in range(lookback):
    temp = testing_set[i,:]
    temp = temp.reshape(1,-1)
    xtest[0,i,:]  = temp
    aLSTM[i, :] = testing_set[i,1:] 

# Prediction
for i in range(lookback,ns+1):
    ytest = model.predict(xtest)
    aLSTM[i, :] = ytest
    for i in range(lookback-1):
        xtest[0,i,:] = xtest[0,i+1,:]
    xtest[0,lookback-1,1:] = ytest



fig, ax = plt.subplots(nrows=4,ncols=2,figsize=(9,9.5))
ax = ax.flat
for i in range(nrs):
    ax[i].plot(t,aTest[:,i],label=r'True Values')
    ax[i].plot(t,aLSTM[:,i],label=r'True Values')
    ax[i].set_xlabel(r'$t$',fontsize=14)
    ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)
    #ax[i].set_title(lbl, fontsize=14)


fig.subplots_adjust(bottom=0.15,hspace=0.7, wspace=0.4)
#fig.savefig('aLSTM.png', dpi = 400, bbox_inches = 'tight', pad_inches = 0)
fig.show()


 
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



