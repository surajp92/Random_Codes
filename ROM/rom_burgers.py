#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:26:40 2019
@author: Suraj Pawar
"""
#%% POD-GP code
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import simps

nx =  1024 # spatial grid number
re_start = 100.0
re_final = 1000.0
n_re = 10
tmax = 1.0
lx = 1.0
n_snap = 100 #number of snapshot per each Re 
n_s = (n_snap+1)*n_re #total number of snapshots in paramatric-space
n_modes = 6
nt = 1000

dx = lx/nx
dt_snap = tmax/n_snap

def rhs(n_modes, b_c, b_l, b_nl, a):
    r2 = np.zeros(n_modes)
    r3 = np.zeros(n_modes)
    r = np.zeros(n_modes)
    
    for k in range(n_modes):
        r2[k] = 0.0
        for i in range(n_modes):
            r2[k] = r2[k] + b_l[i,k]*a[i]
    
    for k in range(n_modes):
        r3[k] = 0.0
        for j in range(n_modes):
            for i in range(n_modes):
                r3[k] = r3[k] + b_nl[i,j,k]*a[i]*a[j]
    
    r = b_c + r2 + r3    
    return r
    
def tdma(a, b, c, r, up, s, e):
    for i in range(s+1,e+1):
        b[i] = b[i] - a[i]/b[i-1]*c[i-1]
        r[i] = r[i] - a[i]/b[i-1]*r[i-1]
    
    up[e] = r[e]/b[e]
    
    for i in range(e-1,s-1,-1):
        up[i] = (r[i]-c[i]*up[i+1])/b[i]
    
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

def uexact(x, t, nu):
    temp = np.exp(1.0/(8.0*nu))
    uexact = (x/(t+1.0))/(1.0+np.sqrt((t+1.0)/temp)*np.exp(x*x/(4.0*nu*(t+1.0))))
    return uexact


#generate snapshot data from exact solution
x = np.zeros(nx+1)
t = np.zeros(n_snap+1)
nu = np.zeros(n_re)
ue = np.zeros((nx+1,n_snap+1,n_re))
for p in range(0,n_re):
    re = re_start + p*(re_final-re_start)/(n_re-1)
    nu[p] = 1.0/re
    for n in range(0,n_snap+1):
        t[n] = dt_snap*n
        for i in range(nx+1):
            x[i]=dx*i
            ue[i,n,p]=uexact(x[i],t[n],nu[p])

#generate snapshot data in paramatric-space
us = np.zeros((nx+1,n_s))
for p in range(0,n_re):
    for n in range(0,n_snap+1):
        for i in range(nx+1):
            us[i,((n_snap+1)*p)+n] = ue[i,n,p]           

#compute mean of snapshots
um = np.zeros(nx+1) 
for n in range(0,n_s):
    um[:] = um[:] + us[:,n] 
um[:]=um[:]/(n_s)

#compute anomaly(fluctuating) part of snapshots
uf = np.zeros((nx+1,n_s))
for n in range(0,n_s):
    for i in range(nx+1):
        uf[i,n] = us[i,n]-um[i]


#generate snapshots time correlation matrix
c = np.zeros((n_s,n_s))
for l in range(n_s):
    for k in range(n_s):
        c[k,l] = simps(uf[:,k]*uf[:,l],x)
            
#solve eigensystem
w, v = LA.eig(c)      
w = np.real(w)
v = np.real(v)

phi = np.zeros((nx+1,n_modes))
v_reduced = v[:,0:n_modes]
phi = uf.dot(v_reduced)

for k in range(n_modes):
    phi[:,k] = phi[:,k]/np.sqrt(abs(w[k]))

#compute RIC (relative inportance index)
ric = np.zeros(n_s)
for k in range(n_s):
    temp = 0.0
    for i in range(k+1):
        temp = temp + w[i]
    ric[k] = temp/sum(w)*100

#%%
# galerkin projection
b_c = np.zeros(n_modes)
b_l = np.zeros((n_modes,n_modes))
b_nl = np.zeros((n_modes,n_modes, n_modes))
phid = np.zeros((nx+1,n_modes))
phidd = np.zeros((nx+1,n_modes))

re_test = 2000.0
nu_test = 1.0/re_test
umdd = np.zeros(nx+1)
umd = np.zeros(nx+1)
umdd = pade4dd(um,dx,nx)
umd = pade4d(um,dx,nx)

for i in range(n_modes):
    phidd[:,i] = pade4dd(phi[:,i],dx,nx)
    phid[:,i] = pade4d(phi[:,i],dx,nx)

#constant term:
for k in range(n_modes):
    b_c[k] = nu_test*simps(phi[:,k]*umdd,x) - simps(phi[:,k]*um*umd,x)

# linear term    
for k in range(n_modes):
    for i in range(n_modes):
        b_l[i,k] = nu_test*simps(phi[:,k]*phidd[:,i],x) - \
                   simps(phi[:,k]*(um*phid[:,i] + phi[:,i]*umd),x) 

# non-linear term                   
for k in range(n_modes):
    for j in range(n_modes):
        for i in range(n_modes):
            b_nl[i,j,k] = - simps(phi[:,k]*phi[:,i]*phid[:,j],x)
                  
# galerkin-rom
dt = tmax/nt
a = np.zeros((n_modes,nt+1))
at = np.zeros((n_modes,nt+1))
ut = np.zeros((nx+1,nt+1))
r = np.zeros(n_modes)
a1 = np.zeros(n_modes)
t = np.linspace(0,tmax,nt+1)    

#exact solution for fluctuating part:
for n in range(nt+1):
    for i in range(nx+1):
        ut[i,n]=uexact(x[i],t[n],nu_test)-um[i]

# projection of exact solution to modes
for k in range(n_modes):
    for l in range(nt+1):
        at[k,l] = simps(phi[:,k]*ut[:,l],x)

a[:,0]=at[:,0] # initial condition for Galerkin ROM

# Galerkin ROM
for k in range(1,nt+1):
    #1.stage
    r = rhs(n_modes, b_c, b_l, b_nl, a[:,k-1])
    a1 = a[:,k-1] + dt*r
    #2. stage
    r = rhs(n_modes, b_c, b_l, b_nl, a1)
    a1 = 0.75*a[:,k-1] + 0.25*a1 + 0.25*dt*r
    #3. stage
    r = rhs(n_modes, b_c, b_l, b_nl, a1)
    a[:,k] = (1.0/3.0)*a[:,k-1] + (2.0/3.0)*a1 + (2.0/3.0)*dt*r

#%% export  data
t = t.reshape(nt+1,1)
ret = np.array([re_test for i in range(nt+1)])
ret = ret.reshape(nt+1,1)
filename = 'at_2000.csv'
rt = np.hstack((ret, t, np.transpose(at)))
np.savetxt(filename, rt, delimiter=",")
filename = 'agp_2000.csv'
rgp = np.hstack((ret, t, np.transpose(a)))
np.savetxt(filename, rgp, delimiter=",")

#%%
#--------------------------------------------------ANN --------------------------------------------------# 
n_train = (nt+1)*n_re #total number of training sample
utrain = np.zeros((nx+1,n_train))
for p in range(0,n_re):
    for n in range(0,nt+1):
        for i in range(0,nx+1):
            utrain[i,((nt+1)*p)+n]=uexact(x[i],t[n],nu[p])-um[i]

# projection of exact solution to modes for training
atrain = np.zeros((n_train,n_modes))
for k in range(n_modes):
    for l in range(n_train):
        atrain[l,k] = simps(phi[:,k]*utrain[:,l],x)

add_train = np.zeros((n_train,2))
for p in range(0,n_re):
    for n in range(0,nt+1):
        add_train[((nt+1)*p)+n,0]=1.0/nu[p]
        add_train[((nt+1)*p)+n,1]=t[n]

training_set = np.hstack((add_train, atrain))

#%%
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import optimizers

import os
if os.path.isfile('best_model.hd5'):
    os.remove('best_model.hd5')

def create_training_data(training_set, m, n, dt):
    """ 
    This function creates the input and output of the neural network. The function takes the
    training data as an input and creates the input for DNN.
    The input for the LSTM is 3-dimensional matrix.
    For lookback = 2;  [y(n)] ------> [(y(n+1)-y(n))/dt]
    """
    xtrain = training_set[0:m-1]
    ytrain = [training_set[i+1,2:n] for i in range(m-1)]
    ytrain = np.array(ytrain)
    return xtrain, ytrain

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
#%%
m,n = training_set.shape

xtrain, ytrain = create_training_data(training_set, m, n, dt)

from sklearn.preprocessing import MinMaxScaler
sc_input = MinMaxScaler(feature_range=(0,1))
sc_input = sc_input.fit(xtrain)
xtrain_scaled = sc_input.transform(xtrain)
xtrain = xtrain_scaled

from sklearn.preprocessing import MinMaxScaler
sc_output = MinMaxScaler(feature_range=(0,1))
sc_output = sc_output.fit(ytrain)
ytrain_scaled = sc_output.transform(ytrain)
ytrain = ytrain_scaled

model = Sequential()
model.add(Dropout(0.2))

# Layers start
input_layer = Input(shape=(n,))

# Hidden layers
x = Dense(10, activation='sigmoid', kernel_initializer='glorot_normal', use_bias=True)(input_layer)# ,
#x = Dense(20, activation='sigmoid', use_bias=True)(x)
op_val = Dense(n-2, activation='linear', use_bias=True)(x)

custom_model = Model(inputs=input_layer, outputs=op_val)
filepath = "best_model.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
history_callback = custom_model.fit(xtrain, ytrain, epochs=200, batch_size=500, verbose=1, validation_split=0.1,
                                    callbacks=callbacks_list)

loss_history = history_callback.history["loss"]
val_loss_history = history_callback.history["val_loss"]
# evaluate the model
scores = custom_model.evaluate(xtrain, ytrain)
print("\n%s: %.2f%%" % (custom_model.metrics_names[1], scores[1]*100))

epochs = range(1, len(loss_history) + 1)

plt.figure()
plt.semilogy(epochs, loss_history, 'b', label='Training loss')
plt.semilogy(epochs, val_loss_history, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#%%
testing_set = rt
m,n=testing_set.shape
tt = t[0]
tt = tt.reshape(1,1)
retest = testing_set[0,0]
retest = retest.reshape(1,1)

custom_model = load_model('best_model.hd5', custom_objects={'coeff_determination': coeff_determination})

xt = [testing_set[0]] # start ytest = y(0)
xt = np.array(xt)

ytest_ml = np.zeros((m,n-2))
ytest_ml[0] = testing_set[0,2:] 

for i in range(1,m):
    xt_sc = sc_input.transform(xt) # scale the input to the model
    yt = custom_model.predict(xt_sc) # predict slope from the model 
    yt_sc = sc_output.inverse_transform(yt) # scale the calculated slope to the training data scale
    ytest_ml[i] = yt_sc
    xtt = yt_sc.reshape(1,n-2) # update the input for next time step 
    tt = tt + dt
    xt = np.hstack((retest, tt, xtt))

#%%
filename = 'aml.csv'
rml = np.hstack((t, ytest_ml))
np.savetxt(filename, rml, delimiter=",")

#%%
import numpy as np
import matplotlib.pyplot as plt

at = np.loadtxt(open('at.csv', "rb"), delimiter=",", skiprows=0)
agp = np.loadtxt(open('agp.csv', "rb"), delimiter=",", skiprows=0)
aml = np.loadtxt(open('aml.csv', "rb"), delimiter=",", skiprows=0)
m,n = at.shape

font = {'family' : 'Times New Roman',
        'size'   : 14}	
plt.rc('font', **font)

nrows = int((n-1)/2)
k = 1
fig, axs = plt.subplots(nrows, 2, figsize=(11,8))#, constrained_layout=True)
for i in range(nrows):
    for j in range(2):
        axs[i,j].plot(at[:,1],at[:,k+1], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
        axs[i,j].plot(agp[:,1],agp[:,k+1], color='blue', linestyle='-.', label=r'$y_'+str(i+1)+'$'+' (GP)', zorder=5) 
        axs[i,j].plot(aml[:,0],aml[:,k], color='red', linestyle='--', label=r'$y_'+str(i+1)+'$'+' (GP)', zorder=5)
        axs[i,j].set_xlim([0, at[m-1,1]])
        axs[i,j].set_ylabel('$a_'+'{'+(str(k)+'}'+'$'), fontsize = 14)
        axs[i,j].set_xlabel('Time', fontsize = 14)
        k = int(k+1)

fig.tight_layout() 

fig.subplots_adjust(bottom=0.1)

line_labels = ["True", "ROM-G", "ML"]#, "ML-Train", "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 13} ) #bbox_to_anchor=(0.5, 0.0), borderaxespad=0.1, 

fig.savefig('gp.eps')#, bbox_inches = 'tight', pad_inches = 0.01)

#%%
filename = 'at_all.csv'
np.savetxt(filename, training_set, delimiter=",")