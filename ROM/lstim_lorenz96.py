#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 08:59:26 2019

@author: Suraj Pawar
"""

from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

# these are our constants
N = 40  # number of variables
F = 4  # forcing

def Lorenz96(x,t):

  # compute state derivatives
  d = np.zeros(N)
  # first the 3 edge cases: i=1,2,N
  d[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
  d[1] = (x[2] - x[N-1]) * x[0]- x[1]
  d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
  # then the general case
  for i in range(2, N-1):
      d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
  # add the forcing term
  d = d + F

  # return the state derivatives
  return d

x0 = F*np.ones(N) # initial state (equilibrium)
x0[1] += 0.01 # add small perturbation to 20th variable
t = np.arange(0.0, 30.0, 0.01)

x = odeint(Lorenz96, x0, t)

# plot first three variables
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
plt.show()

#%%
fig, ax1 = plt.subplots(nrows=1, figsize=(7,3.5))
xx = np.linspace(0,N,N)
ax1.contourf(t,xx,x.T, 40, cmap='jet', interpolation='bilinear')
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras import initializers

#%%
training_set = x
m,n=training_set.shape
dt = t[1] - t[0]
l = 1

#%%
lookback = 20   # history for next time step prediction 
slopenet = "LSTM"

def create_training_data_lstm(training_set, dt, lookback):
    """ 
    This function creates the input and output of the neural network. The function takes the
    training data as an input and creates the input for LSTM neural network based on the lookback.
    The input for the LSTM is 3-dimensional matrix.
    For lookback = 2;  [y(n-2), y(n-1)] ------> [y(n)]
    """
    m,n = training_set.shape
    ytrain = [(training_set[i+1,:]-training_set[i,:]) for i in range(lookback-1,m-1)]
    ytrain = np.array(ytrain)
    xtrain = np.zeros((m-lookback,lookback,n))
    for i in range(m-lookback):
        a = training_set[i]
        for j in range(1,lookback):
            a = np.vstack((a,training_set[i+j]))
        xtrain[i] = a
    
    return xtrain, ytrain

#%%
for k in range(l):
    p = int(k*m/l)
    q = int(p+m)
    xt, yt = create_training_data_lstm(training_set[p:q,:], dt, lookback)
    if k == 0:
        xtrain = xt
        ytrain = np.vstack(yt)
    else:
        xtrain = np.vstack((xtrain,xt))
        ytrain = np.vstack((ytrain,yt))

#%% calculate maximum and minimum for scaling
xmax, xmin = [np.zeros(n) for i in range(2)]
ymax, ymin = [np.zeros(n) for i in range(2)]
for i in range(n):
    xmax[i] = np.max(xtrain[:,:,i])
    xmin[i] = np.min(xtrain[:,:,i])

for i in range(n):
    ymax[i] = max(ytrain[:,i])
    ymin[i] = min(ytrain[:,i])

#%% scale the input and output data to (-1,1)
xtrains = np.zeros((m-lookback*l,lookback,n)) 
ytrains = np.zeros((m-lookback*l,n)) 
for i in range(n):
    xtrains[:,:,i] = (2.0*xtrain[:,:,i]-(xmax[i]+xmin[i]))/(xmax[i]-xmin[i])
    
for i in range(n):
    ytrains[:,i] = (2.0*ytrain[:,i]-(ymax[i]+ymin[i]))/(ymax[i]-ymin[i])

#%%
indices = np.random.randint(0,xtrains.shape[0],2000)
xtrain1 = xtrains[indices]
ytrain1 = ytrains[indices]

#%%
model = Sequential()

model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh', kernel_initializer='glorot_normal'))
#model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
#model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
#model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
model.add(LSTM(40, input_shape=(lookback, n), activation='tanh', kernel_initializer='uniform'))
model.add(Dense(n))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) # compile the model

history = model.fit(xtrains, ytrains, nb_epoch=200, batch_size=500, validation_split=0.1) # run the model

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.semilogy(epochs, loss, 'b', label='Training loss')
plt.semilogy(epochs, val_loss, 'r', label='Validationloss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#%%
testing_set = x
m,n=testing_set.shape

#%%
xt = np.zeros((1,lookback,n))
yt_ml = np.zeros((m,n))
xt_check = np.zeros((m,lookback,n))

for i in range(lookback):
    # create data for testing at first time step
    xt[0,i,:] = testing_set[i]
    yt_ml[i] = testing_set[i]

xt_check[0,:,:] = xt
#%%
xt_sc = np.zeros((1,lookback,n))
yt_sc = np.zeros((1,n))

for i in range(lookback,m):
    for k in range(n):
        xt_sc[:,:,k] = (2.0*xt[:,:,k]-(xmax[k]+xmin[k]))/(xmax[k]-xmin[k])
    yt = model.predict(xt_sc) # slope from LSTM/ ML model
    for k in range(n):
        yt_sc[0,k] = 0.5*(yt[0,k]*(ymax[k]-ymin[k])+(ymax[k]+ymin[k]))
    yt_ml[i] = yt_ml[i-1] + yt_sc # assign variable at next time ste y(n)
    e = xt.copy()   # temporaty variable
    for j in range(lookback-1):
        e[0,j,:] = e[0,j+1,:]
    e[0,lookback-1,:] = yt_ml[i]
    xt = e # update the input for the variable prediction at time step (n+1)
    xt_check[i,:,:] = xt

ytest_ml = yt_ml    

#%% plotting
font = {'family' : 'Times New Roman',
        'size'   : 14}	
plt.rc('font', **font)

nrows = 5
k = 1
fig, axs = plt.subplots(nrows, figsize=(11,8))#, constrained_layout=True)
for i in range(nrows):
    axs[i].plot(t,x[:,i], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
    axs[i].plot(t,ytest_ml[:,i], color='red', linestyle='--', label=r'$y_'+str(i+1)+'$'+' (GP)', zorder=5)
    axs[i].set_ylabel('$a_'+'{'+(str(k)+'}'+'$'), fontsize = 14)
    axs[i].set_xlabel('Time', fontsize = 14)

fig.tight_layout() 

fig.subplots_adjust(bottom=0.1)

line_labels = ["True", "ML"]#, "ML-Train", "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 13} ) #bbox_to_anchor=(0.5, 0.0), borderaxespad=0.1, 

fig.savefig('lorenz96.eps')#, bbox_inches = 'tight', pad_inches = 0.01)