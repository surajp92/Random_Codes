#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:38:12 2019

@author: Suraj Pawar
"""
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
dataset_train = pd.read_csv('./at_all.csv', sep=",",skiprows=0,header = None)#, nrows=1001)
m,n=dataset_train.shape
training_set = dataset_train.iloc[:,0:n].values
t = training_set[:,1]
dt = t[1] - t[0]
training_set = training_set[:,0:n]


#%%
lookback = 4   # history for next time step prediction 
slopenet = "LSTM"

def create_training_data_lstm(training_set, m, n, dt, lookback):
    """ 
    This function creates the input and output of the neural network. The function takes the
    training data as an input and creates the input for LSTM neural network based on the lookback.
    The input for the LSTM is 3-dimensional matrix.
    For lookback = 2;  [y(n-2), y(n-1)] ------> [y(n)]
    """
    ytrain = [training_set[i+1,2:] for i in range(lookback-1,m-1)]
    ytrain = np.array(ytrain)
    xtrain = np.zeros((m-lookback,lookback,n))
    for i in range(m-lookback):
        a = training_set[i]
        for j in range(1,lookback):
            a = np.vstack((a,training_set[i+j]))
        xtrain[i] = a
    
    return xtrain, ytrain

#%%
xtrain, ytrain = create_training_data_lstm(training_set, m, n, dt, lookback)

#%% calculate maximum and minimum for scaling
xmax, xmin = [np.zeros(n) for i in range(2)]
ymax, ymin = [np.zeros(n-2) for i in range(2)]
for i in range(n):
    xmax[i] = max(training_set[:,i])
    xmin[i] = min(training_set[:,i])

for i in range(n-2):
    ymax[i] = max(ytrain[:,i])
    ymin[i] = min(ytrain[:,i])

#%% scale the input and output data to (-1,1)
xtrains = np.zeros((m-lookback,lookback,n)) 
ytrains = np.zeros((m-lookback,n-2)) 
for i in range(n):
    xtrains[:,:,i] = (2.0*xtrain[:,:,i]-(xmax[i]+xmin[i]))/(xmax[i]-xmin[i])
    
for i in range(n-2):
    ytrains[:,i] = (2.0*ytrain[:,i]-(ymax[i]+ymin[i]))/(ymax[i]-ymin[i])
    

#%%
model = Sequential()

model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='sigmoid', kernel_initializer='glorot_normal'))
model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='sigmoid', kernel_initializer='uniform'))
model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='sigmoid', kernel_initializer='uniform'))
#model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='sigmoid', kernel_initializer='uniform'))
model.add(LSTM(40, input_shape=(lookback, n), activation='sigmoid', kernel_initializer='uniform'))
model.add(Dense(n-2))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) # compile the model

history = model.fit(xtrains, ytrains, nb_epoch=500, batch_size=200, validation_split=0.1) # run the model

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
dataset_test = pd.read_csv('./at.csv', sep=",",header = None, skiprows=0)#, nrows=2000)
m,n=dataset_test.shape
testing_set = dataset_test.iloc[:,0:n].values
t = testing_set[:,1]
m,n = testing_set.shape

#%%
xt = np.zeros((1,lookback,n))
yt_ml = np.zeros((m,n-2))
xt_check = np.zeros((m,lookback,n))

for i in range(lookback):
    # create data for testing at first time step
    xt[0,i,:] = testing_set[i]
    yt_ml[i] = testing_set[i,2:]

xt_check[0,:,:] = xt
#%%
xt_sc = np.zeros((1,lookback,n))
yt_sc = np.zeros((1,n-2))

for i in range(lookback,m):
    for k in range(n):
        xt_sc[:,:,k] = (2.0*xt[:,:,k]-(xmax[k]+xmin[k]))/(xmax[k]-xmin[k])
    yt = model.predict(xt_sc) # slope from LSTM/ ML model
    for k in range(n-2):
        yt_sc[0,k] = 0.5*(yt[0,k]*(ymax[k]-ymin[k])+(ymax[k]+ymin[k]))
    yt_ml[i] = yt_sc # assign variable at next time ste y(n)
    e = xt.copy()   # temporaty variable
    for j in range(lookback-1):
        e[0,j,:] = e[0,j+1,:]
    e[0,lookback-1,0] = testing_set[i,0]
    e[0,lookback-1,1] = testing_set[i,1]
    e[0,lookback-1,2:] = yt_sc
    xt = e # update the input for the variable prediction at time step (n+1)
    xt_check[i,:,:] = xt

ytest_ml = yt_ml    

#%%
t = t.reshape(m,1)
filename = 'aml.csv'
rt = np.hstack((t, ytest_ml))
np.savetxt(filename, rt, delimiter=",")

#%%
at = np.loadtxt(open('at.csv', "rb"), delimiter=",", skiprows=0)
agp = np.loadtxt(open('agp.csv', "rb"), delimiter=",", skiprows=0)
aml = np.loadtxt(open('aml.csv', "rb"), delimiter=",", skiprows=0)
m,n = at.shape

#%% plotting
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
        axs[i,j].plot(at[:,1],aml[:,k], color='red', linestyle='--', label=r'$y_'+str(i+1)+'$'+' (GP)', zorder=5)
        axs[i,j].set_xlim([0, at[m-1,1]])
        axs[i,j].set_ylabel('$a_'+'{'+(str(k)+'}'+'$'), fontsize = 14)
        axs[i,j].set_xlabel('Time', fontsize = 14)
        k = int(k+1)

fig.tight_layout() 

fig.subplots_adjust(bottom=0.1)

line_labels = ["True", "ROM-G", "ML"]#, "ML-Train", "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 13} ) #bbox_to_anchor=(0.5, 0.0), borderaxespad=0.1, 

fig.savefig('gp.eps')#, bbox_inches = 'tight', pad_inches = 0.01)
