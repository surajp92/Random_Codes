#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:54:38 2020

@author: suraj
"""
import random
random.seed(10)

import numpy as np
np.random.seed(10)
from numpy import linalg as LA

import time as tm

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.animation as animation

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from keras.regularizers import l2
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

font = {'family' : 'Times New Roman',
        'size'   : 16}    
plt.rc('font', **font)

#'weight' : 'bold'

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

import h5py
from tqdm import tqdm as tqdm

#%%
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

def create_training_data_lstm(training_set, m, n, lookback):
    ytrain = [training_set[i+1] for i in range(lookback-1,m-1)]
    ytrain = np.array(ytrain)
    xtrain = np.zeros((m-lookback,lookback,n))
    for i in range(m-lookback):
        a = training_set[i]
        for j in range(1,lookback):
            a = np.vstack((a,training_set[i+j]))
        xtrain[i] = a

    return xtrain, ytrain

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#%%
f = h5py.File('./sst_weekly.mat','r') # can be downloaded from https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu?usp=sharing
lat = np.array(f['lat'])
lon = np.array(f['lon'])
sst = np.array(f['sst'])
time = np.array(f['time'])

sst2 = np.zeros((len(sst[:,0]),len(lat[0,:]),len(lon[0,:])))
for i in tqdm(range(len(sst[:,0]))):
    sst2[i,:,:] = np.flipud((sst[i,:].reshape(len(lat[0,:]),len(lon[0,:]),order='F')))

#%%
fig,axs = plt.subplots(1,1, figsize=(10,8))

current_cmap = plt.cm.get_cmap('RdYlBu')
current_cmap.set_bad(color='black',alpha=0.8)

cs = axs.imshow(sst2[0,:,:],cmap='RdYlBu')
#axs.grid()
fig.colorbar(cs, ax=axs, orientation='vertical',shrink=0.4)

    
fig.tight_layout()
plt.show()    

#%%
make_animation = False
if make_animation:
    fig = plt.figure()
    #ax = plt.axes(xlim=(0, lx), ylim=(0, ly))  
    #plt.xlabel(r'x')
    #plt.ylabel(r'y')
    
    plt.xticks([])
    plt.yticks([])
        
    # animation function
    def animate(i): 
        cont = plt.imshow(sst2[15*i,:150,:], cmap='seismic')
        return cont  
    
    anim = animation.FuncAnimation(fig, animate, frames=100)
    fig.tight_layout()
    anim.save('animation.mp4')

#%%
sst_no_nan = np.nan_to_num(sst)
sst = sst.T

#%%
num_samples = sst.shape[1]

for i in range(num_samples):
    nan_array = np.isnan(sst[:,i])
    not_nan_array = ~ nan_array
    array2 = sst[:,i][not_nan_array]
    print(i, array2.shape[0])
    if i == 0:
        num_points = array2.shape[0]
        sst_masked = np.zeros((array2.shape[0],num_samples))
    sst_masked[:,i] = array2

#%%
ns = 1500
t = np.linspace(1,ns,ns)
sst_masked_small = sst_masked[:,:ns]
sst_average_small = np.sum(sst_masked_small,axis=1,keepdims=True)/(ns)

#%%
sst_masked_small_fluct = sst_masked_small - sst_average_small    

#%%
nr = 8
PHIw, L, RIC  = POD(sst_masked_small_fluct, nr)     

L_per = np.zeros(L.shape)
for n in range(L.shape[0]):
    L_per[n] = np.sum(L[:n],axis=0,keepdims=True)/np.sum(L,axis=0,keepdims=True)*100

#%%
k = np.linspace(1,ns,ns)
fig, axs = plt.subplots(1, 1, figsize=(7,5))#, constrained_layout=True)
axs.loglog(k,L_per, lw = 2, marker="o", linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
axs.set_xlim([1,ns])
axs.axvspan(0, nr, alpha=0.2, color='red')
fig.tight_layout()
plt.show()

#%%
at = PODproj(sst_masked_small_fluct, PHIw)

fig, ax = plt.subplots(nrows=4,ncols=2,figsize=(12,8),sharex=True)
ax = ax.flat
nrs = at.shape[1]

for i in range(nrs):
    ax[i].plot(t,at[:,i],'k',label=r'True Values')
#    ax[i].set_xlabel(r'$t$',fontsize=14)
    ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
    ax[-1].set_xlim([t[0],t[-1]])

ax[-2].set_xlabel(r'$t$',fontsize=14)    
ax[-1].set_xlabel(r'$t$',fontsize=14)    
fig.tight_layout()

fig.subplots_adjust(bottom=0.1)
line_labels = ["True"]#, "ML-Train", "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
plt.show()

#%%
tfluc = PODrec(at,PHIw)

T = tfluc + sst_average_small

#%%
aa = np.zeros(not_nan_array.shape[0])
aa[aa == 0] = 'nan'
aa[not_nan_array] = T[:,0]
trec = np.flipud((aa.reshape(len(lat[0,:]),len(lon[0,:]),order='F')))

fig,axs = plt.subplots(2,1, figsize=(10,8))

current_cmap = plt.cm.get_cmap('jet')
current_cmap.set_bad(color='white',alpha=1.0)

cs = axs[0].imshow(sst2[0,:,:],cmap='jet')
fig.colorbar(cs, ax=axs[0], orientation='vertical',shrink=1.0)

cs = axs[1].imshow(trec,cmap='jet')
fig.colorbar(cs, ax=axs[1], orientation='vertical',shrink=1)

fig.tight_layout()
plt.show()    

#%%
diff = trec - sst2[0,:,:]    
nan_array_2d = np.isnan(diff)
not_nan_array_2d = ~ nan_array_2d
diff_no_nan = diff[not_nan_array_2d]

l2_norm = np.linalg.norm(diff_no_nan)/np.sqrt(diff_no_nan.shape[0])

#%%
num_samples_train = 1000
lookback = 8

atrain = at[:num_samples_train,:]

m,n = atrain.shape

sc = MinMaxScaler(feature_range=(-1,1))
training_set_scaled = sc.fit_transform(atrain)
training_set = training_set_scaled

#%%
data_sc, labels_sc = create_training_data_lstm(training_set, m, n, lookback)
xtrain, xvalid, ytrain, yvalid = train_test_split(data_sc, labels_sc, test_size=0.3 , shuffle= True)

#%%
training_time_init = tm.time()
model = Sequential()

model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='relu', kernel_initializer='glorot_normal'))
model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='relu', kernel_initializer='glorot_normal'))
model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='relu', kernel_initializer='glorot_normal'))
model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='relu', kernel_initializer='glorot_normal'))
#model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
model.add(LSTM(40, input_shape=(lookback, n), activation='relu', kernel_initializer='glorot_normal'))
model.add(Dense(n, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=[coeff_determination])

model.summary()

# ,kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)

#%%
history = model.fit(xtrain, ytrain, epochs=600, batch_size=32, validation_data= (xvalid,yvalid))

total_training_time = tm.time() - training_time_init
print('Total training time=', total_training_time)
cpu = open("a_cpu.txt", "w+")
cpu.write('training time in seconds =')
cpu.write(str(total_training_time))
cpu.write('\n')

#%%
loss = history.history['loss']
val_loss = history.history['val_loss']
avg_mae = history.history['coeff_determination']
val_avg_mae = history.history['val_coeff_determination']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.semilogy(epochs, loss, 'b', label='Training loss')
plt.semilogy(epochs, val_loss, 'r', label='Validationloss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

plt.figure()
plt.semilogy(epochs, avg_mae, 'b', label=f'Average $R_2$')
plt.semilogy(epochs, val_avg_mae, 'r', label=f'Validation Average $R_2$')
plt.title('Evaluation metric')
plt.legend()
plt.show()

#%%
testing_set = np.copy(at)
testing_set_scaled = sc.fit_transform(testing_set)
testing_set= testing_set_scaled

#%%
m,n = testing_set.shape
ytest = np.zeros((1,lookback,n))
ytest_ml = np.zeros((m,n))

# create input at t = 0 for the model testing
for i in range(lookback):
    ytest[0,i,:] = testing_set[i]
    ytest_ml[i] = testing_set[i]

#%%
testing_time_init = tm.time()

# predict results recursively using the model
for i in range(lookback,m):
    slope_ml = model.predict(ytest)
    ytest_ml[i] = slope_ml
    e = ytest
    for i in range(lookback-1):
        e[0,i,:] = e[0,i+1,:]
    e[0,lookback-1,:] = slope_ml
    ytest = e 

#%%
total_testing_time = tm.time() - testing_time_init
print('Total testing time=', total_testing_time)
cpu.write('testing time in seconds = ')
cpu.write(str(total_testing_time))
cpu.close()

#%%  unscaling
ytest_ml_unscaled = sc.inverse_transform(ytest_ml)
ytest_ml= ytest_ml_unscaled

testing_set_unscaled = sc.inverse_transform(testing_set)
testing_set= testing_set_unscaled

#%%
fig, ax = plt.subplots(nrows=4,ncols=2,figsize=(12,8),sharex=True)
ax = ax.flat
nrs = at.shape[1]

for i in range(nrs):
    ax[i].plot(t,at[:,i],'k',label=r'True Values')
    ax[i].plot(t,ytest_ml[:,i],'b-',label=r'ML ')
    ax[i].set_xlabel(r'$t$',fontsize=14)
    ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
    ax[-1].set_xlim([t[0],t[-1]])
    ax[i].axvspan(0, t[num_samples_train], alpha=0.2, color='darkorange')

ax[-2].set_xlabel(r'$t$',fontsize=14)    
ax[-1].set_xlabel(r'$t$',fontsize=14)    
fig.tight_layout()

fig.subplots_adjust(bottom=0.1)
line_labels = ["True", "ML"]#, "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
plt.show()
fig.savefig('true_ml.png', dpi=200)