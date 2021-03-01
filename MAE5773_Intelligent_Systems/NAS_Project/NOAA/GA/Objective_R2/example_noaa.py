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

import tensorflow as tf
tf.random.set_seed(0)

from numpy import linalg as LA

import time as tm

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.animation as animation

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.regularizers import l2
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

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
my_pc = True
f = h5py.File('./sst_weekly.mat','r') # can be downloaded from https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu?usp=sharing
lat = np.array(f['lat'])
lon = np.array(f['lon'])
sst = np.array(f['sst'])
time = np.array(f['time'])

sst2 = np.zeros((len(sst[:,0]),len(lat[0,:]),len(lon[0,:])))
for i in tqdm(range(len(sst[:,0]))):
    sst2[i,:,:] = np.flipud((sst[i,:].reshape(len(lat[0,:]),len(lon[0,:]),order='F')))

fig,axs = plt.subplots(1,1, figsize=(10,8))

current_cmap = plt.cm.get_cmap('RdYlBu')
current_cmap.set_bad(color='black',alpha=0.8)

cs = axs.imshow(sst2[0,:,:],cmap='RdYlBu')
#axs.grid()
fig.colorbar(cs, ax=axs, orientation='vertical',shrink=0.4)

    
fig.tight_layout()
plt.show()    

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

sst_no_nan = np.nan_to_num(sst)
sst = sst.T

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

sst_masked_small_fluct = sst_masked_small - sst_average_small    

nr = 8
PHIw, L, RIC  = POD(sst_masked_small_fluct, nr)     

L_per = np.zeros(L.shape)
for n in range(L.shape[0]):
    L_per[n] = np.sum(L[:n],axis=0,keepdims=True)/np.sum(L,axis=0,keepdims=True)*100

k = np.linspace(1,ns,ns)
fig, axs = plt.subplots(1, 1, figsize=(7,5))#, constrained_layout=True)
axs.loglog(k,L_per, lw = 2, marker="o", linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
axs.set_xlim([1,ns])
axs.axvspan(0, nr, alpha=0.2, color='red')
fig.tight_layout()
plt.show()

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

data_sc, labels_sc = create_training_data_lstm(training_set, m, n, lookback)
xtrain, xvalid, ytrain, yvalid = train_test_split(data_sc, labels_sc, test_size=0.3 , shuffle= True)


#%%
def lstm_model(n_layers=2,n_cells=40,act_func=3,initializer=2,optimizer=1,type_lstm=1):
    lookback = 8
    nr = 8
    
    lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)

    act_func_dict = {1:'tanh',2:'relu',3:tf.keras.layers.LeakyReLU(alpha=0.1)}
    initializer_dict = {1:'uniform',2:'glorot_normal',3:'random_normal'}
    optimizer_dict = {1:'adam',2:'rmsprop',3:'SGD'}
    
    input = Input(shape=(lookback,nr))
    a = LSTM(n_cells, return_sequences=True)(input)
    
    if type_lstm == 1:
        for k in range(n_layers):
            x = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(a) # main1 
            
            a = Add()([a,x]) # main1 + skip1
    
    elif type_lstm == 2:
        for k in range(n_layers):
            x = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(a) # main1 
            
            x = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(x) # main1
            
            a = Add()([a,x]) # main1 + skip1
    
    elif type_lstm == 3:
        for k in range(n_layers):
            x = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(a) # main1 
            a = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(a) # skip1
            
            x = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(x) # main1
            
            a = Add()([a,x]) # main1 + skip1
            
    elif type_lstm == 4:
        for k in range(n_layers):
            x = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(a) # main1 
            a = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(a) # skip1
            
            a = Add()([a,x]) # main1 + skip1
        
    x = LSTM(n_cells, return_sequences=False)(a)
    x = Dense(nr, activation='linear')(x)
    model = Model(input, x)

    model.compile(loss='mean_squared_error', optimizer=optimizer_dict[optimizer], metrics=[coeff_determination])
    # model.summary()
    # if my_pc:
    #     plot_model(model, to_file=f'model_plot_function_{type_lstm}.png', show_shapes=True, show_layer_names=True)
    
    return model

#%%
from param import ContinuousParam, CategoricalParam, ConstantParam, ContinuousParamGaussian
from genetic_hyperopt import GeneticHyperopt

X, y = xtrain, ytrain

optimizer = GeneticHyperopt(lstm_model, X, y, r2_score, maximize=True)

n_layers_param = ContinuousParam("n_layers", 4, 2, min_limit=2, max_limit=6, is_int=True)
n_cells_param = ContinuousParamGaussian("n_cells", 80, 40, min_limit=40, max_limit=120, is_int=True)
act_func_param = ContinuousParam("act_func", 2, 1, min_limit=1, max_limit=2, is_int=True)
initializer_param = ContinuousParam("initializer", 2, 1, min_limit=1, max_limit=3, is_int=True)
optimizer_param = ContinuousParam("optimizer", 2, 1, min_limit=1, max_limit=3, is_int=True)
type_lstm_param = ContinuousParam("type_lstm", 2, 1, min_limit=1, max_limit=4, is_int=True)

optimizer.add_param(n_layers_param)
optimizer.add_param(n_cells_param)
optimizer.add_param(act_func_param)
optimizer.add_param(initializer_param)
optimizer.add_param(optimizer_param)
optimizer.add_param(type_lstm_param)

training_time_init = tm.time()

best_params, best_score, plotting_stats, best_param_dict = optimizer.evolve()
np.savez('results_noaa.npz',plotting_stats=plotting_stats,best_param_dict=best_param_dict,
         best_params=best_params)

#%%
model = lstm_model(**best_params)
model.summary()
history = model.fit(xtrain, ytrain, epochs=600, batch_size=128, validation_data= (xvalid,yvalid))

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
plt.semilogy(epochs, avg_mae, 'b', label='Average $R_2$')
plt.semilogy(epochs, val_avg_mae, 'r', label='Validation Average $R_2$')
plt.title('Evaluation metric')
plt.legend()
plt.show()

