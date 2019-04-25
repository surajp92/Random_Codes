#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:46:57 2019

@author: Suraj Pawar
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def create_training_data(training_set, m, n, dt, t):
    """ 
    This function creates the input and output of the neural network. The function takes the
    training data as an input and creates the input for DNN.
    The input for the LSTM is 3-dimensional matrix.
    For lookback = 2;  [y(n)] ------> [(y(n+1)-y(n))/dt]
    """
    ytrain = [training_set[i+1,2:n] for i in range(m-1)]
    #ytrain = [(training_set[i+1,2:n]-training_set[i,2:n]) for i in range(m-1)]
    ytrain = np.array(ytrain)
    t = t.reshape(m,1)
    xtrain = training_set[0:m-1]
    return xtrain, ytrain

#%%
dataset_train = pd.read_csv('./at_all.csv', sep=",",skiprows=0,header = None)#, nrows=500)
m,n=dataset_train.shape
training_set = dataset_train.iloc[:,0:n].values
t = training_set[:,1]
dt = t[1] - t[0]
training_set = training_set[:,0:n]

#%%
xtrain, ytrain = create_training_data(training_set, m, n, dt, t)
xmax, xmin = [np.zeros(n) for i in range(2)]
ymax, ymin = [np.zeros(n-2) for i in range(2)]
for i in range(n):
    xmax[i] = max(xtrain[:,i])
    xmin[i] = min(xtrain[:,i])

for i in range(n-2):
    ymax[i] = max(ytrain[:,i])
    ymin[i] = min(ytrain[:,i])

#%%
xtrains = np.zeros((m-1,n)) 
ytrains = np.zeros((m-1,n-2)) 
for i in range(n):
    xtrains[:,i] = (2.0*xtrain[:,i]-(xmax[i]+xmin[i]))/(xmax[i]-xmin[i])

for i in range(n-2):
    ytrains[:,i] = (2.0*ytrain[:,i]-(ymax[i]+ymin[i]))/(ymax[i]-ymin[i])
    
    
#%%
#from sklearn.preprocessing import MinMaxScaler
#sc_input = MinMaxScaler(feature_range=(-1,1))
#sc_input = sc_input.fit(xtrain)
#xtrain_scaled = sc_input.transform(xtrain)
#xtrain = xtrain_scaled
#
#from sklearn.preprocessing import MinMaxScaler
#sc_output = MinMaxScaler(feature_range=(-1,1))
#sc_output = sc_output.fit(ytrain)
#ytrain_scaled = sc_output.transform(ytrain)
#ytrain = ytrain_scaled

indices = np.random.randint(0,xtrains.shape[0],9000)
xtrain1 = xtrains[indices]
ytrain1 = ytrains[indices]

#%%
model = Sequential()
model.add(Dense(40, input_shape=(8,), kernel_initializer='glorot_uniform', activation='sigmoid', use_bias=True))
model.add(Dense(40, init='uniform', activation='sigmoid', use_bias=True))
model.add(Dense(40, init='uniform', activation='sigmoid', use_bias=True))
model.add(Dense(40, init='uniform', activation='sigmoid', use_bias=True))
model.add(Dense(40, init='uniform', activation='sigmoid', use_bias=True))
#model.add(Dense(40, init='uniform', activation='sigmoid', use_bias=True))
model.add(Dense(6, activation='linear', use_bias=True))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) # compile the model

history = model.fit(xtrain1, ytrain1, nb_epoch=500, batch_size=200, validation_split=0.1) # run the model

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
#model = Sequential()
#
## Layers start
#input_layer = Input(shape=(n,))
##model.add(Dropout(0.2))
#
## Hidden layers
#x = Dense(40, activation='sigmoid', kernel_initializer='uniform', use_bias=True)(input_layer)# ,kernel_initializer='glorot_normal',
#x = Dense(40, activation='sigmoid', kernel_initializer='uniform', use_bias=True)(x)
#x = Dense(40, activation='sigmoid', kernel_initializer='uniform', use_bias=True)(x)
#x = Dense(40, activation='sigmoid', kernel_initializer='uniform', use_bias=True)(x)
#x = Dense(40, activation='sigmoid', kernel_initializer='uniform', use_bias=True)(x)
#x = Dense(40, activation='sigmoid', kernel_initializer='uniform', use_bias=True)(x)
#op_val = Dense(n-2, activation='linear', use_bias=True)(x)
#
#custom_model = Model(inputs=input_layer, outputs=op_val)
#filepath = "best_model.hd5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint]
#
##adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#history_callback = custom_model.fit(xtrains, ytrains, epochs=500, batch_size=200, verbose=1, 
#                                    validation_split=0.1, callbacks=callbacks_list)
#
##mean_squared_error
##--------------------------------------------------------------------------------------------------------------#
## training and validation loss. Plot loss
#loss_history = history_callback.history["loss"]
#val_loss_history = history_callback.history["val_loss"]
## evaluate the model
#scores = custom_model.evaluate(xtrain, ytrain)
#print("\n%s: %.2f%%" % (custom_model.metrics_names[1], scores[1]*100))
#
#epochs = range(1, len(loss_history) + 1)
#
#plt.figure()
#plt.semilogy(epochs, loss_history, 'b', label='Training loss')
#plt.semilogy(epochs, val_loss_history, 'r', label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
#plt.show()

#%%
#--------------------------------------------------------------------------------------------------------------#
# predict results recursively using the model 
dataset_test = pd.read_csv('./at.csv', sep=",",header = None, skiprows=0)#, nrows=2000)
m,n=dataset_test.shape
testing_set = dataset_test.iloc[:,0:n].values
t = testing_set[:,1]
m,n=testing_set.shape


#%%
#def model_predict(testing_set, m, n, dt, sc_input, sc_output, t_temp):
#custom_model = load_model('best_model.hd5')#,custom_objects={'coeff_determination': coeff_determination})

ytest = np.zeros((m,n))
ytest[0] = testing_set[0,:]

yt_ml = np.zeros((m,n-2))
yt_ml[0] = testing_set[0,2:] 

xt_sc = np.zeros((1,n))
yt_sc = np.zeros((1,n-2))

for i in range(1,m):
    xt = ytest[i-1]
    xt = xt.reshape(1,n)
    #xt_sc = sc_input.transform(xt) # scale the input to the model
    for k in range(n):
        xt_sc[0,k] = (2*xt[0,k]-(xmax[k]+xmin[k]))/(xmax[k]-xmin[k])
    yt = model.predict(xt_sc) # predict slope from the model 
    #yt_sc = sc_output.inverse_transform(yt) # scale the calculated slope to the training data scale
    for k in range(n-2):
        yt_sc[0,k] = 0.5*(yt[0,k]*(ymax[k]-ymin[k])+(ymax[k]+ymin[k]))
    yt_ml[i] = yt_sc
    ytest[i,0] = testing_set[i,0]
    ytest[i,1] = testing_set[i,1]
    ytest[i,2:] = yt_ml[i]

ytest_ml = yt_ml 
#%%
#custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
#
#testing_set_sc = sc_input.transform(testing_set)
#yt_ml = np.zeros((m,n-2))
#yt_ml[0] = testing_set_sc[0,2:] 
#
#ytest = np.zeros((m,n))
#ytest[0] = testing_set_sc[0,:]
#
#for i in range(1,m):
#    xt = ytest[i-1]
#    xt = xt.reshape(1,n)
#    yt_ml[i] = custom_model.predict(xt) # predict slope from the model 
#    ytest[i,0] = testing_set_sc[i,0]
#    ytest[i,1] = testing_set_sc[i,1]
#    ytest[i,2:] = yt_ml[i]
#
#ytest_ml = sc_output.inverse_transform(yt_ml)

#%%
#testing_set_sc = sc_input.transform(testing_set)
#ytest_ml_sc = custom_model.predict(testing_set_sc)
#ytest_ml = sc_output.inverse_transform(ytest_ml_sc)

#%%
t = t.reshape(m,1)
filename = 'aml.csv'
rt = np.hstack((t, ytest_ml))
np.savetxt(filename, rt, delimiter=",")

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