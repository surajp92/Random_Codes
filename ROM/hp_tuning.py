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
from tensorflow import set_random_seed
set_random_seed(2)
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

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
    ytrain = [(training_set[i+1,2:n]-training_set[i,2:n])/dt for i in range(m-1)]
    ytrain = np.array(ytrain)
    t = t.reshape(m,1)
    xtrain = training_set[0:m-1]
    return xtrain, ytrain

def create_model():
    model = Sequential()

    # Layers start
    input_layer = Input(shape=(n,))
    #model.add(Dropout(0.2))
    
    # Hidden layers
    x = Dense(20, activation='sigmoid', use_bias=True)(input_layer)
    x = Dense(20, activation='sigmoid', use_bias=True)(x)
    #x = Dense(20, activation='sigmoid', use_bias=True)(x)
    #x = Dense(20, activation='sigmoid', use_bias=True)(x)
    #x = Dense(20, activation='sigmoid', use_bias=True)(x)
    op_val = Dense(n-2, activation='linear', use_bias=True)(x)
    
    custom_model = Model(inputs=input_layer, outputs=op_val)
    filepath = "best_model.hd5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
    
    return custom_model
 
#%%
#dataset_val= pd.read_csv('./at.csv', sep=",",skiprows=300,header = None, nrows=150)
#mv,nv=dataset_val.shape
#validation_set = dataset_val.iloc[:,0:n].values
#tv = validation_set[:,0]
#validation_set = validation_set[:,0:nv]
#
##indices_val = np.random.randint(0,300,50)
##xval = xtrain[indices_val]
##yval = ytrain[indices_val]
#
#xval, yval = create_training_data(validation_set, mv, nv, dt, tv)
#
#xval_scaled = sc_input.transform(xval)
#xval = xval_scaled
#
#yval_scaled = sc_output.transform(yval)
#yval = yval_scaled
    
#%%
dataset_train = pd.read_csv('./at_all.csv', sep=",",skiprows=0,header = None)#, nrows=500)
m,n=dataset_train.shape
training_set = dataset_train.iloc[:,0:n].values
t = training_set[:,1]
dt = t[1] - t[0]
training_set = training_set[:,0:n]

xtrain, ytrain = create_training_data(training_set, m, n, dt, t)

#%%
from sklearn.preprocessing import MinMaxScaler
sc_input = MinMaxScaler(feature_range=(0,1))
sc_input = sc_input.fit(xtrain)
xtrain_scaled = sc_input.transform(xtrain)
xtrain_scaled.shape
xtrain = xtrain_scaled

from sklearn.preprocessing import MinMaxScaler
sc_output = MinMaxScaler(feature_range=(0,1))
sc_output = sc_output.fit(ytrain)
ytrain_scaled = sc_output.transform(ytrain)
ytrain_scaled.shape
ytrain = ytrain_scaled

indices = np.random.randint(0,xtrain.shape[0],8000)
xtrain1 = xtrain[indices]
ytrain1 = ytrain[indices]


#%%
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40]
epochs = [10, 50, 100, 200, 400]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(xtrain, ytrain)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#%%
model = Sequential()

# Layers start
input_layer = Input(shape=(n,))
#model.add(Dropout(0.2))

# Hidden layers
x = Dense(20, activation='sigmoid', use_bias=True)(input_layer)
x = Dense(20, activation='sigmoid', use_bias=True)(x)
#x = Dense(20, activation='sigmoid', use_bias=True)(x)
#x = Dense(20, activation='sigmoid', use_bias=True)(x)
#x = Dense(20, activation='sigmoid', use_bias=True)(x)
op_val = Dense(n-2, activation='linear', use_bias=True)(x)

custom_model = Model(inputs=input_layer, outputs=op_val)
filepath = "best_model.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
history_callback = custom_model.fit(xtrain, ytrain, epochs=300, batch_size=50, verbose=1, validation_split=0.4,
                                    callbacks=callbacks_list)

#mean_squared_error
#--------------------------------------------------------------------------------------------------------------#
# training and validation loss. Plot loss
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
#--------------------------------------------------------------------------------------------------------------#
# predict results recursively using the model 
dataset_test = pd.read_csv('./at.csv', sep=",",header = None, skiprows=0)#, nrows=2000)
m,n=dataset_test.shape
testing_set = dataset_test.iloc[:,0:n].values
t = testing_set[:,1]
t_temp = t[0]
t_temp = t_temp.reshape(1,1)
ret = testing_set[0,0]
ret = ret.reshape(1,1)
m,n=testing_set.shape

#%%
#def model_predict(testing_set, m, n, dt, sc_input, sc_output, t_temp):
custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})

ytest = [testing_set[0]] # start ytest = y(0)
ytest = np.array(ytest)

ytest_ml = np.zeros((m,n-2))
ytest_ml[0] = testing_set[0,2:] 

for i in range(1,m):
    ytest_sc = sc_input.transform(ytest) # scale the input to the model
    slope_ml = custom_model.predict(ytest_sc) # predict slope from the model 
    slope_ml_sc = sc_output.inverse_transform(slope_ml) # scale the calculated slope to the training data scale
    a = ytest_ml[i-1] + dt*slope_ml_sc # y1 at next time step
    ytest_ml[i] = a
    ytest = a.reshape(1,n-2) # update the input for next time step 
    t_temp = t_temp + dt
    ytest = np.hstack((ret, t_temp, ytest))


#%%
t = t.reshape(m,1)
filename = 'aml.csv'
rt = np.hstack((t, ytest_ml))
np.savetxt(filename, rt, delimiter=",")
