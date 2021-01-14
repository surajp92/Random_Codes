#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 14:59:52 2021

@author: suraj
"""

import numpy as np
import keras
#from keras.models import Model
#from keras.layers import Dense, Input

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate

import tensorflow.keras.backend as K
import os

import matplotlib.pyplot as plt

def nll_loss(y_true, y_pred):
    epsilon = 1e-6
    mean, sigma_sq = tf.split(y_pred, 2, axis=1, name='split')
#    sigma_sq_sp = y_pred[:,2:] # variance
#    sigma_sq_sp = keras.activations.softplus(sigma_sq) # softplus on the variance
    nll_loss =  0.5 * K.mean(K.log(sigma_sq + epsilon) + K.square(y_true - mean) / (sigma_sq + epsilon))
    
    return nll_loss


def toy_dataset(input):
    output = []

    for inp in input:
        std = 10 if inp < 0 else 2
        out = [inp ** 3 + np.random.normal(0, std), 10*np.sin(inp)  + np.random.normal(0, std)]
        output.append(out)

    return np.array(output)

def mlp_model():
    inp = Input(shape=(1,))
    x = Dense(10, activation="relu")(inp)
    x = Dense(20, activation="relu")(x)
    x = Dense(30, activation="relu")(x)
    mean = Dense(2, activation="linear")(x)
    variance = Dense(2, activation="softplus")(x)
    output = concatenate(inputs=[mean, variance])
    
    model = Model(inp, output)
    
#    print(model.summary())
    
    model.compile(loss=nll_loss, optimizer='adam')
    
    return model

x_train = np.linspace(-4.0, 4.0, num=1200)
x_test = np.linspace(-7.0, 7.0, 200)

y_train = toy_dataset(x_train)
y_test = toy_dataset(x_test)

fig, axs = plt.subplots(2, 1, sharex=True,  figsize=(8,10))
fig.subplots_adjust(hspace=0.15)  

i = 0
axs[i].plot(x_test, y_test[:,i], '.', color=(0, 0.9, 0.0, 0.8), markersize=12, label="Ground truth Points")
axs[i].plot(x_test, x_test ** 3, color='red', label="Ground truth x**3")

i = 1
axs[i].plot(x_test, y_test[:,i], '.', color=(0, 0.9, 0.0, 0.8), markersize=12, label="Ground truth Points")
axs[i].plot(x_test, 10*np.sin(x_test), color='red', label="10 sin(x)")

plt.show()

#%%
num_estimators = 5

models = [None] * num_estimators 

for i in range(num_estimators):        
    models[i] = mlp_model()

folder = 'regression-ens-2f'
if not os.path.exists(folder):
        os.makedirs(folder)
        
#    train_model, pred_model = DeepEnsembleRegressor(mlp_model, 5)
for i in range(num_estimators):
    models[i].fit(x_train, y_train, epochs=200)
    filename = os.path.join(folder, f'model-ensemble-{i}.hdf5')
    models[i].save(filename)

#%%
means = []
variances = []

for i in range(num_estimators):
    y_pred = models[i].predict(x_test)
    mean, variance =  y_pred[:,:2], y_pred[:,2:]
    means.append(mean)
    variances.append(variance)    
    
#%%
means = np.array(means)
variances = np.array(variances)

#%%
mixture_mean = np.mean(means, axis=0)
mixture_var  = np.mean(variances + np.square(means), axis=0) - np.square(mixture_mean)
mixture_var[mixture_var < 0.0] = 0.0
    
y_pred_mean, y_pred_std = mixture_mean, np.sqrt(mixture_var)


print("y pred mean shape: {}, y_pred_std shape: {}".format(y_pred_mean.shape, y_pred_std.shape))

y_pred_up_1 = y_pred_mean + y_pred_std
y_pred_down_1 = y_pred_mean - y_pred_std

y_pred_up_2 = y_pred_mean + 2.0 * y_pred_std
y_pred_down_2 = y_pred_mean - 2.0 * y_pred_std

y_pred_up_3 = y_pred_mean + 3.0 * y_pred_std
y_pred_down_3 = y_pred_mean - 3.0 * y_pred_std

#%%
fig, axs = plt.subplots(2, 1, sharex=True,  figsize=(8,10))
fig.subplots_adjust(hspace=0.15)  

i = 0
axs[i].plot(x_test, y_test[:,i], '.', color=(0, 0.9, 0.0, 0.8), markersize=12, label="Ground truth Points")
axs[i].plot(x_test, x_test ** 3, color='red', label="Ground truth x**3")
axs[i].plot(x_test, y_pred_mean[:,i], color='k', lw=2, label="Predicted mean")
#axs[i].fill_between(x_test, y_pred_down_3[:,i], y_pred_up_3[:,i], color=(0, 0, 0.9, 0.2), label="Three Sigma Confidence Interval")
#axs[i].fill_between(x_test, y_pred_down_2[:,i], y_pred_up_2[:,i], color=(0, 0, 0.9, 0.5), label="Two Sigma Confidence Interval")
axs[i].fill_between(x_test, y_pred_down_1[:,i], y_pred_up_1[:,i], color=(0, 0, 0.9, 0.5), label="One Sigma Confidence Interval")
axs[i].legend()
axs[i].axvspan(-4,4, color='darkorange', alpha=0.1)

i = 1
axs[i].plot(x_test, y_test[:,i], '.', color=(0, 0.9, 0.0, 0.8), markersize=12, label="Ground truth Points")
axs[i].plot(x_test, 10*np.sin(x_test), color='red', label="Ground truth 10 sin(x)")
axs[i].plot(x_test, y_pred_mean[:,i], color='k', lw=2, label="Predicted mean")
#axs[i].fill_between(x_test, y_pred_down_3[:,i], y_pred_up_3[:,i], color=(0, 0, 0.9, 0.2), label="Three Sigma Confidence Interval")
#axs[i].fill_between(x_test, y_pred_down_2[:,i], y_pred_up_2[:,i], color=(0, 0, 0.9, 0.5), label="Two Sigma Confidence Interval")
axs[i].fill_between(x_test, y_pred_down_1[:,i], y_pred_up_1[:,i], color=(0, 0, 0.9, 0.5), label="One Sigma Confidence Interval")
axs[i].legend()
axs[i].axvspan(-4,4, color='darkorange', alpha=0.1)

plt.show()