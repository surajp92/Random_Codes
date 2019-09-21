# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 11:54:55 2019

@author: suraj
"""
import numpy as np
import matplotlib.pyplot as plt

#%%
training_images_file = open('train-images.idx3-ubyte','rb')
training_images = training_images_file.read()
training_images_file.close()
training_images = bytearray(training_images)
training_images = training_images[16:]

training_labels_file = open('train-labels.idx1-ubyte','rb')
training_labels = training_labels_file.read()
training_labels_file.close()
training_labels = bytearray(training_labels)
training_labels = training_labels[8:]

#%%
training_images = np.array(training_images)
training_images = training_images.reshape(60000,-1)

training_labels = np.array(training_labels)
training_labels = training_labels.reshape(60000,-1)

cat, counts = np.unique(training_labels, return_counts=True)
theta_kd = np.zeros((cat.shape[0],training_images.shape[1]))
for c in cat:
    current_x = training_images[training_labels[:,0] == c]
    theta_kd[c,:] = np.sum(current_x,axis=0,keepdims=True)/counts[c]

#%%
training_images[training_images[:,:]>=1] = 1
training_images[training_images[:,:]<1] = 0



#%%
a = training_images[1].reshape(28,28)
plt.imshow(a)