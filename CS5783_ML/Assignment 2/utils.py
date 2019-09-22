# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:50:41 2019

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
def get_data():
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
    
    test_images_file = open('t10k-images.idx3-ubyte','rb')
    test_images = test_images_file.read()
    test_images_file.close()
    test_images = bytearray(test_images)
    test_images = test_images[16:]
    
    test_labels_file = open('t10k-labels.idx1-ubyte','rb')
    test_labels = test_labels_file.read()
    test_labels_file.close()
    test_labels = bytearray(test_labels)
    test_labels = test_labels[8:]
    
    training_images = np.array(training_images)
    training_images = training_images.reshape(60000,-1)
    
    training_labels = np.array(training_labels)
    training_labels = training_labels.reshape(60000,-1)
    
    test_images = np.array(test_images)
    test_images = test_images.reshape(10000,-1)
    
    test_labels = np.array(test_labels)
    test_labels = test_labels.reshape(10000,-1)
    
    training_images[training_images[:,:]>=1] = 1
    training_images[training_images[:,:]<1] = 0
    
    test_images[test_images[:,:]>=1] = 1
    test_images[test_images[:,:]<1] = 0
    
    return training_images, training_labels, test_images, test_labels