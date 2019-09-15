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