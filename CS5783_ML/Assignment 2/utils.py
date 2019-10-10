# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:50:41 2019

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
def get_data():
    training_images_file = open('train-images-idx3-ubyte','rb')
    training_images = training_images_file.read()
    training_images_file.close()
    training_images = bytearray(training_images)
    training_images = training_images[16:]
    
    training_labels_file = open('train-labels-idx1-ubyte','rb')
    training_labels = training_labels_file.read()
    training_labels_file.close()
    training_labels = bytearray(training_labels)
    training_labels = training_labels[8:]
    
    test_images_file = open('t10k-images-idx3-ubyte','rb')
    test_images = test_images_file.read()
    test_images_file.close()
    test_images = bytearray(test_images)
    test_images = test_images[16:]
    
    test_labels_file = open('t10k-labels-idx1-ubyte','rb')
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

#%%
def get_data_nt():
    training_images_file = open('train-images-idx3-ubyte','rb')
    training_images = training_images_file.read()
    training_images_file.close()
    training_images = bytearray(training_images)
    training_images = training_images[16:]
    
    training_labels_file = open('train-labels-idx1-ubyte','rb')
    training_labels = training_labels_file.read()
    training_labels_file.close()
    training_labels = bytearray(training_labels)
    training_labels = training_labels[8:]
    
    test_images_file = open('t10k-images-idx3-ubyte','rb')
    test_images = test_images_file.read()
    test_images_file.close()
    test_images = bytearray(test_images)
    test_images = test_images[16:]
    
    test_labels_file = open('t10k-labels-idx1-ubyte','rb')
    test_labels = test_labels_file.read()
    test_labels_file.close()
    test_labels = bytearray(test_labels)
    test_labels = test_labels[8:]
    
    training_images = np.array(training_images, dtype=int)
    training_images = training_images.reshape(60000,-1)
    
    training_labels = np.array(training_labels,dtype=int)
    training_labels = training_labels.reshape(60000,-1)
    
    test_images = np.array(test_images, dtype=int)
    test_images = test_images.reshape(10000,-1)
    
    test_labels = np.array(test_labels, dtype=int)
    test_labels = test_labels.reshape(10000,-1)
    
    return training_images, training_labels, test_images, test_labels

#%%
def get_data_problem2():
    train_images, train_labels, test_images, test_labels = get_data_nt()
    set1,label1 = train_images[train_labels[:,0]==5], train_labels[train_labels[:,0]==5]
    set1 = set1[np.random.randint(set1.shape[0],size=1000)]
    label1 = label1[:1000,:]
    
    set2, label2 = train_images[train_labels[:,0]!=5], train_labels[train_labels[:,0]!=5]
    mask = np.random.randint(set2.shape[0],size=1000)
    set2 = set2[mask]
    label2 = label2[mask]
    label2[:] = 0
    
    return set1, label1, set2, label2
    
#%%
def get_data_problem3():
    train_images, train_labels, test_images, test_labels = get_data_nt()    
    set1,label1 = train_images[train_labels[:,0]==1], train_labels[train_labels[:,0]==1]
    set2,label2 = train_images[train_labels[:,0]==2], train_labels[train_labels[:,0]==2]
    set7,label7 = train_images[train_labels[:,0]==7], train_labels[train_labels[:,0]==7]
    set1 = set1[np.random.randint(set1.shape[0],size=250)]
    set2 = set2[np.random.randint(set1.shape[0],size=250)]
    set7 = set7[np.random.randint(set1.shape[0],size=250)]
    label1 = label1[:200,:]
    label2 = label2[:200,:]
    label7 = label7[:200,:]
    train_images_new = np.vstack((set1[:200,:],set2[:200,:],set7[:200,:]))
    train_label_new = np.vstack((label1,label2,label7))
    
    label1 = test_labels[test_labels[:,0]==1]
    label2 = test_labels[test_labels[:,0]==2]
    label7 = test_labels[test_labels[:,0]==7]
    label1 = label1[:50,:]
    label2 = label2[:50,:]
    label7 = label7[:50,:]
    test_images_new = np.vstack((set1[200:,:],set2[200:,:],set7[200:,:]))
    test_label_new = np.vstack((label1,label2,label7))
    
    return train_images_new, train_label_new, test_images_new, test_label_new

#%%
def plot_problem2(roc):
    print("See plots")
    fig, axs = plt.subplots(1,1,figsize=(6,5))
    
    axs.plot(roc[:,1],roc[:,2],'b-',lw=3)
    
    axs.set_ylabel('True positive rate (TPR)')
    axs.set_xlabel('False positive rate (FPR)')
    
    axs.set_title('ROC curve')
    
    fig.tight_layout()
    plt.show()
    
#%%
def plot_problem3(test_data, c1, ic1, c2, ic2, c7, ic7):
    print("See plots")
    fig, axs = plt.subplots(2,3,figsize=(8,6))
    
    axs[0,0].imshow(test_data[c1,:].reshape(28,28))
    axs[0,0].set_title('Correctly classified 1')
    #if c1 !=ic1:
    axs[1,0].imshow(test_data[ic1,:].reshape(28,28))
    axs[1,0].set_title('Incorrectly classified 1')
    
    axs[0,1].imshow(test_data[c2,:].reshape(28,28))
    axs[0,1].set_title('Correctly classified 2')
    axs[1,1].imshow(test_data[ic2,:].reshape(28,28))
    axs[1,1].set_title('Correctly classified 2')
    
    axs[0,2].imshow(test_data[c7,:].reshape(28,28))
    axs[0,2].set_title('Correctly classified 7')
    axs[1,2].imshow(test_data[ic7,:].reshape(28,28))
    axs[1,2].set_title('Correctly classified 7')
    
    plt.subplots_adjust(wspace=0, hspace=0)
    
    fig.tight_layout()
    plt.show()
