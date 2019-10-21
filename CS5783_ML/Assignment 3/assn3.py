#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 10:58:10 2019

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.optimize import minimize

font = {'family' : 'Times New Roman',
        'size'   : 12}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
def problem1():
    print('Problem 1: ')
    data = np.loadtxt('crash.txt')
    m,n = data.shape
    
    mask = np.linspace(0,m-1,m)%2
    
    training_set = data[mask==0]
    validation_set = data[mask==1]
    
    L = 20
    
    phi_train = np.zeros((training_set.shape[0],L))
    phi_val = np.zeros((validation_set.shape[0],L))
    phi_data = np.zeros((data.shape[0],L))
    
    error_train = np.zeros((L,1))
    error_val = np.zeros((L,1))
    
    for i in range(L):
        phi_train[:,i] = training_set[:,0]**i
        phi_val[:,i] = validation_set[:,0]**i
        phi_data[:,i] = data[:,0]**i
    
    for l in range(L):
        a = np.dot(phi_train[:,:l+1].T,phi_train[:,:l+1])
        b = np.dot(phi_train[:,:l+1].T,training_set[:,1])
        w = np.linalg.solve(a, b)
        
        that_train = np.dot(phi_train[:,:l+1],w)
        that_val = np.dot(phi_val[:,:l+1],w)
        
        error_train[l] = LA.norm(training_set[:,1] - that_train)
        error_val[l] = LA.norm(validation_set[:,1] - that_val)
    
    l_optimal = np.argmin(error_val) 
    a = np.dot(phi_train[:,:l_optimal+1].T,phi_train[:,:l_optimal+1])
    b = np.dot(phi_train[:,:l_optimal+1].T,training_set[:,1])
    
    w_optimal = np.linalg.solve(a, b)
    
    pred = np.dot(phi_data[:,:l_optimal+1],w_optimal)
    
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10,6))
    
    ax[0].plot(np.linspace(1,L,L),error_train,'k',lw='2',label='Training set')
    ax[0].plot(np.linspace(1,L,L),error_val,'r',lw='2',label='Validation set')
    ax[0].set_xticks(np.linspace(1,L,L))
    ax[0].legend(loc=0)
    
    ax[1].plot(data[:,0],pred,'b',lw=2,label='Prediction')
    ax[1].scatter(training_set[:,0],training_set[:,1],color='r',label='Training set')
    ax[1].scatter(validation_set[:,0],validation_set[:,1],color='k',label='Validation set')
    ax[1].legend(loc=4)
    
    plt.show()


problem1()

#%%
def problem2():
    print('Problem 2: ')
    data = np.loadtxt('crash.txt')
    m,n = data.shape
    
    mask = np.linspace(0,m-1,m)%2
    
    training_set = data[mask==0]
    validation_set = data[mask==1]
    
    P = np.array([5,10,15,20,25])
    
    error_train = np.zeros((P.shape[0],1))
    error_val = np.zeros((P.shape[0],1))
    
    for k in range(P.shape[0]):
        L = P[k]
    
        phi_train = np.zeros((training_set.shape[0],L))
        phi_val = np.zeros((validation_set.shape[0],L))
        phi_data = np.zeros((data.shape[0],L))
            
        mean = np.linspace(0,60,L).reshape(1,-1)
        std = mean[0,1] - mean[0,0]
           
        phi_train = np.exp(-(training_set[:,0].reshape(-1,1) - mean)**2/(2*std**2))
        phi_val = np.exp(-(validation_set[:,0].reshape(-1,1) - mean)**2/(2*std**2))
        phi_data = np.exp(-(data[:,0].reshape(-1,1) - mean)**2/(2*std**2))
        
        a = np.dot(phi_train.T,phi_train)
        b = np.dot(phi_train.T,training_set[:,1])
        w = np.linalg.solve(a, b)
        
        that_train = np.dot(phi_train,w)
        that_val = np.dot(phi_val,w)
            
        error_train[k] = LA.norm(training_set[:,1] - that_train)
        error_val[k] = LA.norm(validation_set[:,1] - that_val)
    
    l_optimal = P[np.argmin(error_val)]
    
    print("L = ",l_optimal)
    
    mean = np.linspace(0,60,l_optimal).reshape(1,-1)
    std = mean[0,1] - mean[0,0]
       
    phi_train = np.exp(-(training_set[:,0].reshape(-1,1) - mean)**2/(2*std**2))
    phi_data = np.exp(-(data[:,0].reshape(-1,1) - mean)**2/(2*std**2))
    
    a = np.dot(phi_train.T,phi_train)
    b = np.dot(phi_train.T,training_set[:,1])
    w = np.linalg.solve(a, b)
    
    pred = np.dot(phi_data,w)
    
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10,6))
        
    ax[0].plot(P,error_train,'k',lw='2',label='Training set')
    ax[0].plot(P,error_val,'r',lw='2',label='Validation set')
    ax[0].set_xticks(P)
    ax[0].legend(loc=0)
    
    ax[1].plot(data[:,0],pred,'b',lw=2,label='Prediction')
    ax[1].scatter(training_set[:,0],training_set[:,1],color='r',label='Training set')
    ax[1].scatter(validation_set[:,0],validation_set[:,1],color='k',label='Validation set')
    ax[1].legend(loc=4)
    
    plt.show()

problem2()

#%%
def problem3():
    data = np.loadtxt('crash.txt')
    m,n = data.shape
    
    mask = np.linspace(0,m-1,m)%2
    
    training_set = data[mask==0]
    validation_set = data[mask==1]
    
    beta = 1/400
    alpha = np.logspace(-8,0,100)
    
    error_train = np.zeros((alpha.shape[0],1))
    error_val = np.zeros((alpha.shape[0],1))
    
    L = 50
    
    for k in range(alpha.shape[0]):
        
        phi_train = np.zeros((training_set.shape[0],L))
        phi_val = np.zeros((validation_set.shape[0],L))
        phi_data = np.zeros((data.shape[0],L))
            
        mean = np.linspace(0,60,L).reshape(1,-1) 
        std = mean[0,1] - mean[0,0]
           
        phi_train = np.exp(-(training_set[:,0].reshape(-1,1) - mean)**2/(2*std**2))
        phi_val = np.exp(-(validation_set[:,0].reshape(-1,1) - mean)**2/(2*std**2))
        phi_data = np.exp(-(data[:,0].reshape(-1,1) - mean)**2/(2*std**2))
        
        a = np.dot(phi_train.T,phi_train) + alpha[k]/beta*np.eye(L)
        b = np.dot(phi_train.T,training_set[:,1])
        w = np.linalg.solve(a, b)
        
        that_train = np.dot(phi_train,w)
        that_val = np.dot(phi_val,w)
            
        error_train[k] = LA.norm(training_set[:,1] - that_train)
        error_val[k] = LA.norm(validation_set[:,1] - that_val)
    
    alpha_optimal = alpha[np.argmin(error_val)]
    
    print('Best alpha = ', alpha_optimal)
    
    mean = np.linspace(0,60,L).reshape(1,-1)
    std = mean[0,1] - mean[0,0]
       
    phi_train = np.exp(-(training_set[:,0].reshape(-1,1) - mean)**2/(2*std**2))
    phi_data = np.exp(-(data[:,0].reshape(-1,1) - mean)**2/(2*std**2))
    
    a = np.dot(phi_train.T,phi_train) + alpha_optimal/beta*np.eye(L)
    b = np.dot(phi_train.T,training_set[:,1])
    w = np.linalg.solve(a, b)
    
    pred = np.dot(phi_data,w)
    
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,4))
        
    ax.plot(data[:,0],pred,'b',lw=2,label='Prediction')
    ax.scatter(training_set[:,0],training_set[:,1],color='r',label='Training set')
    ax.scatter(validation_set[:,0],validation_set[:,1],color='k',label='Validation set')
    ax.legend(loc=4)
    
    plt.show()

problem3()

#%%
def flower_to_float(s):
    d = {b'Iris-setosa':0.,b'Iris-versicolor':1.,b'Iris-virginica':2.}
    return d[s]

irises = np.loadtxt('iris.data',delimiter=',',converters={4:flower_to_float})

data = irises[:,:-1]
data = np.column_stack((np.ones(data.shape[0]),data))

labels = irises[:,-1].astype(int)

onehot = np.zeros((labels.shape[0], 3))
onehot[np.arange(labels.shape[0]), labels[:]] = 1

ind = np.arange(25)
masktrain = np.hstack((ind,ind+50,ind+100))
ind = np.arange(25,50)
masktest = np.hstack((ind,ind+50,ind+100))

xtrain = data[masktrain]
ytrain = onehot[masktrain]

xtest = data[masktest]
ytest = onehot[masktest]

alpha = 0.003

def f(w):
    prior = alpha*np.dot(w.T,w)
    
    num = 0.0
    for k in range(3):
        wk = w[k*5:k*5+5]
        temp = np.sum(ytrain[:,k]*np.dot(wk.T,xtrain.T))
        num = num + temp
    
    denom = 0.0 
    for k in range(3):
        wk = w[k*5:k*5+5]
        temp = np.dot(wk.T,xtrain.T)
        denom = denom + np.exp(temp)
    
    denom = np.sum(np.log(denom))
    
    return prior - (num - denom)

w_init = np.ones(15)
w_hat = minimize(f, w_init).x

w_hat = w_hat.reshape(3,-1)

z = np.dot(w_hat,xtest.T).T

s = np.exp(z)/((np.sum(np.exp(z),axis=1)).reshape(-1,1))

ypred = np.zeros_like(s)
ypred[np.arange(len(s)), s.argmax(1)] = 1

n_class1 = np.sum(ypred[0:25,0])
n_class2 = np.sum(ypred[25:50,1])
n_class3 = np.sum(ypred[50:,2])

accuracy = (n_class1 + n_class2 + n_class3)/ytest.shape[0]

print('Accuracy = ', accuracy)












