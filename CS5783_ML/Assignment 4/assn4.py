#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:17:01 2019

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.optimize import minimize

from numpy.random import seed
seed(1)


#%%
# Define the kernel function
def kernel_se(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

def kernel_e(a, b, param):
    absdist = np.abs(a - b.reshape(-1,))
    return np.exp(-1.0 * (1/param) * absdist)

def gp_kernel_se(xtrain,ytrain,xtest,ytest,param):
    K_ss = kernel_se(xtest, xtest, param)
        
    # Apply the kernel function to our training points
    K = kernel_se(xtrain, xtrain, param)
    L = np.linalg.cholesky(K + (0.09569**2)*np.eye(len(xtrain)))
    
    # Compute the mean at our test points.
    K_s = kernel_se(xtrain, xtest, param)
    Lk = np.linalg.solve(L, K_s)
    mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((xtest.shape[0],))
    
    # Compute the standard deviation so we can plot it
    s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
    stdv = np.sqrt(s2)
    
    return mu, stdv

def gp_kernel_e(xtrain,ytrain,xtest,ytest,param):
    K_ss = kernel_e(xtest, xtest, param)
        
    # Apply the kernel function to our training points
    K = kernel_e(xtrain, xtrain, param)
    L = np.linalg.cholesky(K + (0.09569**2)*np.eye(len(xtrain)))
    
    # Compute the mean at our test points.
    K_s = kernel_e(xtrain, xtest, param)
    Lk = np.linalg.solve(L, K_s)
    mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((xtest.shape[0],))
    
    # Compute the standard deviation so we can plot it
    s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
    stdv = np.sqrt(s2)
    
    return mu, stdv
    
#%%
#def problem1():
data = np.loadtxt('crash.txt')
m,n = data.shape


mindata = np.min(data,axis=0)
data = data - mindata
maxdata = np.max(data,axis=0)
data = data/maxdata

folds = 5
cv_results_se = np.zeros((101,6))
cv_results_e = np.zeros((101,6))
params = np.linspace(0.01,0.5,101)
foldsize = int(data.shape[0]/folds)

random_indices = np.arange(data.shape[0])
np.random.shuffle(random_indices)

for j in range(params.shape[0]):
    param = params[j]
    cv_results_se[j,0] = param
    
    for i in range(5):
        randind_train = np.vstack((random_indices[:i*foldsize].reshape(-1,1),
                                   random_indices[(i+1)*foldsize:].reshape(-1,1)))
        
        randind_test = random_indices[i*foldsize:(i+1)*foldsize]
        
        xtrain = data[randind_train.reshape(-1,),0].reshape(-1,1)
        ytrain = data[randind_train.reshape(-1,),1].reshape(-1,1)
        
        xtest = data[randind_test.reshape(-1,),0].reshape(-1,1)
        ytest = data[randind_test.reshape(-1,),1].reshape(-1,1)
        
        mu_se,stdev_se = gp_kernel_se(xtrain,ytrain,xtest,ytest,param)
        mse = np.mean((np.square(mu_se - ytest.reshape(-1,))))
        cv_results_se[j,i+1] = mse
        
        mu_e,stdev_e = gp_kernel_e(xtrain,ytrain,xtest,ytest,param)
        mse = np.mean((np.square(mu_e - ytest.reshape(-1,))))
        cv_results_e[j,i+1] = mse

mean_mse_se = np.mean(cv_results_se[:,1:],axis=1)        
opt_param_se = params[np.argmin(mean_mse_se)]

mean_mse_e = np.mean(cv_results_e[:,1:],axis=1)        
opt_param_e = params[np.argmin(mean_mse_e)]


xtrain, ytrain = data[:,0].reshape(-1,1), data[:,1].reshape(-1,1)
xtest = np.linspace(0,1,101)
xtest = xtest.reshape(-1,1)

mu_se,stdev_se = gp_kernel_se(xtrain,ytrain,xtest,ytest,opt_param_se)
mu_e,stdev_e = gp_kernel_e(xtrain,ytrain,xtest,ytest,opt_param_e)

print('Squeared exponential')
print('Optimal sigma = ', format(opt_param_se, '.3f'))
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,4))
ax.plot(data[:,0],data[:,1],marker='o',label='True data')
ax.plot(xtest,mu_se,marker='o',label='Gaussian process')
ax.legend()
##plt.gca().fill_between(xtest.flat, mu-1*stdv, mu+1*stdv, color="#dddddd")
plt.show()

print('Exponential')
print('Optimal sigma = ', format(opt_param_e, '.3f'))
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,4))
ax.plot(data[:,0],data[:,1],marker='o',label='True data')
ax.plot(xtest,mu_e,marker='o',label='Gaussian process')
ax.legend()
##plt.gca().fill_between(xtest.flat, mu-1*stdv, mu+1*stdv, color="#dddddd")
plt.show()

#problem1()

#%%
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

test_images = np.array(test_images)
test_images = test_images.reshape(10000,-1)
test_images = test_images.astype(np.float64)


test_labels = np.array(test_labels)
test_labels = test_labels.reshape(10000,-1)

for q in range(4):
    # part 1
    if q == 0:
        K = 10
        random_indices = np.random.randint(0,10000,K)
        centroid = test_images[random_indices] #np.random.rand(10,784)*255
    
    # part 2
    if q == 1:
        K = 10
        centroid = np.zeros((K,test_images.shape[1]))
        centroid[0,:] = test_images[np.random.randint(0,10000,1)]
        for k in range(1,K):
            b = test_images - centroid[k-1,:]
            distkmpp = LA.norm(b,axis=1)**2
            centroid[k,:] = test_images[np.argmax(distkmpp)]
    
    #part 3
    if q == 2:
        K = 10
        centroid = np.zeros((K,test_images.shape[1]))
        for k in range(K):
            centroid[k,:] = test_images[test_labels.reshape(-1,)==k][0]
    
    # part 4
    if q == 3:
        K = 3
        centroid = np.zeros((K,test_images.shape[1]))
        
        centroid[0,:] = test_images[np.random.randint(0,10000,1)]
        for k in range(1,K):
            b = test_images - centroid[k-1,:]
            distkmpp = LA.norm(b,axis=1)**2
            centroid[k,:] = test_images[np.argmax(distkmpp)]
    
    centroid_temp =  np.zeros((K,test_images.shape[1]))
    dist_mat = np.zeros((test_images.shape[0],K))
    residual = 1
    n = 0
    
    while residual  > 0.001:
        for k in range(K):
            b = test_images - centroid[k,:]
            dist_mat[:,k] = LA.norm(b,axis=1)**2
        
        ind = np.argmin(dist_mat,axis=1)
    
        for k in range(K):
            temp = test_images[ind == k]
            #print(k, " ", temp.shape)
            if temp.size != 0:
                centroid_temp[k,:] = np.average(temp,axis=0)
        
        residual= LA.norm(centroid - centroid_temp)
        centroid[:,:] = centroid_temp[:,:]
        print('.',end =" ")
        #print(n, " res = ", residual)
        n = n+1
    
    J = 0
    for k in range(K):
        temp = test_images[ind == k]
        b = temp - centroid[k,:]
        J = J + LA.norm(b)**2
    
    print(n, " Iterations ", J )
    
    if q == 3:
        unique, counts = np.unique(ind, return_counts=True)

        fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(6,3))
        ax = axs.flat
        
        for k in range(K):
            ax[k].imshow(centroid[k].reshape(28,28))
            ax[k].set_xticks([])
            ax[k].set_yticks([])
        
        fig.tight_layout()
        print('Output for problem 2 part 4')
        plt.show()
        
        fig, axs = plt.subplots(nrows=3,ncols=2,figsize=(6,9))
        ax = axs.flat
        p = 0
        
        for i in range(3):
            for j in range(2):
                ax[p].imshow(test_images[ind==unique[i]][j].reshape(28,28))
                ax[p].set_xticks([])
                ax[p].set_yticks([])
                ax[p].set_title('Cluster '+str(i))
                p = p+1
        fig.tight_layout()
        print('Output for problem 2 part '+str(q+1))
        plt.show()
        
    else:        
        fig, axs = plt.subplots(nrows=2,ncols=5,figsize=(10,4))
        ax = axs.flat
        
        for k in range(K):
            ax[k].imshow(centroid[k].reshape(28,28))
            ax[k].set_xticks([])
            ax[k].set_yticks([])
        
        fig.tight_layout()
        print('Output for problem 2 part '+str(q+1))
        plt.show()

#%% problem 3
zvalue = np.array([0,1]) #Latent variable [0=Fair, 1=Loaded]
xvalue = np.array([1,2,3,4,5,6]) #Obervations

pxz = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])
pzz = np.array([[0.95, 0.05],[0.1,0.9]])                

Nsteps = 1000
xtrue = np.zeros(Nsteps)
ztrue = np.zeros(Nsteps)

ztrue[0] = 0
px = pxz[int(ztrue[0])]
xtrue[0] = np.random.choice(xvalue, 1, p=px)

for i in range(1,Nsteps):
   
   pz = pzz[int(ztrue[i-1])]
   ztrue[i] = np.random.choice(zvalue, 1, p=pz)
   
   px = pxz[int(ztrue[i])]
   xtrue[i] = np.random.choice(xvalue, 1, p=px)

#%%
# Transition Probabilities
ptransition = np.array(((0.95, 0.05), (0.1, 0.9)))
 
# Emission Probabilities
pemission = np.array(((1/6, 1/6, 1/6, 1/6, 1/6, 1/6), (0.1, 0.1, 0.1, 0.1, 0.1, 0.5)))

# Equal Probabilities for the initial distribution
initial_distribution = np.array((0.5,0.5))

def forward(x, a, b, initial_distribution):
    alpha = np.zeros((x.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, int(x[0]-1)]
    
    alpha[0, :] = alpha[0,:]/np.sum(alpha[0,:])
    
    for t in range(1, x.shape[0]):
        for j in range(a.shape[0]):
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, int(x[t]-1)]
        alpha[t,:]  = alpha[t,:]/np.sum(alpha[t,:])
        
    return alpha
 

def backward(x, a, b):
    beta = np.zeros((x.shape[0], a.shape[0]))
 
    # setting beta(T) = 1
    beta[-1] = np.ones((a.shape[0]))
    beta[-1,:]  = beta[-1,:]/np.sum(beta[-1,:])
 
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(x.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, int(x[t + 1]-1)]).dot(a[j, :])
        beta[t,:]  = beta[t,:]/np.sum(beta[t,:])
        
    return beta
 
alpha = forward(xtrue, ptransition, pemission, initial_distribution)

beta = backward(xtrue, ptransition, pemission)

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(16,4))
ax.plot(ztrue,'k',label='Actual loaded dice')
ax.plot(alpha[:,1],'r',label='Probability of loaded dice')
ax.legend(loc=2)
ax.set_xlabel('Time steps')
ax.set_title('Forward')
plt.show()

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(16,4))
ax.plot(ztrue,'k',label='Actual loaded dice')
ax.plot(beta[:,1],'r',label='Probability of loaded dice')
ax.legend(loc=2)
ax.set_xlabel('Time steps')
ax.set_title('Backward')
plt.show()

posterior = alpha*beta

posterior = posterior[:,:]/np.sum(posterior,axis=1,keepdims=True)
    

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(16,4))
ax.plot(ztrue,'k',label='Actual loaded dice')
ax.plot(posterior[:,1],'r',label='Probability of loaded dice')
ax.legend(loc=2)
ax.set_xlabel('Time steps')
ax.set_title('Posterior')
plt.show()