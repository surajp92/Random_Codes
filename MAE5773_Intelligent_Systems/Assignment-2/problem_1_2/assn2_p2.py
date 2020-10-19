"""
Created on Wed Sep  9 20:22:38 2020

@author: suraj
"""

import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']



class Coordinate:
    def __init__(self,nodeNumber, x, y):
        self.nodeNumber = nodeNumber
        self.x = x
        self.y = y
        
    
    @staticmethod
    def get_distance(a, b):
        return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
    
    @staticmethod
    def get_total_distance(coords):
        dist = 0
        for first, second in zip(coords[:-1], coords[1:]):
            dist += Coordinate.get_distance(first, second)
        
        dist += Coordinate.get_distance(coords[0], coords[-1])
        
        return dist
    

if __name__ == '__main__':
    # Fill up the coordinates
    
    data = np.genfromtxt('coord_101.txt')
    nodeNumber = data[:,0]
    cx = data[:,1]
    cy = data[:,2]
    N = data.shape[0]
#    cx = np.array([0.6606,0.9695,0.5906,0.2124,0.0398,0.1367,0.9536,0.6091,0.8767,0.8148,0.9500,0.6740,0.5029,0.8274,0.9697,0.5979,0.2184,0.7148,0.2395,0.2867])

#    cy = np.array([0.3876,0.7041,0.0213,0.3429,0.7471,0.5449,0.9464,0.1247,0.1636,0.8668,0.8200,0.3296,0.1649,0.3025,0.8192,0.9392,0.8191,0.4351,0.8646,0.6768])
    
    coords = []
    for i in range(N): # 
        coords.append(Coordinate(nodeNumber[i], cx[i], cy[i]))
    
    # Plot
    fig = plt.figure(figsize=(10,4.5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for first, second in zip(coords[:-1], coords[1:]):
        ax1.plot([first.x, second.x], [first.y, second.y], 'b')
    ax1.plot([coords[0].x, coords[-1].x], [coords[0].y, coords[-1].y], 'b')
    for c in coords:
        ax1.plot(c.x, c.y, 'ro')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    #plt.show()
        
    # Simulated annleaning algorithm
    cost0 = Coordinate.get_total_distance(coords)
    
    Ts = np.sqrt(N) 
    T = np.sqrt(N) 
    factor = 0.999
    T_init = T
    iteration = 0
    stopping_iter = 100000
    stopping_temperature = 1e-28
    scheme = 2 # 1: node, 2: edge
    list_cost = []
    
    while T >= stopping_temperature and iteration < stopping_iter:
        
        list_cost.append(cost0)
        
        print(iteration, 'cost = ', cost0)
        
        T = T*factor
        
        for j in range(1):
            # Exchange two coordinates and get a new neighbour solution
            
            r1, r2 = np.random.randint(0, len(coords), size = 2)
            
            if scheme == 1:
                temp = coords[r1]
                coords[r1] = coords[r2]
                coords[r2] = temp
            elif scheme == 2:
                l = np.random.randint(2, N - 1)
                i = np.random.randint(0, N - l)
                coords[i : (i + l)] = reversed(coords[i : (i + l)])
            

            
            # Get the new cost
            cost1 = Coordinate.get_total_distance(coords)
            
            if cost1 < cost0:
                # accept the new solution
                cost0 = cost1
            else:
                # accept the new (worse) solution with a given probability
                x = np.random.uniform()
                if x < np.exp((cost0 - cost1)/T):
                    # accept the new solution
                    cost0 = cost1
                else:
                    # do not accept the solution
                    if scheme == 1:
                        temp = coords[r1]
                        coords[r1] = coords[r2]
                        coords[r2] = temp
                    elif scheme == 2:
                        coords[i : (i + l)] = reversed(coords[i : (i + l)])

        
        iteration += 1
    
    print(cost0)                            
    # plot the result
    for first, second in zip(coords[:-1], coords[1:]):
        ax2.plot([first.x, second.x], [first.y, second.y], 'b')
    ax2.plot([coords[0].x, coords[-1].x], [coords[0].y, coords[-1].y], 'b')
    for c in coords:
        ax2.plot(c.x, c.y, 'ro')

    data_solution = np.zeros((N,3))
    for i in range(N):
        data_solution[i,0] = int(coords[i].nodeNumber)
        data_solution[i,1] = coords[i].x
        data_solution[i,2] = coords[i].y
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    fig.tight_layout()        
    plt.show()
    fig.savefig(f'p2s_{scheme}_{factor}'+'.pdf', dpi=300)
    fig.savefig(f'p2s_{scheme}_{factor}'+'.png', dpi=300)
    
    fig = plt.figure(figsize=(6,5))
    ax1 = fig.add_subplot(111)
    
    ax1.semilogy(list_cost, 'k')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    fig.tight_layout()  
    plt.show()
    fig.savefig(f'p2i_{scheme}_{factor}'+'.pdf', dpi=300)
    fig.savefig(f'p2i_{scheme}_{factor}'+'.png', dpi=300)
    
    np.savez(f'p2_{scheme}_{factor}', solution = data_solution, list_cosyt = list_cost)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
