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
    def __init__(self,nodeNumber):
        self.nodeNumber = nodeNumber        
    
    @staticmethod
    def get_distance(first, second, M):
        i = int(first.nodeNumber-1)
        j = int(second.nodeNumber-1)
        return M[i,j]
    
    @staticmethod
    def get_total_distance(coords, M):
        dist = 0
        for first, second in zip(coords[:-1], coords[1:]):
            dist += Coordinate.get_distance(first, second, M)
        
        dist += Coordinate.get_distance(coords[0], coords[-1], M)
        
        return dist
    

def tsplib(content, f=1, r=None, learning_plot=False):
    idx = content.index('DIMENSION:') + 1
    n = int(content[idx])
    idx = content.index('EDGE_WEIGHT_FORMAT:') + 1
    if content[idx] != 'FULL_MATRIX':
        return [], 0
    idx = content.index('EDGE_WEIGHT_SECTION') + 1
    inf = int(content[idx])
    data = []
    for i in range(n):
        if len(content) > idx + n:
            data.append(list(map(int, content[idx:idx + n])))
        else:
            return [], 0
        idx += n
    return data

if __name__ == '__main__':
    # Fill up the coordinates
    
    with open('br17.atsp', 'r') as fp:
        file_content = fp.read().split()
        data = tsplib(file_content, learning_plot=True)

    if isinstance(data[0], list):
        n = len(data)
        M = []
        for row in data:
            M.append([9999 if x == 9999 else x for x in row])
        
    M = np.array(M)
    N = M.shape[0]
    
    nodeNumber = np.zeros(N)
   
    coords = []
    for i in range(N): # 
        nodeNumber[i] = int(i+1)
        coords.append(Coordinate(nodeNumber[i]))
               
    # Simulated annleaning algorithm
    cost0 = Coordinate.get_total_distance(coords, M)
    
    Ts = np.sqrt(N) 
    T = np.sqrt(N) 
    factor = 0.999
    T_init = T
    iteration = 0
    stopping_iter = 5000
    stopping_temperature = 1e-12
    scheme = 1 # 1: node, 2: edge
    list_cost = []
    
    while T >= stopping_temperature and iteration < stopping_iter:
        
        list_cost.append(cost0)
        
        print(iteration, 'cost = ', cost0)
        
        T = T*factor
        
        for j in range(5):
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
            cost1 = Coordinate.get_total_distance(coords, M)
            
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

    data_solution = np.zeros((N,3))
    for i in range(N):
        data_solution[i,0] = int(coords[i].nodeNumber)
        
    
    fig = plt.figure(figsize=(6,5))
    ax1 = fig.add_subplot(111)
    
    ax1.semilogy(list_cost, 'k')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    fig.tight_layout()  
    plt.show()
    fig.savefig(f'p3i_{scheme}_{factor}'+'.pdf', dpi=300)
    fig.savefig(f'p3i_{scheme}_{factor}'+'.png', dpi=300)
    
    np.savez(f'p3_{scheme}_{factor}', solution = data_solution, list_cosyt = list_cost)
    
    #%%
    for coord in coords:
        print(coord.nodeNumber)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
