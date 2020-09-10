"""
Created on Wed Sep  9 20:22:38 2020

@author: suraj
"""

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt


class Coordinate:
    def __init__(self, x, y):
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
    
    cx = np.array([0.6606,0.9695,0.5906,0.2124,0.0398,0.1367,0.9536,0.6091,0.8767,0.8148,0.9500,0.6740,0.5029,0.8274,0.9697,0.5979,0.2184,0.7148,0.2395,0.2867])
    cy = np.array([0.3876,0.7041,0.0213,0.3429,0.7471,0.5449,0.9464,0.1247,0.1636,0.8668,0.8200,0.3296,0.1649,0.3025,0.8192,0.9392,0.8191,0.4351,0.8646,0.6768])
    
    coords = []
    for i in range(20): # 
        coords.append(Coordinate(cx[i], cy[i]))
    
    # Plot
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for first, second in zip(coords[:-1], coords[1:]):
        ax1.plot([first.x, second.x], [first.y, second.y], 'b')
    ax1.plot([coords[0].x, coords[-1].x], [coords[0].y, coords[-1].y], 'b')
    for c in coords:
        ax1.plot(c.x, c.y, 'ro')
    
    #plt.show()
        
    # Simulated annleaning algorithm
    cost0 = Coordinate.get_total_distance(coords)

    T = 30 
    factor = 0.99
    T_init = T
    
    for i in range(1000):
        print(i, 'cost = ', cost0)
        
        T = T*factor
        
        for j in range(100):
            # Exchange two coordinates and get a new neighbour solution
            
            r1, r2 = np.random.randint(0, len(coords), size = 2)
            
            temp = coords[r1]
            coords[r1] = coords[r2]
            coords[r2] = temp
            
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
                    temp = coords[r1]
                    coords[r1] = coords[r2]
                    coords[r2] = temp
                            
    # plot the result
    for first, second in zip(coords[:-1], coords[1:]):
        ax2.plot([first.x, second.x], [first.y, second.y], 'b')
    ax2.plot([coords[0].x, coords[-1].x], [coords[0].y, coords[-1].y], 'b')
    for c in coords:
        ax2.plot(c.x, c.y, 'ro')
    
    plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
