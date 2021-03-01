from nsga2.problem import Problem
from nsga2.evolution import Evolution
import matplotlib.pyplot as plt
import math
import numpy as np

#%%
def f1(x):
    result = 1.0 - np.exp(-4.0*x[0])*pow(np.sin(6.0*np.pi*x[0]),6)  
    return result

def f2(x):
    m = len(x)
    g1 = 0.0
    for i in range(1,m):
        g1 += x[i]/(m-1)
    
    g = 1.0 + 9.0*pow(g1,0.25)
    h = 1.0 - (x[0]/g)**2
    
    result = g*h
    
    return result

problem = Problem(num_of_variables=10, objectives=[f1, f2], variables_range=[(0, 1)], same_range=True, expand=False)
evo = Evolution(problem, mutation_param=5)
func = [i.objectives for i in evo.evolve()]

#%%
function1 = [i[0] for i in func]
function2 = [i[1] for i in func]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.plot(function1, function2,'bo')
plt.show()


np.save('zdt6', func)