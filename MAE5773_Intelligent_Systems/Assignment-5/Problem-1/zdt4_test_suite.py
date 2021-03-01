from problem import Problem
from evolution import Evolution
import matplotlib.pyplot as plt
import math
import numpy as np

def f1(x):
    result = x[0]  
    return result

def f2(x):
    m = len(x)
    g = 1.0 + 10.0*(m - 1)
    for i in range(1,m):
        g += (x[i]**2 - 10.0*np.cos(4.0*np.pi*x[i]))
    
    h = 1.0 - np.sqrt(x[0]/g)
    
    result = g*h
    
    return result

problem = Problem(num_of_variables=10, objectives=[f1, f2], variables_range=[(-5, 5)], same_range=True, expand=False)
problem.variables_range[0] = (0,1)
print(problem.variables_range)
evo = Evolution(problem, mutation_param=5)
func = [i.objectives for i in evo.evolve()]

function1 = [i[0] for i in func]
function2 = [i[1] for i in func]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.plot(function1, function2,'bo')
plt.show()


np.save('zdt4', func)