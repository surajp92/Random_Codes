# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script


"""

#-------------------------------------------------------------
#               Genetic Algorithm (GA)
#-------------------------------------------------------------
# To solve optimization problem (minimization) using GA.
#-------------------------------------------------------------
# Python version used: 2.6 / 2.7
#-------------------------------------------------------------


#-------------------------------------------------------------
# Step 1: Library Inclusion                             
#-------------------------------------------------------------
import random
import time
from copy import deepcopy
#import fitnessFunction as ff # Fitness Function and Parameters
import math as m
import numpy as np

#random.seed(1)

import matplotlib.pyplot as plt

startTime = time.time()

#-------------------------------------------------------------
# Step 2: Parameters
#-------------------------------------------------------------

# 2.1 GA Parameters
algoName    = "griewangks_own" # Algo Name
CR 	    = 0.8  	# Crossover Rate
MR 	    = 0.85      # Mutation Rate
RR      = 0.1  # retain rate, retain poor offsprings
alpha = 0.98
beta = 1.001

# 2.2 Global Parameters
iterations  = 100000       # Number of iterations
popSize     = 100       # Population Size(i.e Number of Chromosomes)
pop         = []        # Store Population with Fitness
maxFunEval  = 8000000    # Maximum allowable function evaluations
funEval	    = 0		# Count function evaluations
bestFitness = 999999999  # Store Best Fitness Value
bestChromosome = []     # Store Best Chromosome

#-------------------------------------------------------------
# Fitness Function parameters
#-------------------------------------------------------------
D       = 10   # Problem Dimension
LB      = -600 # Xi value Lower Bound
UB      = 600   # Xi value Size Upper Bound

#-------------------------------------------------------------
# Fitness Function
#-------------------------------------------------------------
def FitnessFunction(x):
    
    term1 = 0.0
    term2 = 1.0

    for i in range(D):
        term1 += (x[i]**2)/4000
    
    for i in range(D):
        term2 = term2*np.cos(x[i]/np.sqrt(i+1))
        
    result = term1 - term2 + 1.0
    
    return result
  
# 2.3 Result Saving Parameters
resultFileName="result"+algoName+".csv"

# 2.4 Stores Chromosome and its fitness collectively
class Individual:
    def __init__(self, C, F):
        self.chromosome=C
        self.fitness=F

# 2.5 Problem parameters
# Problem Parameters are defined in in fitnessFunction.py file



# Function 2: Generate Random Initial Population
def Init():
    global funEval
    for i in range (0, popSize):
        chromosome = []
        for j in range(0,D):
            chromosome.append(round(random.uniform(LB,UB),2))
#            chromosome.append(random.uniform(LB,UB))
            
        fitness = FitnessFunction(chromosome)
        funEval = funEval + 1
        newIndividual = Individual(chromosome,fitness)
        pop.append(newIndividual)
        

# Function 3: Remember Global BEST in the pop;
def MemoriseGlobalBest():
    global bestFitness,bestChromosome
    for p in pop:
        if p.fitness < bestFitness:
            bestFitness=p.fitness
            bestChromosome = deepcopy(p.chromosome)


# Function 4: Perform Crossover Operation
def Crossover():
    global funEval, CR, pop
    
#    print(len(pop))
#    l = random.sample(range(len(pop)), popSize)
#    pop = [pop[l[i]] for i in range(len(l))]
     
    for i in range(0,popSize):

        if (random.random() <= CR):

            # Choose two random indices
            i1,i2 = random.sample(range(0,popSize), 2)

            # Parents
            p1=deepcopy(pop[i1])
            p2=deepcopy(pop[i2])

            # Choose a crossover point 
            pts = np.sort(np.random.choice(np.arange(1,D-1),2,replace=False))

            # Generate new childs 
#            c1=p1.chromosome[0:pt1] + p2.chromosome[pt1:pt2] + p1.chromosome[pt2:]
#            c2=p2.chromosome[0:pt1] + p1.chromosome[pt1:pt2] + p2.chromosome[pt2:]
            
            c1 = p1.chromosome
            c2 = p2.chromosome
            
            for pt in pts:
                temp = c1[:pt] + c2[pt:]
                c2 = c2[:pt] + c1[pt:]
                c1 = deepcopy(temp)
                
            # Get the fitness of childs 
            c1Fitness=FitnessFunction(c1)
            funEval = funEval + 1
            c2Fitness=FitnessFunction(c2)
            funEval = funEval + 1

            # Select between parent and child
            if c1Fitness < p1.fitness:
                pop[i1].fitness=c1Fitness
                pop[i1].chromosome=c1
            
            elif (random.random() <= RR):
#                newIndividual = Individual(c1,c1Fitness)
#                pop.append(newIndividual)
                pop[i1].fitness=c1Fitness
                pop[i1].chromosome=c1
                
            if c2Fitness < p2.fitness:
                pop[i2].fitness=c2Fitness
                pop[i2].chromosome=c2
                
            elif (random.random() <= RR):
#                newIndividual = Individual(c2,c2Fitness)
#                pop.append(newIndividual)
                pop[i2].fitness=c2Fitness
                pop[i2].chromosome=c2
        


# Function 5: Perform Mutation Operation
def Mutation():
    global UB, LB, funEval
    for i in range(0,popSize):

        if (random.random() <= MR):
            
            # Choose random index
            r=random.randint(0,popSize-1)

            # Choose a parent
            p=deepcopy(pop[r])

            # Choose mutation point 
            pt = random.randint(0,D-1)    
            
            # Generate new childs
            c=deepcopy(p.chromosome)

            # Mutation
            c[pt] = round(random.uniform(LB,UB),2) #random.uniform(LB,UB) #

            #Get the fitness of childs
            cFitness=FitnessFunction(c)
            funEval = funEval + 1
            # Select between parent and child
            if cFitness < p.fitness:
                pop[r].fitness=cFitness
                pop[r].chromosome=c
  

#-------------------------------------------------------------
# Step 4: Start Program
#-------------------------------------------------------------
Init()
globalBest=pop[0].chromosome
globalBestFitness=pop[0].fitness
MemoriseGlobalBest()

# Saving Result
fp=open(resultFileName,"w");
fp.write("Iteration,Fitness,Chromosomes\n")

# for plotting the graph

count=0
a=[]

for i in range(0,iterations):
    Crossover()
    Mutation()
    MemoriseGlobalBest()
	
    if funEval >= maxFunEval:
        break

    if i%10==0:
        print("I:",i,"\t Fitness:", bestFitness)
        fp.write(str(i) + "," + str(bestFitness) + "," + str(bestChromosome) + "\n")
        #plotting with y-axis
        count=count+1
        #plotting with x-axis
        a.append(bestFitness)
        
        CR = random.random() #CR*alpha #max(CR*alpha, 0.25)
        MR = random.random() #min(beta*MR, 0.85) #
#        RR = min(beta*RR, 0.2)
#        if len(a) > 2:
#            if np.linalg.norm(a[-1] - a[-2]) < 1e-12:
#                break
    
print("I:",i+1,"\t Fitness:", bestFitness)
fp.write(str(i+1) + "," + str(bestFitness) + "," + str(bestChromosome))    
fp.close()

print("Done")
print("\nBestFitness:", bestFitness)
print("Best chromosome:", bestChromosome)
print("Total Function funEval: ",funEval)
print("Result is saved in", resultFileName)
print("Total Time Taken: ", round(time.time() - startTime,2), " sec\n")

#%%
plt.semilogy(range(count), a, 'ro-')
#plt.ylim([1e-2,1000])
plt.show()





