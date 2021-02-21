#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 22:38:06 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

from genMesh1D import *
from genFEM1D import *
from evalFEfun1D import *
from getErr1D import *

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True


domain = [2,3]
pd = 2
ng = 5

f = lambda x: np.sin(2.0*np.pi*x) + np.exp(-x)
df = lambda x: 2.0*np.pi*np.cos(2.0*np.pi*x) - np.exp(-x)

for i in range(6):
    n = int(5*(2**i))
    mesh = genMesh1D(domain, n)
    fem = genFEM1D(mesh, pd)
    
    uh = f(fem.p)
    
    errL2, errKL2 = getErr1D(uh, f, mesh, fem, ng, 0)
    errH1, errKH2 = getErr1D(uh, df, mesh, fem, ng, 1)
    
    print('#----------------------------------------#')
    print('n = %d' % (n))
    print('L2 error:  %5.3e' % errL2)
    print('H1 error:  %5.3e' % errH1)
    if i>=1:
        rateL2 = np.log(errL2_0/errL2)/np.log(2.0);
        rateH1 = np.log(errH1_0/errH1)/np.log(2.0);
        print('L2 order:  %5.3f' % rateL2)
        print('H1 order:  %5.3f' % rateH1)    
    
    errL2_0 = errL2
    errH1_0 = errH1
         