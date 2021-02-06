#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:26:31 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

from genMesh1D import *

domain = [0,1]
n = 5      
mesh = genMesh1D(domain, n)

print(mesh.p)      
print(mesh.t)      