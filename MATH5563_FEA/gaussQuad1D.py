#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 19:58:10 2021

@author: suraj
"""

import numpy as np
from gaussRef1D import *

def gaussQuad1D(vert, ng):
    w,x = gaussRef1D(ng)

    a = vert[0]
    b = vert[1]
    gw = (b-a)*w/2.0
    gx = (b-a)*x/2.0 + (a+b)/2.0
    
    return gw, gx


