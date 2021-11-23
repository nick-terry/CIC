#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 21:30:01 2020

@author: nick
"""

import pickle
import numpy as np

with open('simDict_10_MATLAB.pck','rb') as f:
    hDict10 = pickle.load(f)

with open('simDict_5_MATLAB.pck','rb') as f:
    hDict5 = pickle.load(f)

def h10(x):
    
    x = np.asarray(x)
    
    # Convert the time-of-failure vector to a boolean vector of contingencies
    contingencies = getContingency(x,w=.2)
    
    n = x.shape[0]

    results = np.zeros(shape=(n,1))
    
    for i in range(n):
 
        result = hDict10[tuple(contingencies[i,:])]
        results[i] = result
        
    return results

def h5(x):
    
    x = np.asarray(x)
    
    # Convert the time-of-failure vector to a boolean vector of contingencies
    contingencies = getContingency(x,w=.2)
    
    n = x.shape[0]

    results = np.zeros(shape=(n,1))
    
    for i in range(n):
 
        result = hDict5[tuple(contingencies[i,:])]
        results[i] = result
        
    return results

def getContingency(x,w=.5):
    
    failed = 1 * (-w/2 <= x) * (x <= w/2)

    return failed

