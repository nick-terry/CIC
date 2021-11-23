#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 07:29:55 2020

@author: nick
"""

import scipy.stats as stat
import scipy.special as spc
import numpy as np
import multiprocessing as mp
import csv
import pickle
import itertools

# Import cem test version (stored locally, NOT a package!)
import simengine as se

def runSim(contingency):
    
    # Create a SimulationEngine for evaluating h(x)
    sim = se.SimulationEngine()
        
    x = np.array(contingency)
    
    # Pull out the branches that failed and simulate.
    branches = list((np.where(x==1)[0]+1).astype(int))
    return sim._simulate2(branches)

if __name__ == '__main__':
    
    np.random.seed(42)
    
    # enumerate all contingencies and create lookup table for the results
    numBr = 10
    l = [0,1]
    contingencies = [list(i) for i in itertools.product(l,repeat=numBr)]
    
    # x = [0,0,0,0,0,0,0,0,0,0]
    # z = runSim(x)
    
    # x = [1,0,1,0,0,0,0,1,0,1]
    # z = runSim(x)
    
    # Create multiprocessing pool w/ 28 nodes for Hyak cluster
    with mp.Pool(28) as _pool:
        result = _pool.map_async(runSim,
                                  contingencies,
                                  callback=lambda x : print('Done!'))
        result.wait()
        resultList = result.get()
    
    outageResults = [int(result[0]) for result in resultList]
    # convert to tuples for storing in dict
    contT = [tuple(c) for c in contingencies]
    simDict = dict(zip(contT,outageResults))
    
    # Save the estimates of failure probabilty to csv
    with open('simDict_10.pck','wb') as f:
        pickle.dump(simDict,f)
